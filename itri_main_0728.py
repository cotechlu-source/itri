#01
import numpy as np
import sys
import cv2
import os
from os import listdir
from os.path import join

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QLabel, QPushButton ,QGraphicsScene,QDialog,QLineEdit,QVBoxLayout ,QHBoxLayout
from PyQt5.QtCore import QThread, pyqtSignal ,Qt, QPoint, QRectF ,QTimer
from PyQt5.QtGui import QPixmap, QPen ,QImage

import matplotlib.path
import math
import utm
import exifread
import time
import yaml
import pathlib
# from osgeo import gdal, osr
from multiprocessing import Pool,shared_memory
import concurrent.futures
from itertools import accumulate
import logging
from datetime import datetime
# import requests




from scipy.linalg import solve

from itri_ui_0728 import Ui_MainWindow

# 0703
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 08/01
from watchdog.observers.polling import PollingObserver
class WatchdogLoader(QThread):
    image_loaded = pyqtSignal(str, object)  # 檔案路徑, 圖片物件

    def __init__(self, folder_to_watch):
        super().__init__()
        self.folder_to_watch = folder_to_watch
        self.loaded_files = set()
        self._observer = PollingObserver()
        self._stop_flag = False
        self.reading_active = False

    def run(self):
        class JPGHandler(FileSystemEventHandler):
            def __init__(self, outer):
                self.outer = outer

            def on_created(self, event):
                logging.info(f"偵測到新檔案，重新掃描資料夾：{self.outer.folder_to_watch}")
                self.outer.scan_folder()

        # 啟用 Handler
        event_handler = JPGHandler(self)
        self._observer.schedule(event_handler, self.folder_to_watch, recursive=False)
        self._observer.start()
        logging.info("啟動監控...")

        try:
            while not self._stop_flag:
                time.sleep(0.5)
        finally:
            logging.info("停止監控")
            self._observer.stop()
            self._observer.join()

    def stop(self):
        """手動停止執行緒用"""
        self._stop_flag = True

    def scan_folder(self):
        current_reading_state = self.reading_active
        for file in os.listdir(self.folder_to_watch):
            if file.lower().endswith(".jpg") and file not in self.loaded_files:
                full_path = os.path.join(self.folder_to_watch, file)
                try:
                    time.sleep(0.1)  # 防止寫入未完成
                    img = cv2.imread(full_path)
                    if img is not None:
                        self.loaded_files.add(file)
                        # 檢查旗標，決定是否發送訊號
                        if current_reading_state:
                            self.image_loaded.emit(full_path, img)
                            logging.info(f"已載入並處理：{file}")
                        else:
                            logging.info(f"偵測到圖片但不處理：{file}")
                    else:
                        logging.warning(f"讀圖失敗：{file}")
                except Exception as e:
                    logging.error(f"讀取異常：{file}，錯誤：{e}")

from dataclasses import dataclass, field
from typing import Optional , List


@dataclass
class ImageInfo:
    path: str
    image: object
    gps: List[float]
    direction: Optional[str] = None
    match_coordinate: List[int] = field(default_factory=lambda: [-1, -1])
    stitch_coordinate: List[int] = field(default_factory=lambda: [-1, -1])
    revise: bool = False
    distance: Optional[float] = None

    def __eq__(self, other):
        if not isinstance(other, ImageInfo):
            return NotImplemented
        return self.path == other.path

def calculate_direction(prev_gps, curr_gps):
    lat_diff = curr_gps[0] - prev_gps[0]
    lng_diff = curr_gps[1] - prev_gps[1]
    if lat_diff >= 0 and lng_diff >= 0:
        return 1
    elif lat_diff >= 0 and lng_diff < 0:
        return 2
    elif lat_diff < 0 and lng_diff < 0:
        return 3
    else:
        return 4


class ImageLoaderThread(QThread):
    # 定義信號
    image_loaded = pyqtSignal(object)  # 發出載入完成的圖片資訊
    progress_updated = pyqtSignal(int, int)  # 發出進度更新 (當前, 總數)
    loading_finished = pyqtSignal()  # 發出全部載入完成信號

    def __init__(self, folder, supported_ext):
        super().__init__()
        self.folder = folder
        self.supported_ext = supported_ext
        self.is_stopped = False

    def stop(self):
        self.is_stopped = True

    def run(self):
        # 取得所有圖片檔案
        image_files = sorted([
            os.path.join(self.folder, f) for f in os.listdir(self.folder)
            if os.path.splitext(f)[1].lower() in self.supported_ext
        ])

        total_files = len(image_files)

        for i, path in enumerate(image_files):
            if self.is_stopped:  # 檢查是否需要停止
                break

            # 載入圖片
            image = cv2.imread(path)
            if image is None:
                continue

            # 讀取GPS資訊
            gps = get_gps_coordinates(path)

            # 建立ImageInfo物件
            info = ImageInfo(path=path, image=image, gps=gps)

            # 發出信號，通知主線程有新圖片載入完成
            self.image_loaded.emit(info)

            # 發出進度更新信號
            self.progress_updated.emit(i + 1, total_files)

        # 發出載入完成信號
        self.loading_finished.emit()

class TemplateMatcherThread(QThread):
    list_signal = pyqtSignal(list,list)  # [x, y]
    def __init__(self, prev_img, curr_img, direction , prev_match_coordinate , curr_match_coordinate, curr_revise, conf):
        super().__init__()
        self.prev_img = prev_img
        self.curr_img = curr_img
        self.direction = direction
        self.match_coordinate = prev_match_coordinate  # (x, y, w, h)
        self.revise_coordinate = curr_match_coordinate
        self.revise_flag = curr_revise
        self.conf = conf
        self.scope_flag = False
    def run(self):
        logging.info(f"[MatchThread] 啟動比對：方向={self.direction}, 使用模板起點={self.match_coordinate}")
        match_scope_offset = 100
        methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF','cv2.TM_SQDIFF_NORMED']
        method_select = 1
        meth = methods[method_select]
        method = eval(meth)
        img_h, img_w = self.curr_img.shape[:2]
        temp_up_p1 = [self.conf['template_area']['temp_up']['x1'], self.conf['template_area']['temp_up']['y1']]
        temp_up_p2 = [self.conf['template_area']['temp_up']['x2'], self.conf['template_area']['temp_up']['y2']]
        temp_up_h = temp_up_p2[1] - temp_up_p1[1]
        temp_up_w = temp_up_p2[0] - temp_up_p1[0]
        temp_down_p1 = [self.conf['template_area']['temp_down']['x1'], self.conf['template_area']['temp_down']['y1']]
        temp_down_p2 = [self.conf['template_area']['temp_down']['x2'], self.conf['template_area']['temp_down']['y2']]
        temp_down_h = temp_down_p2[1] - temp_down_p1[1]
        temp_down_w = temp_down_p2[0] - temp_down_p1[0]
        temp_left_p1 = [self.conf['template_area']['temp_left']['x1'], self.conf['template_area']['temp_left']['y1']]
        temp_left_p2 = [self.conf['template_area']['temp_left']['x2'], self.conf['template_area']['temp_left']['y2']]
        temp_left_h = temp_left_p2[1] - temp_left_p1[1]
        temp_left_w = temp_left_p2[0] - temp_left_p1[0]
        temp_right_p1 = [self.conf['template_area']['temp_right']['x1'], self.conf['template_area']['temp_right']['y1']]
        temp_right_p2 = [self.conf['template_area']['temp_right']['x2'], self.conf['template_area']['temp_right']['y2']]
        temp_right_h = temp_right_p2[1] - temp_right_p1[1]
        temp_right_w = temp_right_p2[0] - temp_right_p1[0]

        self.prev_img = cv2.cvtColor(self.prev_img, cv2.COLOR_BGR2GRAY)
        self.curr_img = cv2.cvtColor(self.curr_img, cv2.COLOR_BGR2GRAY)

        if self.direction == 'up':
            if self.revise_flag and self.revise_coordinate not in (None, [-1, -1]):
                logging.info("[MatchThread] 手動比對模式啟動，跳過 matchTemplate")
                top_left = self.revise_coordinate
                stitch_offset = [top_left[0] - temp_up_p1[0], top_left[1] - temp_up_p1[1]]
                self.list_signal.emit(top_left, stitch_offset)
                return
            temp_img = self.curr_img[temp_up_p1[1]:temp_up_p2[1], temp_up_p1[0]:temp_up_p2[0]]
            # 0821 多一個決定要不要限縮範圍的開關
            if self.scope_flag == True:
                self.matching_scope = [0, 0], [img_w, temp_up_p1[1]]
                if self.match_coordinate not in (None, [-1, -1]):
                    self.matching_scope = ([self.match_coordinate[0] - match_scope_offset,self.match_coordinate[1] - match_scope_offset],
                                           [self.match_coordinate[0] + temp_up_w + match_scope_offset,self.match_coordinate[1] + temp_up_h + match_scope_offset])
            else:
                # 不限範圍，比整張圖
                self.matching_scope = [0, 0], [img_w, img_h]
            (x1, y1), (x2, y2) = self.matching_scope
            # 邊界修正，避免比對範圍超出影像
            x1 = max(0, min(x1, img_w))
            y1 = max(0, min(y1, img_h))
            x2 = max(0, min(x2, img_w))
            y2 = max(0, min(y2, img_h))
            match_img = self.prev_img[y1:y2, x1:x2]
            # 0619 切圖導致比對範圍小於模板大小
            if match_img.shape[0] < temp_img.shape[0] or match_img.shape[1] < temp_img.shape[1]:
                # fallback：回到預設大範圍
                logging.warning("[MatchThread] 範圍太小，使用 fallback 大範圍比對")
                self.matching_scope = [0, 0], [img_w, temp_up_p1[1]]
                (x1, y1), (x2, y2) = self.matching_scope
                x1 = max(0, min(x1, img_w))
                y1 = max(0, min(y1, img_h))
                x2 = max(0, min(x2, img_w))
                y2 = max(0, min(y2, img_h))
                match_img = self.prev_img[y1:y2, x1:x2]
            # 此處保證 match_img now 足夠大（因為 fallback）
            res = cv2.matchTemplate(match_img, temp_img, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            top_left = max_loc
            top_left = [top_left[0] + self.matching_scope[0][0], top_left[1] + self.matching_scope[0][1]]
            stitch_offset = [top_left[0]-temp_up_p1[0],top_left[1]-temp_up_p1[1]]
            self.list_signal.emit(top_left,stitch_offset)

        if self.direction == 'down':
            if self.revise_flag and self.revise_coordinate not in (None, [-1, -1]):
                logging.info("[MatchThread] 手動比對模式啟動，跳過 matchTemplate")
                top_left = self.revise_coordinate
                stitch_offset = [top_left[0] - temp_down_p1[0], top_left[1] - temp_down_p1[1]]
                self.list_signal.emit(top_left, stitch_offset)
                return
            temp_img = self.curr_img[temp_down_p1[1]:temp_down_p2[1], temp_down_p1[0]:temp_down_p2[0]]
            # 0821 多一個決定要不要限縮範圍的開關
            if self.scope_flag == True:
                self.matching_scope = [0, temp_down_p2[1]], [img_w, img_h]
                if self.match_coordinate not in (None, [-1, -1]):
                    self.matching_scope = ([self.match_coordinate[0] - match_scope_offset,
                                            self.match_coordinate[1] - match_scope_offset],
                                           [self.match_coordinate[0] + temp_down_w + match_scope_offset,
                                            self.match_coordinate[1] + temp_down_h + match_scope_offset])
            else:
                self.matching_scope = [0, 0], [img_w, img_h]
            (x1, y1), (x2, y2) = self.matching_scope
            # 邊界修正，避免比對範圍超出影像
            x1 = max(0, min(x1, img_w))
            y1 = max(0, min(y1, img_h))
            x2 = max(0, min(x2, img_w))
            y2 = max(0, min(y2, img_h))
            match_img = self.prev_img[y1:y2, x1:x2]
            # 0619 切圖導致比對範圍小於模板大小
            if match_img.shape[0] < temp_img.shape[0] or match_img.shape[1] < temp_img.shape[1]:
                # fallback：回到預設大範圍
                logging.warning("[MatchThread] 範圍太小，使用 fallback 大範圍比對")
                self.matching_scope = [0, 0], [img_w, temp_up_p1[1]]
                (x1, y1), (x2, y2) = self.matching_scope
                x1 = max(0, min(x1, img_w))
                y1 = max(0, min(y1, img_h))
                x2 = max(0, min(x2, img_w))
                y2 = max(0, min(y2, img_h))
                match_img = self.prev_img[y1:y2, x1:x2]
            # 此處保證 match_img now 足夠大（因為 fallback）
            res = cv2.matchTemplate(match_img, temp_img, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            top_left = max_loc
            top_left = [top_left[0] + self.matching_scope[0][0], top_left[1] + self.matching_scope[0][1]]
            stitch_offset = [top_left[0] - temp_down_p1[0], top_left[1] - temp_down_p1[1]]
            self.list_signal.emit(top_left,stitch_offset)
        if self.direction == 'left':
            if self.revise_flag and self.revise_coordinate not in (None, [-1, -1]):
                logging.info("[MatchThread] 手動比對模式啟動，跳過 matchTemplate")
                top_left = self.revise_coordinate
                stitch_offset = [top_left[0] - temp_left_p1[0], top_left[1] - temp_left_p1[1]]
                self.list_signal.emit(top_left, stitch_offset)
                return
            temp_img = self.curr_img[temp_left_p1[1]:temp_left_p2[1], temp_left_p1[0]:temp_left_p2[0]]
            # 0821 多一個決定要不要限縮範圍的開關
            if self.scope_flag == True:
                self.matching_scope = [0, 0], [temp_left_p1[0], img_h]
                if self.match_coordinate not in (None, [-1, -1]):
                    self.matching_scope = ([self.match_coordinate[0] - match_scope_offset,
                                            self.match_coordinate[1] - match_scope_offset],
                                           [self.match_coordinate[0] + temp_left_w + match_scope_offset,
                                            self.match_coordinate[1] + temp_left_h + match_scope_offset])
            else:
                self.matching_scope = [0, 0], [img_w, img_h]
            (x1, y1), (x2, y2) = self.matching_scope
            # 邊界修正，避免比對範圍超出影像
            x1 = max(0, min(x1, img_w))
            y1 = max(0, min(y1, img_h))
            x2 = max(0, min(x2, img_w))
            y2 = max(0, min(y2, img_h))
            match_img = self.prev_img[y1:y2, x1:x2]
            # 0619 切圖導致比對範圍小於模板大小
            if match_img.shape[0] < temp_img.shape[0] or match_img.shape[1] < temp_img.shape[1]:
                # fallback：回到預設大範圍
                logging.warning("[MatchThread] 範圍太小，使用 fallback 大範圍比對")
                self.matching_scope = [0, 0], [img_w, temp_up_p1[1]]
                (x1, y1), (x2, y2) = self.matching_scope
                x1 = max(0, min(x1, img_w))
                y1 = max(0, min(y1, img_h))
                x2 = max(0, min(x2, img_w))
                y2 = max(0, min(y2, img_h))
                match_img = self.prev_img[y1:y2, x1:x2]
            # 此處保證 match_img now 足夠大（因為 fallback）
            res = cv2.matchTemplate(match_img, temp_img, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            top_left = max_loc
            top_left = [top_left[0] + self.matching_scope[0][0], top_left[1] + self.matching_scope[0][1]]
            stitch_offset = [top_left[0] - temp_left_p1[0], top_left[1] - temp_left_p1[1]]
            self.list_signal.emit(top_left,stitch_offset)
        if self.direction == 'right':
            if self.revise_flag and self.revise_coordinate not in (None, [-1, -1]):
                logging.info("[MatchThread] 手動比對模式啟動，跳過 matchTemplate")
                top_left = self.revise_coordinate
                stitch_offset = [top_left[0] - temp_right_p1[0], top_left[1] - temp_right_p1[1]]
                self.list_signal.emit(top_left, stitch_offset)
                return
            temp_img = self.curr_img[temp_right_p1[1]:temp_right_p2[1], temp_right_p1[0]:temp_right_p2[0]]
            # 0821 多一個決定要不要限縮範圍的開關
            if self.scope_flag == True:
                self.matching_scope = [temp_right_p2[0], 0], [img_w, img_h]
                if self.match_coordinate not in (None, [-1, -1]):
                    self.matching_scope = ([self.match_coordinate[0] - match_scope_offset,
                                            self.match_coordinate[1] - match_scope_offset],
                                           [self.match_coordinate[0] + temp_right_w + match_scope_offset,
                                            self.match_coordinate[1] + temp_right_h + match_scope_offset])
            else:
                self.matching_scope = [0, 0], [img_w, img_h]
            (x1, y1), (x2, y2) = self.matching_scope
            # 邊界修正，避免比對範圍超出影像
            x1 = max(0, min(x1, img_w))
            y1 = max(0, min(y1, img_h))
            x2 = max(0, min(x2, img_w))
            y2 = max(0, min(y2, img_h))
            match_img = self.prev_img[y1:y2, x1:x2]
            # 0619 切圖導致比對範圍小於模板大小
            if match_img.shape[0] < temp_img.shape[0] or match_img.shape[1] < temp_img.shape[1]:
                # fallback：回到預設大範圍
                logging.warning("[MatchThread] 範圍太小，使用 fallback 大範圍比對")
                self.matching_scope = [0, 0], [img_w, temp_up_p1[1]]
                (x1, y1), (x2, y2) = self.matching_scope
                x1 = max(0, min(x1, img_w))
                y1 = max(0, min(y1, img_h))
                x2 = max(0, min(x2, img_w))
                y2 = max(0, min(y2, img_h))
                match_img = self.prev_img[y1:y2, x1:x2]
            # 此處保證 match_img now 足夠大（因為 fallback）
            res = cv2.matchTemplate(match_img, temp_img, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            top_left = max_loc
            top_left = [top_left[0] + self.matching_scope[0][0], top_left[1] + self.matching_scope[0][1]]
            stitch_offset = [top_left[0] - temp_right_p1[0], top_left[1] - temp_right_p1[1]]
            self.list_signal.emit(top_left,stitch_offset)

# from osgeo import gdal, osr
class StitchThread(QThread):
    result_signal = pyqtSignal(np.ndarray)  # 拼接完的圖 emit 給主線程

    def __init__(self, path_list,gps_list, stitch_coordinate_list):
        super().__init__()
        self.path_list = path_list
        self.gps_list = gps_list
        self.stitch_coordinate_list = stitch_coordinate_list

    def run(self):
        # 拼接時間戳
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_list = []

        for path in self.path_list:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                logging.error(f"[錯誤] 無法讀取圖片：{path}")
                continue
            image_list.append(img)

        if len(image_list) != len(self.stitch_coordinate_list):
            logging.error("[錯誤] 圖片數量與座標數量不符")
            return

        half_width = image_list[0].shape[1] // 2  # 寬度的一半
        half_height = image_list[0].shape[0] // 2  # 高度的一半

        # === 新增：GSD變化量分析 ===
        gsd_analysis_start = time.time()
        logging.info("[GSD變化量檢查] 開始分析每組相鄰圖片的GSD")
        gsd_values = []
        
        # 計算每組相鄰圖片的GSD
        for i in range(len(self.gps_list) - 1):
            if i < len(self.stitch_coordinate_list):
                gps_pair = [self.gps_list[i], self.gps_list[i + 1]]
                stitch_coord = self.stitch_coordinate_list[i]
                
                try:
                    gsd = calculate_meters_per_pixel(gps_pair[0], gps_pair[1], stitch_coord)
                    gsd_values.append(gsd)
                    logging.info(f"[GSD檢查] 第{i}→{i+1}組: GPS1={gps_pair[0]}, GPS2={gps_pair[1]}, Stitch_coord={stitch_coord}, GSD={gsd:.6f}m")
                except Exception as e:
                    logging.warning(f"[GSD檢查] 第{i}→{i+1}組計算失敗: {e}")
        
        # 分析GSD變化量
        if len(gsd_values) > 0:
            min_gsd = min(gsd_values)
            max_gsd = max(gsd_values)
            avg_gsd = sum(gsd_values) / len(gsd_values)
            gsd_range = max_gsd - min_gsd
            gsd_variation = (gsd_range / avg_gsd) * 100 if avg_gsd > 0 else 0
            
            logging.info(f"[GSD統計] 總共{len(gsd_values)}組數據")
            logging.info(f"[GSD統計] 最小值: {min_gsd:.6f}m")
            logging.info(f"[GSD統計] 最大值: {max_gsd:.6f}m")
            logging.info(f"[GSD統計] 平均值: {avg_gsd:.6f}m")
            logging.info(f"[GSD統計] 變化範圍: {gsd_range:.6f}m")
            logging.info(f"[GSD統計] 變化率: {gsd_variation:.2f}%")
            
            # 使用平均GSD
            m_per_pixel = avg_gsd
            
            if gsd_variation > 20:
                logging.warning(f"[GSD警告] GSD變化率較大({gsd_variation:.2f}%)，可能影響拼接精度")
        else:
            logging.warning("[GSD檢查] 無法計算任何GSD值，使用最後兩點計算")
            # 如果沒有有效GSD值，回退到原方法
            m_per_pixel = calculate_meters_per_pixel(self.gps_list[-2], self.gps_list[-1], self.stitch_coordinate_list[-1])
        
        gsd_analysis_time = time.time() - gsd_analysis_start
        logging.info(f"[GSD分析] 完成，耗時: {gsd_analysis_time:.3f}秒")
        # === GSD變化量分析結束 ===


        # 拼接函式 (透明)
        final_img, _ , center_coordinate_list = stitch_seg_images_transparent(self.stitch_coordinate_list, image_list)
        png_path = f'./data/{time_str}.png'
        cv2.imwrite(png_path, final_img)
        # resize_img = cv2.resize(final_img, (final_img.shape[1] // 4, final_img.shape[0] // 4))
        resize_img = cv2.resize(final_img, (final_img.shape[1], final_img.shape[0]))

        # 07/15測試 自動計算座標GPS
        # m_per_pixel = calculate_meters_per_pixel(self.gps_list[-2],self.gps_list[-1],self.last_match_coordinate)
        bearing = calculate_bearing(self.gps_list[-2],self.gps_list[-1])
        new_lat, new_lng = calculate_new_latlng_coordinate(self.gps_list[-1][0], self.gps_list[-1][1], [half_width, -half_height], m_per_pixel,bearing)
        right_top_gps =[new_lat,new_lng]
        new_lat, new_lng = calculate_new_latlng_coordinate(self.gps_list[-1][0], self.gps_list[-1][1], [-half_width, -half_height], m_per_pixel,bearing)
        left_top_gps = [new_lat, new_lng]

        logging.info(f'GSD(m) = {m_per_pixel} , 經緯角度 = {bearing} , 右上GPS = {right_top_gps} , 左上GPS = {left_top_gps}')

        resized_path = f'./resize_img.png'
        cv2.imwrite(resized_path, resize_img)

        # 07/11測試
        '''
        png = gdal.Open(f'./resize_img.png')
        width = png.RasterXSize
        height = png.RasterYSize

        center_coordinate_list = [[x // 4, y // 4] for [x, y] in center_coordinate_list]

        pixel_pts = np.array([
            center_coordinate_list[0],  # 下面中心
            np.array(center_coordinate_list[-1]) + np.array([half_width//4, -half_height//4]), # 右上
            np.array(center_coordinate_list[-1]) + np.array([-half_width//4, -half_height//4])  # 左上
        ])
        geo_pts = np.array([
            [self.gps_list[0][1], self.gps_list[0][0]],  # 對應經緯度1
            [right_top_gps[1], right_top_gps[0]],  # 對應經緯度2
            [left_top_gps[1],  left_top_gps[0]]  # 對應經緯度3
        ])
        logging.info(f'pixel_pts = {pixel_pts}')
        logging.info(f'geo_pts = {geo_pts}')
        A = np.vstack([pixel_pts.T, np.ones(3)]).T  # 每行 [x, y, 1]
        lons = geo_pts[:, 0]
        lats = geo_pts[:, 1]
        a, b, c = solve(A, lons)  # 解 a, b, c
        d, e, f = solve(A, lats)  # 解 d, e, f
        geotransform = (c, a, b, f, d, e)
        logging.info(f'geotransform = {geotransform}')

        driver = gdal.GetDriverByName('GTiff')
        geotiff = driver.Create(f'./data/{time_str}.tiff', width, height, png.RasterCount, gdal.GDT_Byte)
        geotiff.SetGeoTransform(geotransform)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        geotiff.SetProjection(srs.ExportToWkt())
        for i in range(1, png.RasterCount + 1):
            geotiff.GetRasterBand(i).WriteArray(png.GetRasterBand(i).ReadAsArray())
        geotiff.FlushCache()
        geotiff = None
        '''
        # geo_dict = {'geotransform': geotransform}
        # with open(f"/data/{time_str}.json", "w") as file:
        #   json.dump(geo_dict, file)
        # requests.post('http://localhost:50080/crater_recognize', data={'image_name': time_str})`

        # 發送訊號給主線程（縮圖或完整圖）
        self.result_signal.emit(resize_img)

class LatLngDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("輸入經緯度")
        self.setFixedSize(300, 150)

        self.lat_input = QLineEdit()
        self.lng_input = QLineEdit()

        layout = QVBoxLayout()

        lat_layout = QHBoxLayout()
        lat_layout.addWidget(QLabel("緯度："))
        lat_layout.addWidget(self.lat_input)

        lng_layout = QHBoxLayout()
        lng_layout.addWidget(QLabel("經度："))
        lng_layout.addWidget(self.lng_input)

        button_layout = QHBoxLayout()
        ok_button = QPushButton("確定")
        cancel_button = QPushButton("取消")
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)

        layout.addLayout(lat_layout)
        layout.addLayout(lng_layout)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def get_values(self):
        return self.lat_input.text(), self.lng_input.text()

#----------10/25-----讀經緯度，篩選範圍內影像
import re
def get_gps_coordinates(image_path):
    """
    自動判斷使用EXIF或檔名來讀取GPS座標
    優先使用EXIF，如果沒有GPS資訊則使用檔名解析
    """
    try:
        # 先嘗試從EXIF讀取GPS資訊
        coord = read_exif(image_path)
        if coord:  # 如果成功讀取到座標
            print(f"使用EXIF讀取GPS座標: {coord}")
            return coord
    except Exception as e:
        print(f"EXIF讀取失敗: {e}")

    try:
        # 如果EXIF讀取失敗，則使用檔名解析
        coord = read_gps(image_path)
        print(f"使用檔名讀取GPS座標: {coord}")
        return coord
    except Exception as e:
        print(f"檔名解析也失敗: {e}")
        return None
def read_exif(image_path):
    """讀取圖片EXIF中的GPS資訊"""
    try:
        with open(image_path, 'rb') as img_file:
            exifread_data = exifread.process_file(img_file)

            latitude = None
            longitude = None

            # 尋找GPS資訊
            for tag in exifread_data.keys():
                if tag == 'GPS GPSLatitude':
                    latitude = exifread_data[tag]
                if tag == 'GPS GPSLongitude':
                    longitude = exifread_data[tag]

            # 如果沒有找到GPS資訊，回傳None
            if latitude is None or longitude is None:
                return None

            # 處理緯度
            latitude_values = latitude.values
            latitude_deg = latitude_values[0]
            latitude_min = latitude_values[1]
            latitude_sec = latitude_values[2]
            latitude_sec_str = str(latitude_sec)

            if '/' in latitude_sec_str:
                numerator, denominator = map(int, latitude_sec_str.split('/'))
                latitude_sec_float = numerator / denominator
            else:
                latitude_sec_float = float(latitude_sec_str)

            # 處理經度
            longitude_values = longitude.values
            longitude_deg = longitude_values[0]
            longitude_min = longitude_values[1]
            longitude_sec = longitude_values[2]
            longitude_sec_str = str(longitude_sec)

            if '/' in longitude_sec_str:
                numerator, denominator = map(int, longitude_sec_str.split('/'))
                longitude_sec_float = numerator / denominator
            else:
                longitude_sec_float = float(longitude_sec_str)

            # 計算最終座標
            coord_longitude = longitude_deg + (longitude_min / 60) + (longitude_sec_float / 3600)
            coord_latitude = latitude_deg + (latitude_min / 60) + (latitude_sec_float / 3600)

            coord = [coord_latitude, coord_longitude]
            return coord

    except Exception as e:
        return None
def read_gps(image_path):
    """從檔名解析GPS座標"""
    import re
    import os

    filename = os.path.basename(image_path)
    filename_without_ext = os.path.splitext(filename)[0]
    pattern = r'^(\d{14})_([+-]?\d+\.?\d*)_([+-]?\d+\.?\d*)$'
    match = re.match(pattern, filename_without_ext)

    if not match:
        raise ValueError(f"檔名格式不正確：{filename}。")

    timestamp = match.group(1)  # 時間戳記（如果需要的話）
    latitude = float(match.group(2))  # 緯度
    longitude = float(match.group(3))  # 經度

    # 回傳格式與原本函式相同：[緯度, 經度]
    coord = [latitude, longitude]
    return coord

def func_range(coord,range_a,range_b,range_c,range_d):
    # 定義四個座標點（依照你的需求調整座標）
    points = np.array([range_a, range_b, range_c, range_d])
    # 10/25 要確認順序是否影響範圍
    # p[1]代表y值排序、p[0]代表x值排序
    sorted_points = sorted(points, key=lambda p: (p[1], p[0]))
    left_top = sorted_points[0]
    # 確認最左上的點後，計算其他點角度，按順時間順序排列
    sorted_points = sorted(points, key=lambda p: np.arctan2(p[1] - left_top[1], p[0] - left_top[0]))
    # 建立多邊形的路徑
    polygon_path = matplotlib.path.Path(sorted_points)
    # 定義你要檢查的第五個點
    target_point = np.array(coord)
    # 檢查點是否在多邊形內
    is_within = polygon_path.contains_point(target_point)

    if is_within:
        return True
    else:
        return False
#----------11/05-----算GPS經緯
def latlng_to_utm(lat, lng):
    """將經緯度轉換為 UTM 坐標，並返回 X、Y、區域編號及字母"""
    utm_x, utm_y, zone_number, zone_letter = utm.from_latlon(lat, lng)
    return utm_x, utm_y, zone_number, zone_letter
def calculate_meters_per_pixel(point_1, point_2, offset):
    """計算每 pixel 的實際距離"""
    lat1, lng1 = point_1
    lat2, lng2 = point_2
    pixel_offset_x, pixel_offset_y = offset

    # 將經緯度轉為 UTM 坐標
    utm_x1, utm_y1, zone_number, zone_letter = latlng_to_utm(lat1, lng1)
    utm_x2, utm_y2, _, _ = latlng_to_utm(lat2, lng2)

    # 計算 UTM 坐標下的實際距離
    distance_m = math.sqrt((utm_x2 - utm_x1) ** 2 + (utm_y2 - utm_y1) ** 2)

    # 計算像素距離
    pixel_distance = math.sqrt(pixel_offset_x ** 2 + pixel_offset_y ** 2)

    # 計算每 pixel 的實際距離
    meters_per_pixel = distance_m / pixel_distance
    return meters_per_pixel
def calculate_bearing(point_1, point_2):
    """計算從第一個點到第二個點的方向角度"""
    lat1, lng1 = point_1
    lat2, lng2 = point_2
    # 將經緯度轉換為弧度
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lng_rad = math.radians(lng2 - lng1)

    # 計算方向角度
    x = math.sin(delta_lng_rad) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - \
        math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lng_rad)

    initial_bearing = math.atan2(x, y)
    # 將角度轉換為度
    initial_bearing = math.degrees(initial_bearing)

    # 將方向角度轉換為 0 到 360 度
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing
def calculate_new_latlng_coordinate(lat, lng, gps_xy_offset, meters_per_pixel, bearing):
    # 因為gps_crop_offset的xy座標，是基於正上方為bearing而非正北方
    """利用方位角和偏移量一起計算新的 UTM 座標，並將其轉換為經緯度"""
    utm_x, utm_y, zone_number, zone_letter = utm.from_latlon(lat, lng)

    # 07/15
    # 計算移動方向的角度，以方位角和 x, y 偏移共同決定
    # bearing是順時針旋轉，math.atan2 計算出的角度是逆時針旋轉
    move_angle = math.radians(360-bearing) + math.atan2(-gps_xy_offset[1], gps_xy_offset[0])
    # offset_angle_deg = math.degrees(math.atan2(-gps_xy_offset[1], gps_xy_offset[0]))  # 轉成角度
    # offset_angle_clockwise = (360 - offset_angle_deg) % 360  #改成順時針
    # move_angle_deg = (bearing + offset_angle_clockwise) % 360   #跟順時針的bearing相加 (角度)
    # move_angle_rad = math.radians(move_angle_deg)  # 轉成弧度
    # print(f'move_angle = {move_angle_rad}')

    # 計算總偏移距離
    total_offset_pixels = math.sqrt(gps_xy_offset[0] ** 2 + gps_xy_offset[1] ** 2)
    total_offset_m = total_offset_pixels * meters_per_pixel

    # 計算新 UTM 座標
    new_utm_x = utm_x + total_offset_m * math.cos(move_angle)
    new_utm_y = utm_y + total_offset_m * math.sin(move_angle)

    # 將新 UTM 座標轉換為經緯度
    new_lat, new_lng = utm.to_latlon(new_utm_x, new_utm_y, zone_number, zone_letter)
    return new_lat, new_lng
def gps_distance(point_1, point_2):
    lat1, lng1 = point_1
    lat2, lng2 = point_2
    x1, y1, zone1, _ = latlng_to_utm(lat1, lng1)
    x2, y2, zone2, _ = latlng_to_utm(lat2, lng2)

    if zone1 != zone2:
        raise ValueError("兩點不在同一個 UTM 區，無法直接比較距離")

    distance = math.hypot(x2 - x1, y2 - y1)  # 平面距離（公尺）
    return distance

"""
#----------11/11------新的切割、拼接
def crop_valid_area(im_path, valid_area_list):
    v_x1, v_y1, v_x2, v_y2 = valid_area_list
    im = cv2.imread(im_path)
    h, w = im.shape[:2]
    v_x2 = w if v_x2 == -1 else v_x2
    v_y2 = h if v_y2 == -1 else v_y2
    v_im = im[v_y1:v_y2, v_x1:v_x2]
    v_im_gray = cv2.cvtColor(v_im, cv2.COLOR_BGR2GRAY)
    return v_im, v_im_gray
def stitching(img, stitch_info, top_left, conf):
    img_h, img_w = img.shape[:2]
    old_sti_x1 = 0 - stitch_info.target_crd_on_sti[0]
    old_sti_y1 = 0 - stitch_info.target_crd_on_sti[1]
    temp_p1 = (conf['template_area']['x1']-conf['valid_area']['rect']['x1'], conf['template_area']['y1'])

    src_img_x1 = top_left[0] - temp_p1[0]
    src_img_y1 = top_left[1] - temp_p1[1]

    GPS_sti_offset = [src_img_x1,src_img_y1]

    x_offset = 0 if min(old_sti_x1, src_img_x1) >= 0 else 0 - min(old_sti_x1, src_img_x1)
    y_offset = 0 if min(old_sti_y1, src_img_y1) >= 0 else 0 - min(old_sti_y1, src_img_y1)

    old_sti_x1 = old_sti_x1 + x_offset
    old_sti_y1 = old_sti_y1 + y_offset

    src_img_x1 = src_img_x1 + x_offset
    src_img_y1 = src_img_y1 + y_offset

    st_h, st_w = stitch_info.stitch_img.shape[:2]
    old_sti_x2 = old_sti_x1 + st_w
    old_sti_y2 = old_sti_y1 + st_h

    sr_h, sr_w = img.shape[:2]
    src_img_x2 = src_img_x1 + sr_w
    src_img_y2 = src_img_y1 + sr_h

    final_sti_w = max(old_sti_x2, src_img_x2)
    final_sti_h = max(old_sti_y2, src_img_y2)

    # final_sti_img = np.zeros((final_sti_h, final_sti_w, 3), np.uint8)
    # final_sti_img.fill(255)
    #
    # old_sti = stitch_info.stitch_img
    # final_sti_img[old_sti_y1:old_sti_y2, old_sti_x1:old_sti_x2] = stitch_info.stitch_img
    # final_sti_img[src_img_y1:src_img_y2, src_img_x1:src_img_x2] = img

    # 11/19 設定輸出影像的大小並加上透明背景
    final_sti_img_transparent = np.zeros((final_sti_h, final_sti_w, 4), dtype=np.uint8)
    final_sti_img_transparent[:, :, 3] = 0  # 設定 Alpha 通道為 0 (全透明)
    # 轉換為 BGRA 格式
    sti_info_stitching_img = cv2.cvtColor(stitch_info.stitch_img, cv2.COLOR_BGR2BGRA)
    current_color_img_bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    # 將影像填入指定區域並設定為不透明
    final_sti_img_transparent[old_sti_y1:old_sti_y2, old_sti_x1:old_sti_x2] = sti_info_stitching_img
    final_sti_img_transparent[src_img_y1:src_img_y2, src_img_x1:src_img_x2] = current_color_img_bgra

    return final_sti_img_transparent, [src_img_x1, src_img_y1] , GPS_sti_offset
#----------11/18----xy平面各點中心、角度
def calculate_image_centers(center_list, offset, image_size):
    # 偏移量
    x_offset, y_offset = offset
    # 根據偏移量調整所有已有的中心點位置
    if x_offset < 0 and y_offset < 0:  # 左上
        for i in range(len(center_list)):
        # 前面的點被往右下推
            center_list[i][0] = center_list[i][0] - x_offset
            center_list[i][1] = center_list[i][1] - y_offset
        # 最後加入新圖的中心點
        new_x_center = center_list[-1][0] + x_offset
        new_y_center = center_list[-1][1] + y_offset
        center_list.append([new_x_center, new_y_center])
    elif x_offset > 0 and y_offset < 0:  # 右上
        for i in range(len(center_list)):
            center_list[i][1] = center_list[i][1] - y_offset
        new_x_center = center_list[-1][0] + x_offset
        new_y_center = image_size[1]/2
        center_list.append([new_x_center, new_y_center])
    elif x_offset < 0 and y_offset > 0:  # 左下
        for i in range(len(center_list)):
        # 前面的點被往右推，但y不動
            center_list[i][0] = center_list[i][0] - x_offset
        # 新點的x = 原本的image_size[0]
        new_x_center = image_size[0] / 2
        new_y_center = center_list[-1][1] + y_offset
        center_list.append([new_x_center, new_y_center])
    elif x_offset > 0 and y_offset > 0:  # 右下
        for i in range(len(center_list)):
            new_x_center = center_list[-1][0] + x_offset
            new_y_center = center_list[-1][1] + y_offset
        center_list.append([new_x_center, new_y_center])
    elif x_offset == 0:  # 僅 y 偏移
        for i in range(len(center_list)):
            center_list[i][1] += y_offset
        new_x_center = center_list[-1][0]
        new_y_center = center_list[-1][1] + y_offset
        center_list.append([new_x_center, new_y_center])
    elif y_offset == 0:  # 僅 x 偏移
        for i in range(len(center_list)):
            center_list[i][0] += x_offset
        new_x_center = center_list[-1][0] + x_offset
        new_y_center = center_list[-1][1]
        center_list.append([new_x_center, new_y_center])
    return center_list
def calculate_movement_angle(offset):
    dx, dy = offset
    # 使用 atan2 計算角度 (以度數表示)
    angle = math.degrees(math.atan2(dy, dx))
    original_angle = (angle + 360) % 360
    # 反轉 y 軸，以符合「上方為 0 度」
    angle = math.degrees(math.atan2(-dy, dx))
    reverse_y_angle = (angle + 360) % 360
    # print(f'original_angle = {original_angle},reverse_y_angle = {reverse_y_angle}')
    return reverse_y_angle
#----------12/06----反推xy_offset(巡檢範圍座標、座標旋轉位置)
def calculate_xy_offset(point_1,point_2, meters_per_pixel, bearing):
    lat1, lng1 = point_1
    lat2, lng2 = point_2

    # 轉換經緯度為 UTM 坐標
    utm_x1, utm_y1, zone_number, zone_letter = utm.from_latlon(lat1, lng1)
    utm_x2, utm_y2, _, _ = utm.from_latlon(lat2, lng2)

    # 計算 UTM 坐標的偏移量（平面上的距離）
    delta_x = utm_x2 - utm_x1
    delta_y = utm_y2 - utm_y1

    # 計算總偏移距離（實際距離，米）
    total_offset_m = math.sqrt(delta_x ** 2 + delta_y ** 2)

    # 將實際偏移量縮放到像素空間
    total_offset_pixels = total_offset_m / meters_per_pixel

    # 計算原始偏移的方向角（基於平面，未旋轉）
    original_angle = math.atan2(delta_y, delta_x)

    # 計算相對影像方向的角度修正
    move_angle = original_angle - math.radians(bearing)

    # 計算在影像平面上的 x, y 偏移量
    x_offset = total_offset_pixels * math.cos(move_angle)
    y_offset = -total_offset_pixels * math.sin(move_angle)  # 注意 y 軸方向的反轉（圖像坐標系）

    return x_offset, y_offset
def rotate_points_with_matrix(points, rotation_matrix):
    points_array = np.array(points, dtype=np.float32)
    points_homogeneous = np.hstack([points_array, np.ones((points_array.shape[0], 1), dtype=np.float32)])
    transformed_points = rotation_matrix @ points_homogeneous.T
    transformed_points = transformed_points[:2].T
    return transformed_points.tolist()
def calculate_quadrants(gps_list,remove_range=1):
    if len(gps_list) < 2:
        return []  # 若 GPS 點少於 2 個，無法計算
    previous_lat, previous_lng = gps_list[0]
    direction_list = []
    for i in range(1, len(gps_list)):
        current_lat, current_lng = gps_list[i]
        lat_diff = current_lat - previous_lat
        lng_diff = current_lng - previous_lng
        # 判斷象限
        if lat_diff >= 0 and lng_diff >= 0:
            quadrant = 1
        elif lat_diff >= 0 and lng_diff < 0:
            quadrant = 2
        elif lat_diff < 0 and lng_diff < 0:
            quadrant = 3
        else:
            quadrant = 4
        if i == 1 :
            direction_list.append(quadrant)
        direction_list.append(quadrant)
        previous_lat, previous_lng = current_lat, current_lng
    print(f'direction_list = {direction_list}')
    return direction_list
# ---------01/13----GDAL
# def gps_geotransform(input_png,output_tiff):
#     # 左上
#     lon_0_0 = 121.04134659
#     lat_0_0 = 24.77565692
#     # 左下
#     lon_0_1479 = 121.04282717  # (0, 1479) 經度
#     lat_0_1479 = 24.77721331  # (0, 1479) 緯度
#     # 右上
#     lon_20488_0 = 121.041227  # (20488, 0) 經度
#     lat_20488_0 = 24.775753739456956  # (20488, 0) 緯度
#
#     png = gdal.Open(input_png)
#     width = png.RasterXSize
#     height = png.RasterYSize
#
#     # 計算像素尺寸
#     delta_lon_x = lon_20488_0 - lon_0_0
#     delta_lat_y = lat_0_1479 - lat_0_0
#     A = delta_lon_x / width  # X 方向每像素的經度差
#     E = delta_lat_y / height  # Y 方向每像素的緯度差
#
#     # 計算旋轉角度 (θ)
#     theta = math.atan2(delta_lat_y, delta_lon_x)
#
#     # 計算旋轉分量
#     B = (lon_0_1479 - lon_0_0) / height  # X 軸的旋轉分量
#     D = (lat_20488_0 - lat_0_0) / width  # Y 軸的旋轉分量
#
#     # 計算左上角像素中心點的經緯度
#     C = lon_0_0  # 左上角像素中心的經度
#     F = lat_0_0  # 左上角像素中心的緯度
#     print(math.degrees(theta))
#     print(C, A, B, F, D, E)
#     print(A * width + B * height + C, D * width + E * height + F)
#     geotransform = [C, A, B, F, D, E]
#
#     driver = gdal.GetDriverByName('GTiff')
#     geotiff = driver.Create(output_tiff, width, height, png.RasterCount, gdal.GDT_Byte)
#     geotiff.SetGeoTransform(geotransform)
#     srs = osr.SpatialReference()
#     srs.ImportFromEPSG(4326)
#     geotiff.SetProjection(srs.ExportToWkt())
#     for i in range(1, png.RasterCount + 1):
#         geotiff.GetRasterBand(i).WriteArray(png.GetRasterBand(i).ReadAsArray())
#     png = None
#     geotiff = None
#     return geotransform
#----------04/01----計算象限

#----------12/09----排序座標位置
"""


def create_sort_key(points):
    # 計算中心點
    center_x = sum(p[0] for p in points) / len(points)
    center_y = sum(p[1] for p in points) / len(points)
    # 返回排序函數
    def sort_key(point):
        x, y = point
        if x < center_x and y < center_y:  # 左上
            return (0, y, x)
        elif x >= center_x and y < center_y:  # 右上
            return (1, y, x)
        elif x >= center_x and y >= center_y:  # 右下
            return (2, y, x)
        elif x < center_x and y >= center_y:  # 左下
            return (3, y, x)
    return sort_key
def calculate_stitch_position(top_left_list, offset):
    x_offset, y_offset = offset
    if x_offset < 0 and y_offset < 0:  # 左上
        for i in range(len(top_left_list)):
            top_left_list[i][0] += abs(x_offset)  # 右移所有影像
            top_left_list[i][1] += abs(y_offset)  # 下移所有影像
        new_x = 0
        new_y = 0
    elif x_offset > 0 and y_offset < 0:  # 右上
        for i in range(len(top_left_list)):
            top_left_list[i][1] += abs(y_offset)  # 下移所有影像
        new_x = top_left_list[-1][0] + x_offset
        new_y = 0
    elif x_offset < 0 and y_offset > 0:  # 左下
        if top_left_list[-1][0] == 0:
            for i in range(len(top_left_list)):
                top_left_list[i][0] -= x_offset  # 右移所有影像
            new_x = top_left_list[-1][0] + x_offset
            new_y = top_left_list[-1][1] + y_offset
        # 0218  當有往右，再往左時會出錯
        else:
            rezero = top_left_list[-1][0] + x_offset
            # -4 = 5 + -9
            if rezero < 0 :
                for i in range(len(top_left_list)):
                    top_left_list[i][0] -= rezero
                new_x = 0
                new_y = top_left_list[-1][1] + y_offset
            # 79 = 81 + -2
            else:
                new_x = top_left_list[-1][0] + x_offset
                new_y = top_left_list[-1][1] + y_offset
    elif x_offset > 0 and y_offset > 0:  # 右下
        new_x = top_left_list[-1][0] + x_offset
        new_y = top_left_list[-1][1] + y_offset
    elif x_offset == 0:  # 只有 y 偏移 , x = 0
        new_x = top_left_list[-1][0] + x_offset
        new_y = top_left_list[-1][1] + y_offset
    top_left_list.append([new_x, new_y])
    return top_left_list


# ----05/23----分段拼接、分段透視轉換
def stitch_seg_images(stitch_offset_list, valid_images):
    img_h, img_w = valid_images[0].shape[:2]
    # 座標疊加
    for i in range(1, len(stitch_offset_list)):
        stitch_offset_list[i] = [a + b for a, b in zip(stitch_offset_list[i], stitch_offset_list[i - 1])]
    print(stitch_offset_list)
    # 找最大、最小值
    min_x = min(point[0] for point in stitch_offset_list)
    min_y = min(point[1] for point in stitch_offset_list)
    # 計算偏移量，確保最小座標是 (0,0)
    offset_x = -min_x if min_x < 0 else 0
    offset_y = -min_y if min_y < 0 else 0
    # 座標 無負數正規化
    final_coordinate_list = [[x + offset_x, y + offset_y] for x, y in stitch_offset_list]
    print(f"final_coordinate_list = {final_coordinate_list}")
    # 影像中心座標
    center_coordinate_list = []
    for i in range(0, len(final_coordinate_list)):
        center_coord = [final_coordinate_list[i][0] + int(img_w * 0.5), final_coordinate_list[i][1] + int(img_h * 0.5)]
        center_coordinate_list.append(center_coord)
    print(f"center_coordinate_list = {center_coordinate_list}")
    canvas_width = max(x for x, y in final_coordinate_list) + img_w
    canvas_height = max(y for x, y in final_coordinate_list) + img_h
    print(f"畫布大小 = {canvas_width, canvas_height}")
    final_sti_img = np.zeros((canvas_height, canvas_width, 3), np.uint8)
    final_sti_img.fill(255)
    for i in range(0, len(final_coordinate_list)):
        final_sti_img[final_coordinate_list[i][1]:final_coordinate_list[i][1] + img_h,
        final_coordinate_list[i][0]:final_coordinate_list[i][0] + img_w] = valid_images[i]
    return final_sti_img,final_coordinate_list,center_coordinate_list
def is_white(pixel):
    return np.array_equal(pixel, [255, 255, 255])
def find_corners(segment):
    h, w = segment.shape[:2]
    top_left = top_right = bottom_left = bottom_right = None

    # 左上
    for y in range(h):
        for x in range(w):
            if not is_white(segment[y, x]):
                top_left = (x, y)
                break
        if top_left:
            break

    # 右上
    for y in range(h):
        for x in range(w - 1, -1, -1):
            if not is_white(segment[y, x]):
                top_right = (x, y)
                break
        if top_right:
            break

    # 左下
    for y in range(h - 1, -1, -1):
        for x in range(w):
            if not is_white(segment[y, x]):
                bottom_left = (x, y)
                break
        if bottom_left:
            break

    # 右下
    for y in range(h - 1, -1, -1):
        for x in range(w - 1, -1, -1):
            if not is_white(segment[y, x]):
                bottom_right = (x, y)
                break
        if bottom_right:
            break

    return top_left, top_right, bottom_left, bottom_right
    
# ----07/15----分段拼接 (透明)
def stitch_seg_images_transparent(stitch_offset_list, valid_images):
    img_h, img_w = valid_images[0].shape[:2]
    # 座標疊加
    for i in range(1, len(stitch_offset_list)):
        stitch_offset_list[i] = [a + b for a, b in zip(stitch_offset_list[i], stitch_offset_list[i - 1])]
    logging.debug(stitch_offset_list)
    # 找最大、最小值
    min_x = min(point[0] for point in stitch_offset_list)
    min_y = min(point[1] for point in stitch_offset_list)
    # 計算偏移量，確保最小座標是 (0,0)
    offset_x = -min_x if min_x < 0 else 0
    offset_y = -min_y if min_y < 0 else 0
    # 座標 無負數正規化
    final_coordinate_list = [[x + offset_x, y + offset_y] for x, y in stitch_offset_list]
    logging.debug(f"final_coordinate_list = {final_coordinate_list}")
    # 影像中心座標
    center_coordinate_list = []
    for i in range(0, len(final_coordinate_list)):
        center_coord = [final_coordinate_list[i][0] + int(img_w * 0.5), final_coordinate_list[i][1] + int(img_h * 0.5)]
        center_coordinate_list.append(center_coord)
    logging.debug(f"center_coordinate_list = {center_coordinate_list}")
    canvas_width = max(x for x, y in final_coordinate_list) + img_w
    canvas_height = max(y for x, y in final_coordinate_list) + img_h
    logging.debug(f"畫布大小 = {canvas_width, canvas_height}")
    final_sti_img = np.zeros((canvas_height, canvas_width, 4), np.uint8)  # 4 channels: RGBA
    final_sti_img[:, :, 3] = 0  # A 通道設為 0（完全透明）
    for i in range(len(final_coordinate_list)):
      img = valid_images[i]
      img_bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
      # 轉為 BGRA 格式
      #img_bgra = np.zeros((img_h, img_w, 4), dtype=np.uint8)
      #img_bgra[:, :, :3] = img  # 複製BGR通道
      #img_bgra[:, :, 3] = 255
      final_sti_img[
          final_coordinate_list[i][1]:final_coordinate_list[i][1] + img_h,
          final_coordinate_list[i][0]:final_coordinate_list[i][0] + img_w] = img_bgra
    return final_sti_img,final_coordinate_list,center_coordinate_list
def is_background(pixel):
    if len(pixel) == 4:
        return pixel[3] == 0  # RGBA: 完全透明
    else:
        return np.array_equal(pixel, [255, 255, 255])  # RGB: 純白
def find_corners_transparent(segment):
    h, w = segment.shape[:2]
    top_left = top_right = bottom_left = bottom_right = None

    # 左上
    for y in range(h):
        for x in range(w):
            if not is_background(segment[y, x]):
                top_left = (x, y)
                break
        if top_left:
            break

    # 右上
    for y in range(h):
        for x in range(w - 1, -1, -1):
            if not is_background(segment[y, x]):
                top_right = (x, y)
                break
        if top_right:
            break

    # 左下
    for y in range(h - 1, -1, -1):
        for x in range(w):
            if not is_background(segment[y, x]):
                bottom_left = (x, y)
                break
        if bottom_left:
            break

    # 右下
    for y in range(h - 1, -1, -1):
        for x in range(w - 1, -1, -1):
            if not is_background(segment[y, x]):
                bottom_right = (x, y)
                break
        if bottom_right:
            break

    return top_left, top_right, bottom_left, bottom_right
    
def warp_segment(segment, corners, target_width):
    tl, tr, bl, br = corners

    # 統一寬度時，依比例調整右側座標
    current_width = np.linalg.norm(np.array(tr) - np.array(tl))
    scale = target_width / current_width

    # 對 tr, br 的 x 座標進行縮放
    tr_scaled = (tl[0] + (tr[0] - tl[0]) * scale, tl[1] + (tr[1] - tl[1]) * scale)
    br_scaled = (bl[0] + (br[0] - bl[0]) * scale, bl[1] + (br[1] - bl[1]) * scale)

    # 計算高度（不動）
    height_left = np.linalg.norm(np.array(bl) - np.array(tl))
    height_right = np.linalg.norm(np.array(br) - np.array(tr))
    target_height = int(max(height_left, height_right))

    src_pts = np.array([tl, tr_scaled, bl, br_scaled], dtype=np.float32)
    dst_pts = np.array([
        [0, 0],
        [target_width - 1, 0],
        [0, target_height - 1],
        [target_width - 1, target_height - 1]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(segment, matrix, (target_width, target_height), borderValue=(255,255,255))

    return warped

# 0619 暗角處理
def correct_vignetting(image, strength=1.0, smooth=1.0):
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2

    # 建立距離遮罩
    Y, X = np.ogrid[:h, :w]
    distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)

    # === 修正邏輯 ===
    vignette_mask = 1 + ((distance / max_dist) ** smooth) * (strength - 1)
    vignette_mask = np.clip(vignette_mask, 1.0, strength)

    # 套用遮罩補償亮度
    corrected = np.zeros_like(image, dtype=np.float32)
    for c in range(3):
        corrected[..., c] = image[..., c] * vignette_mask

    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return corrected
def adjust_images_to_average_brightness(images, target='mean'):
    brightness_list = []
    Y_channels = []

    # Step 1: 預先計算所有亮度與 Y 通道
    for img in images:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        Y = ycrcb[:, :, 0].astype(np.float32)
        Y_channels.append((ycrcb, Y))
        brightness_list.append(np.mean(Y))

    # Step 2: 計算目標亮度
    if target == 'mean':
        target_brightness = np.mean(brightness_list)
    else:
        target_brightness = float(target)

    print(f"目標平均亮度為：{target_brightness:.2f}")

    # Step 3: 調整每張圖片亮度
    adjusted_images = []
    for (ycrcb, Y), original_brightness in zip(Y_channels, brightness_list):
        delta = target_brightness - original_brightness
        Y += delta
        Y = np.clip(Y, 0, 255)
        ycrcb[:, :, 0] = Y.astype(np.uint8)
        adjusted = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        adjusted_images.append(adjusted)

    return adjusted_images
# 0619 去除模糊   (底下都有改，包含showtemp改抓記憶體)
def read_with_roi(filepath, border_margin=0):
    img = cv2.imread(filepath)
    # 暗角處理
    img = correct_vignetting(img, strength=1.8, smooth=2.0)
    if img is None:
        return None
    h, w = img.shape[:2]
    return  img[0+border_margin:h-border_margin,0+border_margin:w-border_margin]
    # return img[y_start:y_end, x_start:x_end]

# 0620  邊緣處理
def build_gaussian_pyramid(img, levels):
    gp = [img.astype(np.float32)]
    for _ in range(levels):
        img = cv2.pyrDown(img)
        gp.append(img.astype(np.float32))
    return gp
def build_laplacian_pyramid(gp):
    lp = []
    for i in range(len(gp) - 1):
        size = (gp[i].shape[1], gp[i].shape[0])
        expanded = cv2.pyrUp(gp[i+1], dstsize=size)
        lap = cv2.subtract(gp[i], expanded)
        lp.append(lap)
    lp.append(gp[-1])
    return lp
def blend_pyramids(lpA, lpB, gpMask):
    blended = []
    for la, lb, gm in zip(lpA, lpB, gpMask):
        gm = gm / 255.0
        gm = np.repeat(gm[:, :, np.newaxis], 3, axis=2)
        blended.append(la * gm + lb * (1 - gm))
    return blended
def reconstruct_from_pyramid(pyramid):
    img = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        size = (pyramid[i].shape[1], pyramid[i].shape[0])
        img = cv2.pyrUp(img, dstsize=size)
        img = cv2.add(img, pyramid[i])
    return np.clip(img, 0, 255).astype(np.uint8)
def soften_seams(img, final_coordinate_list, blend_height=100, levels=6):
    height, width = img.shape[:2]
    result_img = img.copy()

    for i in range(0, len(final_coordinate_list)):
        if final_coordinate_list[i][1] != 0 :
            x_start, y_seam = final_coordinate_list[i]
            x_end = x_start + width

            # 擷取上下影像塊
            top = result_img[y_seam - blend_height : y_seam, x_start:x_end]
            bottom = result_img[y_seam : y_seam + blend_height, x_start:x_end]

            # Sigmoid 遮罩（由白到黑）
            t = np.linspace(-6, 6, blend_height)
            sigmoid_mask = (1 / (1 + np.exp(t))) * 255
            mask = sigmoid_mask.astype(np.uint8).reshape(-1, 1)
            mask = np.repeat(mask, top.shape[1], axis=1)

            # 建立 pyramids
            gpMask = build_gaussian_pyramid(mask, levels)
            lpTop = build_laplacian_pyramid(build_gaussian_pyramid(top, levels))
            lpBottom = build_laplacian_pyramid(build_gaussian_pyramid(bottom, levels))
            blended_pyr = blend_pyramids(lpTop, lpBottom, gpMask)
            blended = reconstruct_from_pyramid(blended_pyr)

            # 貼回處理後的融合區塊
            h_blend = blended.shape[0] // 2
            result_img[y_seam - h_blend : y_seam + h_blend, x_start:x_end] = blended

    return result_img


"""
class CropThread(QThread):
    progress_signal = pyqtSignal(int)  # 傳送進度百分比
    list_signal = pyqtSignal(list, list)  # 傳送陣列
    def __init__(self, img_list, conf,valid_images,direction_list,direction_label,revise_match_list):
        super().__init__()
        # MainWindow 給的參數
        self.img_fullpath_valid_list = img_list
        self.conf = conf
        self.valid_images = valid_images
        self.direction_list = direction_list
        self.direction_label = direction_label
        self.revise_match_list = revise_match_list  # 給答案修正
        self.template_img_list = []

    def run(self):
        start_time = time.time()
        # 04/17
        # 依方向取四邊(ex.2是往上飛，3往左，4往下)
        methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
        method_select = 1
        meth = methods[method_select]
        method = eval(meth)
        img_h, img_w = self.valid_images[0].shape[:2]
        temp_up_p1 = [self.conf['template_area']['temp_up']['x1'], self.conf['template_area']['temp_up']['y1']]
        temp_up_p2 = [self.conf['template_area']['temp_up']['x2'], self.conf['template_area']['temp_up']['y2']]
        temp_up_h = temp_up_p2[1] - temp_up_p1[1]
        temp_up_w = temp_up_p2[0] - temp_up_p1[0]
        temp_down_p1 = [self.conf['template_area']['temp_down']['x1'], self.conf['template_area']['temp_down']['y1']]
        temp_down_p2 = [self.conf['template_area']['temp_down']['x2'], self.conf['template_area']['temp_down']['y2']]
        temp_down_h = temp_down_p2[1] - temp_down_p1[1]
        temp_down_w = temp_down_p2[0] - temp_down_p1[0]
        temp_left_p1 = [self.conf['template_area']['temp_left']['x1'], self.conf['template_area']['temp_left']['y1']]
        temp_left_p2 = [self.conf['template_area']['temp_left']['x2'], self.conf['template_area']['temp_left']['y2']]
        temp_left_h = temp_left_p2[1] - temp_left_p1[1]
        temp_left_w = temp_left_p2[0] - temp_left_p1[0]
        temp_right_p1 = [self.conf['template_area']['temp_right']['x1'], self.conf['template_area']['temp_right']['y1']]
        temp_right_p2 = [self.conf['template_area']['temp_right']['x2'], self.conf['template_area']['temp_right']['y2']]
        temp_right_h = temp_right_p2[1] - temp_right_p1[1]
        temp_right_w = temp_right_p2[0] - temp_right_p1[0]
        match_scope_offset = 100
        self.match_coordinate_list = []
        stitch_offset_list = []
        stitch_offset_list.insert(0, [0, 0])

        for i in range(1,len(self.valid_images)):  #valid_images圖片數=n，template_img_list樣本數=n-1，match_coordinate_list對應上一張座標=n-1，stitch_offset_list與上一張差值=n
            current_img = self.valid_images[i]
            current_img_gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
            previous_img = self.valid_images[i - 1]
            previous_img_gray = cv2.cvtColor(previous_img, cv2.COLOR_BGR2GRAY)
            if self.direction_list[i] == self.direction_label[3] : # 往右
                revise_entry = next((item for item in self.revise_match_list if item[0] == i-1), None)
                if revise_entry:
                    # 直接使用修正後的座標
                    top_left = tuple(revise_entry[1])
                    print(f"{i}使用 revise_match_list 中的修正座標: {top_left}")
                else:
                    temp_img = current_img_gray[temp_right_p1[1]:temp_right_p2[1], temp_right_p1[0]:temp_right_p2[0]]
                    self.matching_scope = [temp_right_p2[0], 0], [img_w, img_h]
                    if self.direction_list[i] == self.direction_list[i-1] and i!=1  :
                        self.matching_scope = ([self.match_coordinate_list[i-2][0]-match_scope_offset, self.match_coordinate_list[i-2][1]-match_scope_offset],
                                               [self.match_coordinate_list[i-2][0]+temp_right_w+match_scope_offset, self.match_coordinate_list[i-2][1]+temp_right_h+match_scope_offset])
                    (x1, y1), (x2, y2) = self.matching_scope
                    # 邊界修正，避免比對範圍超出影像
                    x1 = max(0, min(x1, img_w))
                    y1 = max(0, min(y1, img_h))
                    x2 = max(0, min(x2, img_w))
                    y2 = max(0, min(y2, img_h))
                    match_img = previous_img_gray[y1:y2, x1:x2]
                    # 0619 切圖導致比對範圍小於模板大小
                    if match_img.shape[0] < temp_img.shape[0] or match_img.shape[1] < temp_img.shape[1]:
                        print(f"[WARN] 第 {i} 張圖限縮範圍太小，fallback 回大範圍比對")
                        # fallback：回到預設大範圍
                        self.matching_scope = [0, 0], [img_w, temp_up_p1[1]]
                        (x1, y1), (x2, y2) = self.matching_scope
                        x1 = max(0, min(x1, img_w))
                        y1 = max(0, min(y1, img_h))
                        x2 = max(0, min(x2, img_w))
                        y2 = max(0, min(y2, img_h))
                        match_img = previous_img_gray[y1:y2, x1:x2]
                    # 此處保證 match_img now 足夠大（因為 fallback）
                    res = cv2.matchTemplate(match_img, temp_img, method)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    top_left = max_loc
                    top_left = (top_left[0] + self.matching_scope[0][0], top_left[1] + self.matching_scope[0][1])
                stitch_offset_list.append([top_left[0] - temp_right_p1[0], top_left[1] - temp_right_p1[1]])
            elif self.direction_list[i] == self.direction_label[0] : # 往上拼，template在下
                revise_entry = next((item for item in self.revise_match_list if item[0] == i - 1), None)
                if revise_entry:
                    # 直接使用修正後的座標
                    top_left = tuple(revise_entry[1])
                    print(f"{i}使用 revise_match_list 中的修正座標: {top_left}")
                else:
                    temp_img = current_img_gray[temp_up_p1[1]:temp_up_p2[1], temp_up_p1[0]:temp_up_p2[0]]
                    self.matching_scope = [0, 0], [img_w, temp_up_p1[1]]
                    if self.direction_list[i] == self.direction_list[i-1] and i!=1 :
                        self.matching_scope = ([self.match_coordinate_list[i-2][0]-match_scope_offset, self.match_coordinate_list[i-2][1]-match_scope_offset],
                                               [self.match_coordinate_list[i-2][0]+temp_up_w+match_scope_offset, self.match_coordinate_list[i-2][1]+temp_up_h+match_scope_offset])
                    (x1, y1), (x2, y2) = self.matching_scope
                    # 邊界修正，避免比對範圍超出影像
                    x1 = max(0, min(x1, img_w))
                    y1 = max(0, min(y1, img_h))
                    x2 = max(0, min(x2, img_w))
                    y2 = max(0, min(y2, img_h))
                    match_img = previous_img_gray[y1:y2, x1:x2]
                    # 0619 切圖導致比對範圍小於模板大小
                    if match_img.shape[0] < temp_img.shape[0] or match_img.shape[1] < temp_img.shape[1]:
                        print(f"[WARN] 第 {i} 張圖限縮範圍太小，fallback 回大範圍比對")
                        # fallback：回到預設大範圍
                        self.matching_scope = [0, 0], [img_w, temp_up_p1[1]]
                        (x1, y1), (x2, y2) = self.matching_scope
                        x1 = max(0, min(x1, img_w))
                        y1 = max(0, min(y1, img_h))
                        x2 = max(0, min(x2, img_w))
                        y2 = max(0, min(y2, img_h))
                        match_img = previous_img_gray[y1:y2, x1:x2]
                    # 此處保證 match_img now 足夠大（因為 fallback）
                    res = cv2.matchTemplate(match_img, temp_img, method)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    top_left = max_loc
                    top_left = (top_left[0] + self.matching_scope[0][0], top_left[1] + self.matching_scope[0][1])
                stitch_offset_list.append([top_left[0] - temp_up_p1[0], top_left[1] - temp_up_p1[1]])

            elif self.direction_list[i] == self.direction_label[2] :  # 往左拼，template在右
                revise_entry = next((item for item in self.revise_match_list if item[0] == i - 1), None)
                if revise_entry:
                    # 直接使用修正後的座標
                    top_left = tuple(revise_entry[1])
                    print(f"{i}使用 revise_match_list 中的修正座標: {top_left}")
                else:
                    temp_img = current_img_gray[temp_left_p1[1]:temp_left_p2[1], temp_left_p1[0]:temp_left_p2[0]]
                    cv2.imwrite(f"./test_img_save/temp_0417_{i}.jpg", temp_img)
                    self.matching_scope = [0, 0], [temp_left_p1[0], img_h]
                    if self.direction_list[i] == self.direction_list[i-1] and i!=1 :
                        self.matching_scope = ([self.match_coordinate_list[i-2][0]-match_scope_offset, self.match_coordinate_list[i-2][1]-match_scope_offset],
                                               [self.match_coordinate_list[i-2][0]+temp_left_w+match_scope_offset, self.match_coordinate_list[i-2][1]+temp_left_h+match_scope_offset])
                    (x1, y1), (x2, y2) = self.matching_scope
                    # 邊界修正，避免比對範圍超出影像
                    x1 = max(0, min(x1, img_w))
                    y1 = max(0, min(y1, img_h))
                    x2 = max(0, min(x2, img_w))
                    y2 = max(0, min(y2, img_h))
                    match_img = previous_img_gray[y1:y2, x1:x2]
                    # 0619 切圖導致比對範圍小於模板大小
                    if match_img.shape[0] < temp_img.shape[0] or match_img.shape[1] < temp_img.shape[1]:
                        print(f"[WARN] 第 {i} 張圖限縮範圍太小，fallback 回大範圍比對")
                        # fallback：回到預設大範圍
                        self.matching_scope = [0, 0], [img_w, temp_up_p1[1]]
                        (x1, y1), (x2, y2) = self.matching_scope
                        x1 = max(0, min(x1, img_w))
                        y1 = max(0, min(y1, img_h))
                        x2 = max(0, min(x2, img_w))
                        y2 = max(0, min(y2, img_h))
                        match_img = previous_img_gray[y1:y2, x1:x2]
                    # 此處保證 match_img now 足夠大（因為 fallback）
                    res = cv2.matchTemplate(match_img, temp_img, method)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    top_left = max_loc
                    top_left = (top_left[0] + self.matching_scope[0][0], top_left[1] + self.matching_scope[0][1])
                    cv2.imwrite(f"./test_img_save/match_0417_{i}.jpg", match_img)
                stitch_offset_list.append([top_left[0] - temp_left_p1[0], top_left[1] - temp_left_p1[1]])

            elif self.direction_list[i] == self.direction_label[1] :  # 往下拼，template在上，matching範圍在下
                revise_entry = next((item for item in self.revise_match_list if item[0] == i - 1), None)
                if revise_entry:
                    # 直接使用修正後的座標
                    top_left = tuple(revise_entry[1])
                    print(f"{i}使用 revise_match_list 中的修正座標: {top_left}")
                else:
                    temp_img = current_img_gray[temp_down_p1[1]:temp_down_p2[1], temp_down_p1[0]:temp_down_p2[0]]
                    self.matching_scope = [0, temp_down_p2[1]], [img_w, img_h]
                    if self.direction_list[i] == self.direction_list[i-1] and i!=1 :
                        self.matching_scope = ([self.match_coordinate_list[i-2][0]-match_scope_offset, self.match_coordinate_list[i-2][1]-match_scope_offset],
                                               [self.match_coordinate_list[i-2][0]+temp_down_w+match_scope_offset, self.match_coordinate_list[i-2][1]+temp_down_h+match_scope_offset])
                    (x1, y1), (x2, y2) = self.matching_scope
                    # 邊界修正，避免比對範圍超出影像
                    x1 = max(0, min(x1, img_w))
                    y1 = max(0, min(y1, img_h))
                    x2 = max(0, min(x2, img_w))
                    y2 = max(0, min(y2, img_h))
                    match_img = previous_img_gray[y1:y2, x1:x2]
                    # 0619 切圖導致比對範圍小於模板大小
                    if match_img.shape[0] < temp_img.shape[0] or match_img.shape[1] < temp_img.shape[1]:
                        print(f"[WARN] 第 {i} 張圖限縮範圍太小，fallback 回大範圍比對")
                        # fallback：回到預設大範圍
                        self.matching_scope = [0, 0], [img_w, temp_up_p1[1]]
                        (x1, y1), (x2, y2) = self.matching_scope
                        x1 = max(0, min(x1, img_w))
                        y1 = max(0, min(y1, img_h))
                        x2 = max(0, min(x2, img_w))
                        y2 = max(0, min(y2, img_h))
                        match_img = previous_img_gray[y1:y2, x1:x2]
                    # 此處保證 match_img now 足夠大（因為 fallback）
                    res = cv2.matchTemplate(match_img, temp_img, method)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    top_left = max_loc
                    top_left = (top_left[0] + self.matching_scope[0][0], top_left[1] + self.matching_scope[0][1])
                stitch_offset_list.append([top_left[0] - temp_down_p1[0], top_left[1] - temp_down_p1[1]])
            self.template_img_list.append(temp_img)
            self.match_coordinate_list.append(top_left)
        print(self.match_coordinate_list)
        print(stitch_offset_list)
        self.list_signal.emit(stitch_offset_list, self.match_coordinate_list)
        end_time = time.time()
        print(end_time-start_time)
    # @staticmethod
    # def process_template_matching(task):
    #     # i, valid_images, matching_scope, flag_same_size, template_img_list = task
    #     i, target_img, matching_scope, flag_same_size, temp_img = task
    #     # i == 0，會多回傳一個None
    #     if i == 0:
    #         return (i, None)
    #     # temp_img = template_img_list[i]
    #     if flag_same_size == True:
    #         # target_img = cv2.cvtColor(valid_images[i-1], cv2.COLOR_BGR2GRAY)
    #         target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    #     else:
    #         target_img = cv2.imread(f'./result/crop_images/{i - 1}.jpg', cv2.IMREAD_GRAYSCALE)
    #     methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF',
    #                'cv2.TM_SQDIFF_NORMED']
    #     method_select = 1
    #     meth = methods[method_select]
    #     method = eval(meth)
    #     # 限縮target
    #     match_img = target_img[matching_scope[0][1]:matching_scope[1][1], matching_scope[0][0]:matching_scope[1][0]]
    #     res = cv2.matchTemplate(match_img, temp_img, method)
    #     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    #     # print(max_val)
    #     if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
    #         top_left = min_loc
    #     else:
    #         top_left = max_loc
    #         top_left = (top_left[0] + matching_scope[0][0], top_left[1] + matching_scope[0][1])
    #     return (i, top_left)

class WorkerThread(QThread):
    progress_signal = pyqtSignal(int)  # 傳送進度百分比
    list_signal = pyqtSignal(list,list)  # 傳送陣列
    def __init__(self, img_list,conf,gps_list,template_list,valid_images):
        super().__init__()
        # MainWindow 給的參數
        self.img_fullpath_valid_list = img_list
        self.conf = conf
        self.GPS_valid_list = gps_list
        self.template_list = template_list
        self.valid_images = valid_images
        # 初始化宣告
        self.stitch_info = None
        self.GPS_sti_offset_list = [[0, 0]]
        self.coord_gps_temp = []
        self.coord_gps_temp.append({'coord': [None, None], 'gps': [None, None]})
        self.GPS_crop_center_list = []
        self.XY_crop_center_list = []
        self.crop_point_list = []
        self.flag_same_size = False
    def run(self):
        start_time = time.time()
        # 0214 先計算出大圖大小 > 各小圖於大圖位置 > 最後再一次賦值
        temp_p1 = (self.conf['template_area']['x1'], self.conf['template_area']['y1'])
        # 計算每張圖的xy位移
        stitch_offset_list = [[x - temp_p1[0], y - temp_p1[1]] for x, y in self.template_list]
        print(f"stitch_offset_list: {stitch_offset_list}")
        # 累加計算 x 和 y
        accumulated = list(accumulate(stitch_offset_list, lambda a, b: [a[0] + b[0], a[1] + b[1]]))
        x_values, y_values = zip(*accumulated)
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)

        h, w = self.valid_images[0].shape[:2]
        image_size = [w, h]

        # 計算畫布大小 canvas_width,canvas_height
        canvas_width = image_size[0] + max(abs(x_min), abs(x_max),abs(x_max-x_min))
        canvas_height = image_size[1] + max(abs(y_min), abs(y_max))
        print(f"畫布大小 = {canvas_width, canvas_height}")

        # 計算各小圖位於大圖位置 top_left_list
        top_left_list = [[0, 0]]
        for i in range(0, len(stitch_offset_list)):
            calculate_stitch_position(top_left_list, stitch_offset_list[i])
        print(f'top_left_list = {top_left_list}')


        # # 計算角落四點
        # print(self.GPS_valid_list)
        # meters_per_pixel = calculate_meters_per_pixel(self.GPS_valid_list[0], self.GPS_valid_list[1],stitch_offset_list[0])
        # bearing = calculate_bearing(self.GPS_valid_list[0][0], self.GPS_valid_list[0][1], self.GPS_valid_list[1][0],self.GPS_valid_list[1][1])
        # xy_angle = calculate_movement_angle(stitch_offset_list[0][0], stitch_offset_list[0][1])
        # rotate_angle = bearing - xy_angle
        # print(meters_per_pixel, bearing, xy_angle, rotate_angle)
        # print(self.conf['valid_scope'])
        #
        # total_meters_per_pixel = 0
        # for i in range (1,len(stitch_offset_list)):
        #     total_meters_per_pixel += calculate_meters_per_pixel(self.GPS_valid_list[i-1], self.GPS_valid_list[i], stitch_offset_list[i-1])
        # average_meters_per_pixel = total_meters_per_pixel / len(stitch_offset_list)
        # print(average_meters_per_pixel)

        # # 左上
        # x_offset, y_offset = calculate_xy_offset(self.GPS_valid_list[0], self.conf['valid_scope'][2], meters_per_pixel,rotate_angle)
        # result = [sum(values) for values in zip(top_left_list[0], [2304, 1728], [x_offset, y_offset])]
        # print(result)
        # # 右上
        # x_offset, y_offset = calculate_xy_offset(self.GPS_valid_list[0], self.conf['valid_scope'][3], meters_per_pixel,rotate_angle)
        # result = [sum(values) for values in zip(top_left_list[0], [2304, 1728], [x_offset, y_offset])]
        # print(result)
        # # 左下
        # x_offset, y_offset = calculate_xy_offset(self.GPS_valid_list[-1], self.conf['valid_scope'][1], meters_per_pixel,rotate_angle)
        # result = [sum(values) for values in zip(top_left_list[-1], [2304, 1728], [x_offset, y_offset])]
        # print(result)
        # # 右下
        # x_offset, y_offset = calculate_xy_offset(self.GPS_valid_list[-1], self.conf['valid_scope'][0], meters_per_pixel,rotate_angle)
        # result = [sum(values) for values in zip(top_left_list[-1], [2304, 1728], [x_offset, y_offset])]
        # print(result)




        # # 另一種計算top_left_list的方法，最後才加
        # test_list = [[0,0]]
        # for offset in stitch_offset_list:
        #     new_position = [test_list[-1][0] + offset[0], test_list[-1][1] + offset[1]]
        #     test_list.append(new_position)
        # print(f"original top_left_list = {test_list}")
        # max_negative_x = min([x[0] for x in test_list if x[0] < 0])
        # for i in range(len(test_list)):
        #     test_list[i][0] += abs(max_negative_x)
        # print(f"Updated top_left_list = {test_list}")

        # 多工讀圖
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     images = list(executor.map(cv2.imread, self.img_fullpath_valid_list))
        # 創建空白大圖，並把小圖填入
        final_sti_img = np.zeros((canvas_height, canvas_width, 3), np.uint8)
        final_sti_img.fill(255)

        for i in range(0, len(self.valid_images)):
            if (top_left_list[i][1] + image_size[1] > canvas_height or
                    top_left_list[i][0] + image_size[0] > canvas_width):
                print(f"第 {i} 張圖像超出了畫布範圍！")
                continue
            final_sti_img[top_left_list[i][1]:top_left_list[i][1] + image_size[1],top_left_list[i][0]:top_left_list[i][0] + image_size[0]] = self.valid_images[i]
        start_save_time = time.time()
        cv2.imwrite(f"./result/stitch/stitch_{len(self.img_fullpath_valid_list) - 1}.png", final_sti_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        end_time = time.time()
        print(f'stitching_time = {end_time - start_time}')
        print(f'save_time = {end_time - start_save_time}')
        self.list_signal.emit(self.GPS_sti_offset_list,self.crop_point_list)
        """


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    Progress_Signal = QtCore.pyqtSignal(int)
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.setupUi(self)

        self.btn_open_dir.clicked.connect(self.open_dir)
        self.btn_write.clicked.connect(self.yaml_write)
        self.btn_load.clicked.connect(self.yaml_load)
        self.tableWidget.cellClicked.connect(self.cell_clicked)
        # self.btn_open.clicked.connect(self.show_temp)
        # self.tableWidget.cellDoubleClicked.connect(self.cell_double_clicked)
        # self.btn_crop.clicked.connect(self.start_crop)
        # self.btn_stitch.clicked.connect(self.start_stitching)


        # output: folder create
        os.makedirs("logs", exist_ok=True)
        # 建立唯一 log 檔名（依日期時間）
        self.time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"logs/{self.time_str}.log"
        # 設定 logging 格式與輸出檔案
        logging.basicConfig(
            level=logging.INFO,  # 用DEBUG會存一堆EXIF資訊
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),
                logging.StreamHandler()  # 同時印在終端機
            ]
        )

        self.step_flag = 0
        self.worker_thread = None
        self.mouse_flag = False
        self.flag_draw = False

        self.widget.setVisible(False)
        self.label_6.setVisible(False)
        self.label_7.setVisible(False)

        # 重寫最終滑鼠事件
        self.conf = None
        self.pushButton_config.clicked.connect(self.open_config)
        self.graphicsView.setVisible(False)
        self.selection_rect = None
        self.start_pos = None
        self.graphicsView.mousePressEvent = self.start_selection
        self.graphicsView.mouseMoveEvent = self.update_selection
        self.graphicsView.mouseReleaseEvent = self.finish_selection

        # 0703
        conf_file = open(self.lineEdit_config.text(), 'r')
        self.conf = yaml.safe_load(conf_file)
        self.set_lineedit()
        self.img_h = 0
        self.img_w = 0
        self.revise_match_list = []
        self.watch_folder = r"D:\itri\20250907\test"
        self.watchdog_info_list = []
        self.matching_busy = False

        # 0801
        self.reading_active = False
        self.loader_thread = WatchdogLoader(self.watch_folder)
        self.loader_thread.reading_active = self.reading_active
        self.loader_thread.image_loaded.connect(self.handle_new_image)
        self.loader_thread.start()

        self.pushButton_show.clicked.connect(self.show_template)
        self.pushButton_rematching.clicked.connect(self.rematch_selected)
        self.pushButton_stitch.clicked.connect(self.stitch_class)
        self.show_count = 0
        self.rematch_running = False
        self.btn_change.clicked.connect(self.toggle_change_mode)

        # 比對佇列
        self.matching_busy = False
        self.pending_pairs = []
        self.current_match_thread = None
        self.last_checked_row = None

        # 開始/暫停讀圖
        self.pushButton_test.clicked.connect(self.toggle_reading)
        # # 查詢目前Class資料
        # self.pushButton_test_2.clicked.connect(self.print_watchdog_info_list)
        # # 開啟單一圖片給上座標產TIFF
        # self.pushButton_test_3.clicked.connect(self.open_image)
        self.pushButton_test_2.clicked.connect(self.big_match)
        self.pushButton_test_3.clicked.connect(self.big_stitch)

    # def big_match(self):
    #     self.label_6.setVisible(True)
    #     self.label_7.setVisible(True)
    #     file_path, _ = QFileDialog.getOpenFileName(self, "選擇圖片檔案", "", "Image Files (*.png *.jpg *.JPG)")
    #     if not file_path:
    #         return
    #     self.big_prev = cv2.imread(file_path)
    #     prev_img_h, prev_img_w = self.big_prev.shape[:2]
    #     top_height = min(3648, prev_img_h)  # 避免影像本身小於 3000px
    #     self.prev_cropped = self.big_prev[0:top_height, 0:prev_img_w]
    #     prev_gray = cv2.cvtColor(self.prev_cropped, cv2.COLOR_BGR2GRAY)
    #
    #     file_path, _ = QFileDialog.getOpenFileName(self, "選擇圖片檔案", "", "Image Files (*.png *.jpg *.JPG)")
    #     if not file_path:
    #         return
    #     self.big_curr = cv2.imread(file_path)
    #     curr_img_h, curr_img_w = self.big_curr.shape[:2]
    #     bottom_height = min(3648, curr_img_h)  # 避免影像本身小於 3000px
    #     self.curr_cropped = self.big_curr[curr_img_h - bottom_height : curr_img_h, 0:curr_img_w]
    #     self.curr_img = cv2.cvtColor(self.curr_cropped, cv2.COLOR_BGR2GRAY)
    #     h, w = self.curr_img.shape[:2]
    #     cx, cy = w // 2, h // 2  # 中心點
    #     x1, x2 = cx - 200, cx + 200
    #     y1, y2 = cy - 200, cy + 200
    #     temp_img = self.curr_img[y1:y2, x1:x2]
    #
    #     curr_show = self.curr_cropped.copy()
    #     cv2.rectangle(curr_show, (x1, y1), (x2, y2), (12, 31, 242), 10)
    #     curr_show = cv2.cvtColor(curr_show, cv2.COLOR_BGR2RGB)
    #     Ny, Nx, channels = curr_show.shape
    #     self.curr_view = QtGui.QImage(curr_show.data, Nx, Ny, QtGui.QImage.Format_RGB888)
    #     self.label_7.setPixmap(QtGui.QPixmap.fromImage(self.curr_view))
    #
    #     h, w = prev_gray.shape[:2]
    #     self.matching_scope = [0, 0], [w, h]
    #     match_img = prev_gray[0:h, 0:w]
    #
    #     methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF','cv2.TM_SQDIFF_NORMED']
    #     method_select = 1
    #     meth = methods[method_select]
    #     method = eval(meth)
    #     res = cv2.matchTemplate(match_img, temp_img, method)
    #     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    #     top_left = max_loc
    #     top_left = [top_left[0] + self.matching_scope[0][0], top_left[1] + self.matching_scope[0][1]]
    #     self.big_stitch_offset = [top_left[0] - x1, top_left[1] - y1]
    #     print(top_left , self.big_stitch_offset)
    #
    #     ## 顯示比對結果
    #     prev_show = self.prev_cropped.copy()
    #     cv2.rectangle(prev_show, (top_left[0], top_left[1]), (top_left[0]+400, top_left[1]+400), (12, 31, 242), 10)
    #     prev_show = cv2.cvtColor(prev_show, cv2.COLOR_BGR2RGB)
    #     Ny, Nx, channels = prev_show.shape
    #     self.prev_view = QtGui.QImage(prev_show.data, Nx, Ny, QtGui.QImage.Format_RGB888)
    #     self.label_6.setPixmap(QtGui.QPixmap.fromImage(self.prev_view))
    #
    #     # 增加修改事件
    #     self.step_flag = 10

    def big_match(self):
        self.label_6.setVisible(True)
        self.label_7.setVisible(True)

        # -------- 選取上方影像 --------
        file_path, _ = QFileDialog.getOpenFileName(self, "選擇圖片檔案", "", "Image Files (*.png *.jpg *.JPG)")
        if not file_path:
            return

        # 保留 alpha 並統一處理通道
        self.big_prev = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        print(f"上方影像形狀: {self.big_prev.shape}")

        # 統一處理上方影像通道
        if len(self.big_prev.shape) == 3 and self.big_prev.shape[2] == 4:
            # BGRA 格式
            b, g, r, a = cv2.split(self.big_prev)
            self.prev_rgb = cv2.merge([b, g, r])
            self.prev_alpha = a
        elif len(self.big_prev.shape) == 3 and self.big_prev.shape[2] == 3:
            # BGR 格式
            self.prev_rgb = self.big_prev.copy()
            self.prev_alpha = None
        else:
            # 灰階圖像，轉為 BGR
            self.prev_rgb = cv2.cvtColor(self.big_prev, cv2.COLOR_GRAY2BGR)
            self.prev_alpha = None

        prev_img_h, prev_img_w = self.prev_rgb.shape[:2]
        top_height = min(3648, prev_img_h)
        self.prev_cropped = self.prev_rgb[0:top_height, 0:prev_img_w]

        if self.prev_alpha is not None:
            self.prev_alpha_cropped = self.prev_alpha[0:top_height, 0:prev_img_w]
        else:
            self.prev_alpha_cropped = None

        prev_gray = cv2.cvtColor(self.prev_cropped, cv2.COLOR_BGR2GRAY)

        # -------- 選取下方影像 --------
        file_path, _ = QFileDialog.getOpenFileName(self, "選擇圖片檔案", "", "Image Files (*.png *.jpg *.JPG)")
        if not file_path:
            return

        self.big_curr = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        print(f"下方影像形狀: {self.big_curr.shape}")

        # 統一處理下方影像通道
        if len(self.big_curr.shape) == 3 and self.big_curr.shape[2] == 4:
            # BGRA 格式
            b, g, r, a = cv2.split(self.big_curr)
            self.curr_rgb = cv2.merge([b, g, r])
            self.curr_alpha = a
        elif len(self.big_curr.shape) == 3 and self.big_curr.shape[2] == 3:
            # BGR 格式
            self.curr_rgb = self.big_curr.copy()
            self.curr_alpha = None
        else:
            # 灰階圖像，轉為 BGR
            self.curr_rgb = cv2.cvtColor(self.big_curr, cv2.COLOR_GRAY2BGR)
            self.curr_alpha = None

        curr_img_h, curr_img_w = self.curr_rgb.shape[:2]
        bottom_height = min(3648, curr_img_h)
        self.curr_cropped = self.curr_rgb[curr_img_h - bottom_height: curr_img_h, 0:curr_img_w]

        if self.curr_alpha is not None:
            self.curr_alpha_cropped = self.curr_alpha[curr_img_h - bottom_height: curr_img_h, 0:curr_img_w]
        else:
            self.curr_alpha_cropped = None

        self.curr_img = cv2.cvtColor(self.curr_cropped, cv2.COLOR_BGR2GRAY)

        # -------- 取中心範圍進行模板匹配 --------
        h, w = self.curr_img.shape[:2]
        cx, cy = w // 2, h // 2
        x1, x2 = cx - 200, cx + 200
        y1, y2 = cy - 200, cy + 200

        # 確保範圍不超出邊界
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        temp_img = self.curr_img[y1:y2, x1:x2]

        # -------- 安全地顯示下方影像與模板框 --------
        def safe_display_image(img, alpha_channel, label):
            """安全地將圖像轉換為 PyQt 可顯示的格式"""
            try:
                # 確保圖像是 BGR 格式
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] != 3:
                    img = img[:, :, :3]  # 只取前三個通道

                # 如果有 alpha 通道，創建 BGRA 圖像
                if alpha_channel is not None and alpha_channel.shape[:2] == img.shape[:2]:
                    img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                    img_rgba[:, :, 3] = alpha_channel
                    img_show = cv2.cvtColor(img_rgba, cv2.COLOR_BGRA2RGBA)
                    format_type = QtGui.QImage.Format_RGBA8888
                else:
                    img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    format_type = QtGui.QImage.Format_RGB888

                # 確保圖像是連續的
                img_show = np.ascontiguousarray(img_show)

                Ny, Nx = img_show.shape[:2]
                channels = img_show.shape[2] if len(img_show.shape) > 2 else 1

                qimage = QtGui.QImage(img_show.data, Nx, Ny, format_type)
                label.setPixmap(QtGui.QPixmap.fromImage(qimage))

            except Exception as e:
                print(f"顯示圖像時發生錯誤: {e}")
                # 備用方案：顯示純色圖像
                fallback_img = np.zeros((300, 300, 3), dtype=np.uint8)
                fallback_img[:] = [128, 128, 128]  # 灰色
                fallback_show = cv2.cvtColor(fallback_img, cv2.COLOR_BGR2RGB)
                qimage = QtGui.QImage(fallback_show.data, 300, 300, QtGui.QImage.Format_RGB888)
                label.setPixmap(QtGui.QPixmap.fromImage(qimage))

        # 顯示下方影像與模板框
        curr_show = self.curr_cropped.copy()
        cv2.rectangle(curr_show, (x1, y1), (x2, y2), (12, 31, 242), 10)
        safe_display_image(curr_show, self.curr_alpha_cropped, self.label_7)

        # -------- 模板匹配 --------
        h, w = prev_gray.shape[:2]
        self.matching_scope = [0, 0], [w, h]
        match_img = prev_gray[0:h, 0:w]

        methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF',
                   'cv2.TM_SQDIFF_NORMED']
        method_select = 1
        method = eval(methods[method_select])

        try:
            res = cv2.matchTemplate(match_img, temp_img, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            top_left = max_loc
            top_left = [top_left[0] + self.matching_scope[0][0], top_left[1] + self.matching_scope[0][1]]
            self.big_stitch_offset = [top_left[0] - x1, top_left[1] - y1]
            print(f"匹配位置: {top_left}, 拼接偏移: {self.big_stitch_offset}")
        except Exception as e:
            print(f"模板匹配時發生錯誤: {e}")
            # 設定預設偏移
            top_left = [x1, y1]
            self.big_stitch_offset = [0, 0]

        # -------- 顯示上方影像匹配結果 --------
        prev_show = self.prev_cropped.copy()
        # 確保矩形範圍不超出圖像邊界
        rect_x1, rect_y1 = top_left[0], top_left[1]
        rect_x2, rect_y2 = top_left[0] + 400, top_left[1] + 400
        rect_x1 = max(0, min(rect_x1, self.prev_cropped.shape[1] - 1))
        rect_y1 = max(0, min(rect_y1, self.prev_cropped.shape[0] - 1))
        rect_x2 = max(rect_x1 + 1, min(rect_x2, self.prev_cropped.shape[1]))
        rect_y2 = max(rect_y1 + 1, min(rect_y2, self.prev_cropped.shape[0]))

        cv2.rectangle(prev_show, (rect_x1, rect_y1), (rect_x2, rect_y2), (12, 31, 242), 10)
        safe_display_image(prev_show, self.prev_alpha_cropped, self.label_6)

        # -------- 設定步驟旗標 --------
        self.step_flag = 10

    # def big_stitch(self):
    #     prev_img_h, prev_img_w = self.big_prev.shape[:2]
    #     curr_img_h, curr_img_w = self.big_curr.shape[:2]
    #     canvas_w = max(prev_img_w, self.big_stitch_offset[0] + curr_img_w)
    #     canvas_h = max(prev_img_h, self.big_stitch_offset[1] + curr_img_h)
    #     canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    #     canvas[0:prev_img_h, 0:prev_img_w] = self.big_prev
    #     # === 安全拼接 ===
    #     x, y = self.big_stitch_offset
    #
    #     # 如果有負 offset，就計算需要往右下補多少
    #     offset_x = max(0, -x)
    #     offset_y = max(0, -y)
    #
    #     # 計算新的畫布大小
    #     canvas_w = max(prev_img_w + offset_x, x + curr_img_w + offset_x)
    #     canvas_h = max(prev_img_h + offset_y, y + curr_img_h + offset_y)
    #
    #     canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    #
    #     # 放 prev_img
    #     canvas[offset_y:offset_y + prev_img_h, offset_x:offset_x + prev_img_w] = self.big_prev
    #
    #     # 放 curr_img
    #     x_new = x + offset_x
    #     y_new = y + offset_y
    #
    #     # 計算重疊區域
    #     y1, y2 = max(y_new, offset_y), min(y_new + curr_img_h, canvas_h)
    #     x1, x2 = max(x_new, offset_x), min(x_new + curr_img_w, canvas_w)
    #
    #     # 先放 curr_img
    #     canvas[y_new:y_new + curr_img_h, x_new:x_new + curr_img_w] = self.big_curr
    #
    #     # 融合重疊區域
    #     overlap_canvas = canvas[y1:y2, x1:x2]
    #     overlap_curr = self.big_curr[y1 - y_new:y2 - y_new, x1 - x_new:x2 - x_new]
    #     canvas[y1:y2, x1:x2] = (overlap_canvas // 2 + overlap_curr // 2)
    #
    #     cv2.imwrite('./0825test.jpg', canvas)

    def big_stitch(self):
        # 處理圖像通道，確保格式統一
        def process_image(img):
            if len(img.shape) == 3 and img.shape[2] == 4:
                # BGRA 格式，分離 BGR 和 Alpha
                bgr = img[:, :, :3]
                alpha = img[:, :, 3]
                return bgr, alpha
            elif len(img.shape) == 3 and img.shape[2] == 3:
                # BGR 格式，沒有 Alpha
                return img, None
            else:
                # 灰階，轉為 BGR
                bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                return bgr, None

        # 處理兩張大圖
        prev_bgr, prev_alpha = process_image(self.big_prev)
        curr_bgr, curr_alpha = process_image(self.big_curr)

        prev_img_h, prev_img_w = prev_bgr.shape[:2]
        curr_img_h, curr_img_w = curr_bgr.shape[:2]

        # 原始算法：安全拼接
        x, y = self.big_stitch_offset

        # 如果有負 offset，就計算需要往右下補多少
        offset_x = max(0, -x)
        offset_y = max(0, -y)

        # 計算新的畫布大小
        canvas_w = max(prev_img_w + offset_x, x + curr_img_w + offset_x)
        canvas_h = max(prev_img_h + offset_y, y + curr_img_h + offset_y)

        # 判斷是否有透明通道
        has_alpha = (prev_alpha is not None) or (curr_alpha is not None)

        if has_alpha:
            # 創建 BGRA 畫布
            canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
            # 如果沒有 alpha 通道，創建全不透明的
            if prev_alpha is None:
                prev_alpha = np.full((prev_img_h, prev_img_w), 255, dtype=np.uint8)
            if curr_alpha is None:
                curr_alpha = np.full((curr_img_h, curr_img_w), 255, dtype=np.uint8)
        else:
            # 創建 BGR 畫布
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # 放 prev_img
        canvas[offset_y:offset_y + prev_img_h, offset_x:offset_x + prev_img_w, :3] = prev_bgr
        if has_alpha:
            canvas[offset_y:offset_y + prev_img_h, offset_x:offset_x + prev_img_w, 3] = prev_alpha

        # 放 curr_img
        x_new = x + offset_x
        y_new = y + offset_y

        # 計算重疊區域
        y1, y2 = max(y_new, offset_y), min(y_new + curr_img_h, canvas_h)
        x1, x2 = max(x_new, offset_x), min(x_new + curr_img_w, canvas_w)

        # 先放 curr_img
        canvas[y_new:y_new + curr_img_h, x_new:x_new + curr_img_w, :3] = curr_bgr
        if has_alpha:
            canvas[y_new:y_new + curr_img_h, x_new:x_new + curr_img_w, 3] = curr_alpha

        # 融合重疊區域
        if y2 > y1 and x2 > x1:  # 確保有重疊區域
            overlap_canvas = canvas[y1:y2, x1:x2, :3]
            overlap_curr = curr_bgr[y1 - y_new:y2 - y_new, x1 - x_new:x2 - x_new]
            canvas[y1:y2, x1:x2, :3] = (overlap_canvas // 2 + overlap_curr // 2)

            # 如果有 alpha 通道，也融合 alpha
            if has_alpha:
                overlap_canvas_alpha = canvas[y1:y2, x1:x2, 3]
                overlap_curr_alpha = curr_alpha[y1 - y_new:y2 - y_new, x1 - x_new:x2 - x_new]
                # Alpha 通道使用最大值（保持不透明）
                canvas[y1:y2, x1:x2, 3] = np.maximum(overlap_canvas_alpha, overlap_curr_alpha)

        # 根據是否有透明通道選擇保存格式
        if has_alpha:
            cv2.imwrite('./0825test.png', canvas)
            print("已保存為 PNG 格式（含透明通道）")
        else:
            cv2.imwrite('./0825test.jpg', canvas)
            print("已保存為 JPG 格式")
    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self,"選擇圖片檔案","","Image Files (*.png *.jpg *.JPG)")
        if not file_path:
            return
        self.for_select_path = file_path
        image = cv2.imread(file_path)
        # 將 BGR 轉成 RGB（OpenCV → Qt）
        resized_img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 轉換成 QImage
        h, w, ch = resized_img_rgb.shape
        bytes_per_line = ch * w
        q_image = QImage(resized_img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # 轉為 QPixmap
        self.original_pixmap = QPixmap.fromImage(q_image)
        self.graphicsView.setVisible(True)
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)
        self.thumbnail_pixmap = self.original_pixmap.scaled(self.original_pixmap.width(),self.original_pixmap.height(), Qt.KeepAspectRatio)
        self.pixmap_item = self.scene.addPixmap(self.thumbnail_pixmap)
        self.step_flag = 200

    # 0804------------------------------------------------
    def print_watchdog_info_list(self):
        print("=== Watchdog Info List ===")
        for i, info in enumerate(self.watchdog_info_list):
            print(f"[{i}]")
            print(f"  Path: {info.path}")
            print(f"  GPS: {info.gps}")
            print(f"  Direction: {info.direction}")
            print(f"  Match Coord: {info.match_coordinate}")
            print(f"  Stitch Coord: {info.stitch_coordinate}")
            print(f"  Revise: {info.revise}")
            print(f"  Distance: {info.distance}")
            print("-" * 30)

    def toggle_reading(self):
        if self.matching_busy:  # 如果正在處理中，先不允許切換
            logging.warning("正在處理圖片，請稍後再切換")
            return
        self.reading_active = not self.reading_active
        self.loader_thread.reading_active = self.reading_active  # 同步更新到 thread

        if self.reading_active:
            self.pushButton_test.setText("停止讀圖")
            logging.info("開始讀取圖片")
        else:
            self.pushButton_test.setText("開始讀圖")
            logging.info("停止讀取圖片（僅監控）")
    # 0709------------------------------------------------
    def handle_new_image(self, path, image):
        # 檢查圖片是否已存在
        if any(info.path == path for info in self.watchdog_info_list):
            logging.info(f"[略過] {path} 已存在於列表中，跳過處理")
            self.ready_to_match = True  # 重新啟用
            return
            
        # 檢查各種狀態
        if getattr(self, "rebuilding", False):
            logging.warning("表格正在重建，暫停新增圖片")
            self.ready_to_match = True  # 重新啟用
            return
            
        if self.rematch_running:
            logging.warning("目前正在重新比對中，暫停自動比對")
            self.ready_to_match = True  # 重新啟用
            return
        
        # 建立圖片資訊並加入列表
        gps = get_gps_coordinates(path)
        info = ImageInfo(path=path, image=image, gps=gps)
        self.watchdog_info_list.append(info)
        self.update_table(info)
        
        # 如果有前一張圖片，加入比對佇列
        if len(self.watchdog_info_list) >= 2:
            prev = self.watchdog_info_list[-2]
            curr = self.watchdog_info_list[-1]
            
            # 避免重複加入相同的比對對
            if (prev, curr) not in self.pending_pairs:
                self.pending_pairs.append((prev, curr))
                logging.info(f"[佇列] 加入比對對：{os.path.basename(prev.path)} <-> {os.path.basename(curr.path)}")
        
        # 嘗試開始比對（只在沒有忙碌時）
        if not self.matching_busy:
            self.try_start_match()
        
        self.ready_to_match = True  # 重新啟用

    def start_match_if_ready(self, prev, curr):
        if self.matching_busy:
            # 如果正在匹配，加入佇列等待
            if (prev, curr) not in self.pending_pairs:
                self.pending_pairs.append((prev, curr))
            return

        distance = gps_distance(prev.gps, curr.gps)
        curr.distance = distance
        direction_quadrants = calculate_direction(prev.gps, curr.gps)

        if direction_quadrants == int(self.lineEdit_up.text()):
            curr.direction = 'up'
        elif direction_quadrants == int(self.lineEdit_down.text()):
            curr.direction = 'down'
        elif direction_quadrants == int(self.lineEdit_left.text()):
            curr.direction = 'left'
        elif direction_quadrants == int(self.lineEdit_right.text()):
            curr.direction = 'right'

        logging.info(f"方向: prev={prev.direction}, curr={curr.direction}, 距離={curr.distance:.2f}m")

        prev_match_coord = prev.match_coordinate if prev.direction == curr.direction else None
        curr_match_coord = curr.match_coordinate if prev.direction == curr.direction else None


        # 確保前一個線程已經結束
        if self.current_match_thread and self.current_match_thread.isRunning():
            self.current_match_thread.quit()
            self.current_match_thread.wait()

        # 取得圖片尺寸
        self.img_h, self.img_w = curr.image.shape[:2]

        # 只在第一次比對時檢查template座標
        if not hasattr(self, 'template_checked'):
            self.template_checked = True
            for template_name in self.conf['template_area']:
                template_coords = self.conf['template_area'][template_name]
                # 檢查x1, y1, x2, y2是否超出圖片範圍
                if (template_coords['x1'] >= self.img_w or
                        template_coords['y1'] >= self.img_h or
                        template_coords['x2'] >= self.img_w or
                        template_coords['y2'] >= self.img_h or
                        template_coords['x1'] < 0 or
                        template_coords['y1'] < 0 or
                        template_coords['x2'] < 0 or
                        template_coords['y2'] < 0):
                    logging.warning(
                        f"警告: {template_name} 座標超出圖片範圍 (圖片大小: {self.img_w}x{self.img_h})，已重設為安全值")
                    # 自適應設定座標，以圖片中心為起點
                    center_x = self.img_w // 2
                    center_y = self.img_h // 2
                    template_coords['x1'] = center_x
                    template_coords['y1'] = center_y
                    # 確保x2, y2不會超出圖片邊界
                    template_coords['x2'] = min(center_x + 100, self.img_w - 1)
                    template_coords['y2'] = min(center_y + 100, self.img_h - 1)

        # 根據圖片數量決定是否啟用按鈕
        if len(self.watchdog_info_list) == 2:
            prev.direction = curr.direction
            self.pushButton_show.setEnabled(True)
            self.pushButton_rematching.setEnabled(True)
            self.pushButton_stitch.setEnabled(True)
        if len(self.watchdog_info_list) == 3:
            self.show_template()

        self.matching_busy = True
        self.current_match_thread = TemplateMatcherThread(
            prev.image, curr.image, curr.direction, prev_match_coord,curr_match_coord,curr.revise, self.conf
        )
        self.current_match_thread.list_signal.connect(
            lambda match_coord, stitch_coord, target=curr:
            self.update_match_coordinate(target, match_coord, stitch_coord)
        )
        self.current_match_thread.finished.connect(self.on_match_thread_finished)
        self.current_match_thread.start()

    def on_match_thread_finished(self):
        """線程結束時的回調"""
        self.matching_busy = False
        # 清理線程引用
        if self.current_match_thread:
            self.current_match_thread.deleteLater()
            self.current_match_thread = None
        
        # 08/01  避免太快Process finished with exit code -1073741819 (0xC0000005)
        # self.try_start_match()
        QTimer.singleShot(100, self.try_start_match)  # 100ms = 0.1秒
    
    def try_start_match(self):
        """改成FIFO比對佇列，並加入線程狀態檢查"""
        if self.matching_busy or not self.pending_pairs:
            return
        
        # 確保當前沒有線程在執行
        if self.current_match_thread and self.current_match_thread.isRunning():
            return
            
        prev, curr = self.pending_pairs.pop(0)
        logging.info(f"[佇列處理] 開始比對：{os.path.basename(prev.path)} <-> {os.path.basename(curr.path)}, 剩餘佇列: {len(self.pending_pairs)}")
        self.start_match_if_ready(prev, curr)
    
    def update_match_coordinate(self, image_info, match_coord, stitch_coord):
        if match_coord and (match_coord[0] < 0 or match_coord[1] < 0):
            logging.warning(f"[座標修正] {os.path.basename(image_info.path)} 偵測到負座標 {match_coord}，重設為預設值 [-1, -1]")
            match_coord = [-1, -1]
            stitch_coord = [-1, -1]

        image_info.match_coordinate = match_coord
        image_info.stitch_coordinate = stitch_coord

        if match_coord == [-1, -1]:
            logging.warning(f"[比對結果] {os.path.basename(image_info.path)} 匹配失敗，座標已重設為預設值")
        else:
            logging.info(f"[比對成功] {os.path.basename(image_info.path)} 匹配座標：{match_coord} 相對座標: {stitch_coord}")

        self.update_table(image_info)
        
        # 如果不是重比對模式，將下一張圖片加入佇列
        if not getattr(self, 'rematch_running', False):
            idx = self.watchdog_info_list.index(image_info)
            if idx + 1 < len(self.watchdog_info_list):
                next_img = self.watchdog_info_list[idx + 1]
                if (image_info, next_img) not in self.pending_pairs:
                    self.pending_pairs.append((image_info, next_img))
                    logging.info(f"[佇列] 加入下一組比對：{os.path.basename(image_info.path)} <-> {os.path.basename(next_img.path)}")

    def show_template(self):
        self.label_6.setVisible(True)
        self.label_7.setVisible(True)
        self.tableWidget.setEnabled(True)

        if self.step_flag == 0 :
            logging.info("[step_flag] 切換至 200 ： 顯示比對結果")
            self.step_flag = 200

        temp_up_w = self.conf['template_area']['temp_up']['x2'] - self.conf['template_area']['temp_up']['x1']
        temp_up_h = self.conf['template_area']['temp_up']['y2'] - self.conf['template_area']['temp_up']['y1']
        temp_down_w = self.conf['template_area']['temp_down']['x2'] - self.conf['template_area']['temp_down']['x1']
        temp_down_h = self.conf['template_area']['temp_down']['y2'] - self.conf['template_area']['temp_down']['y1']
        temp_left_w = self.conf['template_area']['temp_left']['x2'] - self.conf['template_area']['temp_left']['x1']
        temp_left_h = self.conf['template_area']['temp_left']['y2'] - self.conf['template_area']['temp_left']['y1']
        temp_right_w = self.conf['template_area']['temp_right']['x2'] - self.conf['template_area']['temp_right']['x1']
        temp_right_h = self.conf['template_area']['temp_right']['y2'] - self.conf['template_area']['temp_right']['y1']

        if len(self.watchdog_info_list) >= self.show_count + 2 :
            prev = self.watchdog_info_list[self.show_count]
            curr = self.watchdog_info_list[self.show_count+1]
            # 下圖 template
            if curr.direction == 'up':
                curr_img = curr.image.copy()
                cv2.rectangle(curr_img,
                              (self.conf['template_area']['temp_up']['x1'],
                               self.conf['template_area']['temp_up']['y1']),
                              (self.conf['template_area']['temp_up']['x2'],
                               self.conf['template_area']['temp_up']['y2']), (0, 255, 255), 10)
                prev_img = prev.image.copy()
                cv2.rectangle(prev_img, (curr.match_coordinate[0],curr.match_coordinate[1]),
                                         (curr.match_coordinate[0] + temp_up_w, curr.match_coordinate[1] + temp_up_h),(12, 31, 242), 10)
            if curr.direction == 'down':
                curr_img = curr.image.copy()
                cv2.rectangle(curr_img,
                              (self.conf['template_area']['temp_down']['x1'],
                               self.conf['template_area']['temp_down']['y1']),
                              (self.conf['template_area']['temp_down']['x2'],
                               self.conf['template_area']['temp_down']['y2']), (0, 255, 255), 10)
                prev_img = prev.image.copy()
                cv2.rectangle(prev_img, (curr.match_coordinate[0],curr.match_coordinate[1]),
                                         (curr.match_coordinate[0] + temp_down_w, curr.match_coordinate[1] + temp_down_h),(12, 31, 242), 10)
            if curr.direction == 'left':
                curr_img = curr.image.copy()
                cv2.rectangle(curr_img,
                              (self.conf['template_area']['temp_left']['x1'],
                               self.conf['template_area']['temp_left']['y1']),
                              (self.conf['template_area']['temp_left']['x2'],
                               self.conf['template_area']['temp_left']['y2']), (0, 255, 255), 10)
                prev_img = prev.image.copy()
                cv2.rectangle(prev_img, (curr.match_coordinate[0],curr.match_coordinate[1]),
                                         (curr.match_coordinate[0] + temp_left_w, curr.match_coordinate[1] + temp_left_h),(12, 31, 242), 10)
            if curr.direction == 'right':
                curr_img = curr.image.copy()
                cv2.rectangle(curr_img,
                              (self.conf['template_area']['temp_right']['x1'],
                               self.conf['template_area']['temp_right']['y1']),
                              (self.conf['template_area']['temp_right']['x2'],
                               self.conf['template_area']['temp_right']['y2']), (0, 255, 255), 10)
                prev_img = prev.image.copy()
                cv2.rectangle(prev_img, (curr.match_coordinate[0],curr.match_coordinate[1]),
                                         (curr.match_coordinate[0] + temp_right_w, curr.match_coordinate[1] + temp_right_h),(12, 31, 242), 10)
            # 右圖 curr
            if self.step_flag == 300:
                logging.info("[step_flag] 切換至 300 ： 修改template位置")
                cv2.rectangle(curr_img, (self.rect_start[0], self.rect_start[1]),(self.rect_end[0], self.rect_end[1]), (22, 103, 242), 10)
            curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB)
            Ny, Nx, channels = curr_img.shape
            self.curr_view = QtGui.QImage(curr_img.data, Nx, Ny, QtGui.QImage.Format_RGB888)
            self.label_7.setPixmap(QtGui.QPixmap.fromImage(self.curr_view))

            # 左圖
            prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2RGB)
            Ny, Nx, channels = prev_img.shape
            self.prev_view = QtGui.QImage(prev_img.data, Nx, Ny, QtGui.QImage.Format_RGB888)
            self.label_6.setPixmap(QtGui.QPixmap.fromImage(self.prev_view))
            self.show_count = self.show_count +1

            # 預覽拼接狀況
            this_stitch_coordinate = [[0,0], curr.stitch_coordinate]
            this_image = [prev.image, curr.image]
            final_sti_img, final_coordinate_list,center_coordinate_list = stitch_seg_images(this_stitch_coordinate,this_image)
            # resized_img = cv2.resize(final_sti_img, (final_sti_img.shape[1] // 8, final_sti_img.shape[0] // 8))
            resized_img = cv2.resize(final_sti_img, (final_sti_img.shape[1]//2, final_sti_img.shape[0]//2))


            # 將 BGR 轉成 RGB（OpenCV → Qt）
            resized_img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            # 轉換成 QImage
            h, w, ch = resized_img_rgb.shape
            bytes_per_line = ch * w
            q_image = QImage(resized_img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # 轉為 QPixmap
            self.original_pixmap = QPixmap.fromImage(q_image)
            self.graphicsView.setVisible(True)
            self.scene = QGraphicsScene()
            self.graphicsView.setScene(self.scene)
            self.thumbnail_pixmap = self.original_pixmap.scaled(self.original_pixmap.width(),self.original_pixmap.height(),Qt.KeepAspectRatio)
            self.pixmap_item = self.scene.addPixmap(self.thumbnail_pixmap)
    
    def rematch_selected(self):
        """重新比對勾選範圍"""
        if self.matching_busy:
            logging.warning(f"比對進行中，請稍候完成再重跑")
            return
        checked_rows = []
        for row in range(self.tableWidget.rowCount()):
            checkbox_widget = self.tableWidget.cellWidget(row, 0)
            if checkbox_widget:
                checkbox = checkbox_widget.findChild(QtWidgets.QCheckBox)
                if checkbox and checkbox.isChecked():
                    checked_rows.append(row)
        if len(checked_rows) < 2:
            logging.warning(f"[選擇範圍] 勾選圖片數量不足")
            QtWidgets.QMessageBox.information(self, "提醒", "勾選圖片數量不足")
            return
        logging.info(f"[重新比對選擇範圍] 開始，範圍: {checked_rows}")
        #  鎖定新圖處理
        self.rematch_running = True
        self.loader_thread.blockSignals(True)

        checked_indices = sorted(checked_rows)

        # 只對勾選範圍內相鄰的圖片進行比對
        for i in range(len(checked_indices) - 1):
            curr_idx = checked_indices[i + 1]  # 當前圖片索引
            prev_idx = checked_indices[i]  # 前一張圖片索引

            # 確保索引在有效範圍內
            if curr_idx >= len(self.watchdog_info_list) or prev_idx >= len(self.watchdog_info_list):
                logging.warning(f"索引超出範圍: prev_idx={prev_idx}, curr_idx={curr_idx}")
                continue

            prev = self.watchdog_info_list[prev_idx]
            curr = self.watchdog_info_list[curr_idx]

            # if curr.revise:  # 跳過手動修改過的
            #     logging.info(f"跳過 {os.path.basename(curr.path)}（已手動修改）")
            #     continue

            logging.info(f"[比對前] {os.path.basename(curr.path)} match_coordinate: {curr.match_coordinate}")
            prev_match_coord = prev.match_coordinate if prev.direction == curr.direction else None
            curr_match_coord = curr.match_coordinate if prev.direction == curr.direction else None
            self.matching_busy = True  # 加鎖
            match_thread = TemplateMatcherThread(prev.image, curr.image, curr.direction, prev_match_coord, curr_match_coord, curr.revise, self.conf)
            # 用 QEventLoop 讓主線程等待 thread 執行完成
            loop = QtCore.QEventLoop()

            def create_match_result_handler(target_obj):
                def match_result(match_coord, stitch_coord):
                    logging.info(f"[比對後] {os.path.basename(target_obj.path)} match_coordinate: {match_coord}")
                    self.update_match_coordinate(target_obj, match_coord, stitch_coord)
                    self.matching_busy = False  # 釋放鎖
                    loop.quit()  # 中斷等待
                return match_result

            match_result_handler = create_match_result_handler(curr)

            match_thread.list_signal.connect(match_result_handler)
            match_thread.finished.connect(loop.quit)  # 萬一有例外也保底退出 loop
            match_thread.start()

            # 等待此 thread 結束
            loop.exec_()
            match_thread.wait()  # 確保 QThread 完全釋放

        logging.info(f"[重新比對選擇範圍] 完成，範圍: {checked_rows}")
        self.loader_thread.blockSignals(False)
        self.rematch_running = False

        self.show_count = self.show_count - 1
        self.show_template()
    
    def rematch_all(self):
        if self.matching_busy:
            logging.warning(f"比對進行中，請稍候完成再重跑")
            return
        logging.info("[重新比對全部] 開始")
        #  鎖定新圖處理
        self.rematch_running = True
        self.loader_thread.blockSignals(True)
        for i in range(1, len(self.watchdog_info_list)):
            prev = self.watchdog_info_list[i - 1]
            curr = self.watchdog_info_list[i]
            if curr.revise:  # 跳過手動修改過的
                logging.info(f"跳過 {os.path.basename(curr.path)}（已手動修改）")
                continue
            logging.info(f"[比對前] {os.path.basename(curr.path)} match_coordinate: {curr.match_coordinate}")
            prev_match_coord = prev.match_coordinate if prev.direction == curr.direction else None
            self.matching_busy = True  # 加鎖
            match_thread = TemplateMatcherThread(prev.image, curr.image, curr.direction, prev_match_coord, self.conf)
            # 用 QEventLoop 讓主線程等待 thread 執行完成
            loop = QtCore.QEventLoop()
            
            def match_result(match_coord, stitch_coord, target=curr):
                logging.info(f"[比對後] {os.path.basename(target.path)} match_coordinate: {match_coord}")
                self.update_match_coordinate(target, match_coord, stitch_coord)
                loop.quit() # 中斷等待
            match_thread.list_signal.connect(match_result)
            match_thread.finished.connect(loop.quit)  # 萬一有例外也保底退出 loop
            match_thread.start()
            # 等待此 thread 結束
            loop.exec_()
            match_thread.wait()  # 確保 QThread 完全釋放
            while self.matching_busy:
                QtWidgets.QApplication.processEvents()
        logging.info("[重新比對全部] 完成")
        self.loader_thread.blockSignals(False)
        self.rematch_running = False
        # 若有新圖未比對，自動補一次
        if len(self.watchdog_info_list) >= 2:
            prev = self.watchdog_info_list[-2]
            curr = self.watchdog_info_list[-1]
            if curr.match_coordinate == [-1, -1] and not curr.revise:
                self.start_match_if_ready(prev, curr)
        self.show_count = self.show_count - 1
        self.show_template()

    def stitch_class(self):
        checked_rows = []
        for row in range(self.tableWidget.rowCount()):
            checkbox_widget = self.tableWidget.cellWidget(row, 0)
            if checkbox_widget:
                checkbox = checkbox_widget.findChild(QtWidgets.QCheckBox)
                if checkbox and checkbox.isChecked():
                    checked_rows.append(row)
        if len(checked_rows) < 2:
            logging.warning(f"[選擇範圍] 勾選圖片數量不足")
            QtWidgets.QMessageBox.information(self, "提醒", "勾選圖片數量不足")
            return
        logging.info(f"[選擇範圍] {checked_rows}")
        stitch_coordinate_list = [[0,0]]
        path_list = []
        gps_list = []
        for i in range (0,len(checked_rows)):
            info = self.watchdog_info_list[checked_rows[i]]
            path_list.append(info.path)
            gps_list.append(info.gps)
            if i != 0:
                stitch_coordinate_list.append(info.stitch_coordinate)
            # if i == len(checked_rows)-1:
            #     last_match_coordinate = info.match_coordinate
        # 因為不要複製info.img到其他線程造成UI卡住，所以只傳需要的list
        self.stitch_thread = StitchThread(path_list,gps_list, stitch_coordinate_list)
        self.stitch_thread.result_signal.connect(self.show_stitch_result)
        self.stitch_thread.start()

    def show_stitch_result(self, img):
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        bytes_per_line = channel * width
        q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.original_pixmap = QPixmap.fromImage(q_image)
        self.graphicsView.setVisible(True)
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)
        self.scale_factor = 1
        self.thumbnail_pixmap = self.original_pixmap.scaled(int(self.original_pixmap.width() * self.scale_factor),
                                                            int(self.original_pixmap.height() * self.scale_factor),
                                                            Qt.KeepAspectRatio)
        self.pixmap_item = self.scene.addPixmap(self.thumbnail_pixmap)
        # cv2.imshow("Stitch Preview", img)
        self.pixel_pts = np.empty((0, 2), dtype=np.float32)
        self.geo_pts = np.empty((0, 2), dtype=np.float64)
        self.for_select_path = f'./resize_img.png'

    # 交換順序模式
    def toggle_change_mode(self):
        if self.step_flag != 500:
            self.step_flag = 500
            self.btn_change.setStyleSheet("background-color: orange; color: white; font-weight: bold;")
        else:
            self.step_flag = 200
            self.btn_change.setStyleSheet("background-color: rgb(0, 70, 176); color: white;")

    def closeEvent(self, event):
        self.loader_thread.stop()
        self.loader_thread.wait()
        event.accept()

    def open_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "選擇資料夾")
        if not folder:
            return

        # 1. 停止所有正在執行的線程
        self.stop_all_threads()

        # 2. 清空舊的紀錄
        self.watchdog_info_list.clear()
        self.pending_pairs.clear()
        self.tableWidget.setRowCount(0)
        if hasattr(self, 'template_checked'):
            delattr(self, 'template_checked')

        # 3. 停用按鈕，顯示載入中狀態
        self.pushButton_show.setEnabled(False)
        self.pushButton_rematching.setEnabled(False)
        self.pushButton_stitch.setEnabled(False)

        # 4. 建立並啟動圖片載入線程
        supported_ext = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
        self.image_loader_thread = ImageLoaderThread(folder, supported_ext)

        # 連接信號到對應的處理函數
        self.image_loader_thread.image_loaded.connect(self.handle_loaded_image)
        self.image_loader_thread.progress_updated.connect(self.handle_progress_update)
        self.image_loader_thread.loading_finished.connect(self.handle_loading_finished)

        # 啟動線程
        self.image_loader_thread.start()

        # 5. 重新監控新資料夾
        self.loader_thread = WatchdogLoader(folder)
        self.loader_thread.image_loaded.connect(self.handle_new_image)
        self.loader_thread.start()
    def handle_loaded_image(self, info):
        """處理載入完成的單張圖片"""
        self.watchdog_info_list.append(info)
        self.update_table(info)
    def handle_progress_update(self, current, total):
        """處理進度更新（可選）"""
        # 如果你有進度條，可以在這裡更新
        self.progressBar.setValue(int(current * 100 / total))
        # print(f"載入進度: {current}/{total}")
    def handle_loading_finished(self):
        """處理全部圖片載入完成"""
        # 隱藏進度條
        # self.progressBar.setVisible(False)

        # 建立比對佇列
        for i in range(1, len(self.watchdog_info_list)):
            prev = self.watchdog_info_list[i - 1]
            curr = self.watchdog_info_list[i]
            self.pending_pairs.append((prev, curr))

        # 開始第一個比對
        self.try_start_match()

        # 啟用按鈕
        if len(self.watchdog_info_list) >= 2:
            curr = self.watchdog_info_list[-1]  # 取得最後一張圖片
            self.img_h, self.img_w = curr.image.shape[:2]
            self.pushButton_show.setEnabled(True)
            self.pushButton_rematching.setEnabled(True)
            self.pushButton_stitch.setEnabled(True)

    def stop_all_threads(self):
        """停止所有正在執行的線程"""
        # 停止舊的 watchdog
        if hasattr(self, 'loader_thread') and self.loader_thread.isRunning():
            self.loader_thread.stop()
            self.loader_thread.wait(3000)  # 最多等待3秒
            if self.loader_thread.isRunning():
                self.loader_thread.terminate()
    
        # 停止當前的匹配線程
        if self.current_match_thread and self.current_match_thread.isRunning():
            self.current_match_thread.quit()
            self.current_match_thread.wait(3000)  # 最多等待3秒
            if self.current_match_thread.isRunning():
                self.current_match_thread.terminate()
            self.current_match_thread = None
    
        self.matching_busy = False
    # ----------------------------------------------------

    def open_config(self):
        self.widget.setVisible(not self.widget.isVisible())
        self.set_lineedit()
        # self.step_flag = 500  # 調動順序
    def start_selection(self, event):
        if event.button() == Qt.LeftButton:
            self.start_pos = self.graphicsView.mapToScene(event.pos())
            point = [
                int(self.start_pos.x()),
                int(self.start_pos.y())
            ]
            print(f'點擊{point}')
            if self.step_flag == 200 or self.step_flag == 300:
                dialog = LatLngDialog(self)
                if dialog.exec_() == QDialog.Accepted:
                    lat, lng = dialog.get_values()
                    if lat and lng:
                        # 加入像素點
                        pixel_pt = np.array([[point[0], point[1]]], dtype=np.float32)
                        self.pixel_pts = np.vstack([self.pixel_pts, pixel_pt])

                        # 加入經緯度點
                        geo_pt = np.array([[float(lng), float(lat)]], dtype=np.float64)  # 經度在前
                        self.geo_pts = np.vstack([self.geo_pts, geo_pt])

                        print("目前像素點:", self.pixel_pts)
                        print("目前經緯度點:", self.geo_pts)
                '''
                # 當集滿三點時進行計算
                if len(self.pixel_pts) == 3 and len(self.geo_pts) == 3:
                    A = np.vstack([self.pixel_pts.T, np.ones(3)]).T  # 每行 [x, y, 1]
                    lons = self.geo_pts[:, 0]
                    lats = self.geo_pts[:, 1]
                    a, b, c = solve(A, lons)  # 解 a, b, c
                    d, e, f = solve(A, lats)  # 解 d, e, f
                    geotransform = (c, a, b, f, d, e)
                    print(geotransform)
                    png = gdal.Open(self.for_select_path)
                    width = png.RasterXSize
                    height = png.RasterYSize
                    driver = gdal.GetDriverByName('GTiff')
                    geotiff = driver.Create(f'./manual.tiff', width, height, png.RasterCount, gdal.GDT_Byte)
                    geotiff.SetGeoTransform(geotransform)
                    srs = osr.SpatialReference()
                    srs.ImportFromEPSG(4326)
                    geotiff.SetProjection(srs.ExportToWkt())
                    for i in range(1, png.RasterCount + 1):
                        geotiff.GetRasterBand(i).WriteArray(png.GetRasterBand(i).ReadAsArray())
                '''
            if self.step_flag == 4:
                self.point_list.append(point)
                if len(self.point_list) == 4:
                    print("原始點:", self.point_list)
                    sort_key = create_sort_key(self.point_list)
                    sorted_points = sorted(self.point_list, key=sort_key)
                    min_x = max(0, min(p[0] for p in sorted_points))
                    max_x = min(self.cur_img.shape[1], max(p[0] for p in sorted_points))
                    min_y = max(0, min(p[1] for p in sorted_points))
                    max_y = min(self.cur_img.shape[0], max(p[1] for p in sorted_points))
                    # 2025/06/11
                    self.cur_img = self.cur_img[min_y:max_y, min_x:max_x]
                    # cv2.imwrite('./test_img_save/final_merged_stitched.png', self.cur_img,[cv2.IMWRITE_PNG_COMPRESSION, 0])
                    # 依照圖片大小儲存圖片
                    height, width = self.cur_img.shape[:2]
                    max_dim = max(height, width)
                    if max_dim > 65535:
                        cv2.imwrite('./test_img_save/final_merged_stitched.png', self.cur_img,[cv2.IMWRITE_PNG_COMPRESSION, 0])
                    else:
                        cv2.imwrite('./test_img_save/final_merged_stitched.jpg', self.cur_img)
                    # 額外再存一張resize 1/16
                    resized_img = cv2.resize(self.cur_img, (self.cur_img.shape[1] // 4, self.cur_img.shape[0] // 4))
                    resized_path = './resized_input.png'
                    cv2.imwrite(resized_path, resized_img)

                    rgb_image = cv2.cvtColor(self.cur_img, cv2.COLOR_BGR2RGB)
                    height, width, channel = rgb_image.shape
                    bytes_per_line = channel * width
                    q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
                    self.original_pixmap = QPixmap.fromImage(q_image)
                    self.graphicsView.setVisible(True)
                    self.scene = QGraphicsScene()
                    self.graphicsView.setScene(self.scene)
                    self.scale_factor = 0.25
                    self.thumbnail_pixmap = self.original_pixmap.scaled(int(self.original_pixmap.width() * self.scale_factor),
                                                                        int(self.original_pixmap.height() * self.scale_factor),
                                                                        Qt.KeepAspectRatio)
                    self.pixmap_item = self.scene.addPixmap(self.thumbnail_pixmap)
                # 06/30
                if len(self.point_list) == 7:
                    height, width = self.cur_img.shape[:2]
                    pixel_pts = np.array([
                        self.point_list[4],  # 像素點1
                        self.point_list[5],  # 像素點2
                        self.point_list[6]  # 像素點3
                    ])
                    geo_pts = np.array([
                        [121.03182871853623, 24.79775311784976],  # 對應經緯度1
                        [121.03187050670385, 24.79782809976728],  # 對應經緯度2
                        [121.03538520634177, 24.796119707193924]  # 對應經緯度3
                    ])
                    A = np.vstack([pixel_pts.T, np.ones(3)]).T  # 每行 [x, y, 1]
                    lons = geo_pts[:, 0]
                    lats = geo_pts[:, 1]
                    a, b, c = solve(A, lons)  # 解 a, b, c
                    d, e, f = solve(A, lats)  # 解 d, e, f
                    geotransform = (c, a, b, f, d, e)
                    print(geotransform)
                    # driver = gdal.GetDriverByName('GTiff')
                    # geotiff = driver.Create('output_georeferenced.tif', width, height, png.RasterCount, gdal.GDT_Byte)
                    # geotiff.SetGeoTransform(geotransform)
                    # srs = osr.SpatialReference()
                    # srs.ImportFromEPSG(4326)
                    # geotiff.SetProjection(srs.ExportToWkt())
                    # for i in range(1, png.RasterCount + 1):
                    #     geotiff.GetRasterBand(i).WriteArray(png.GetRasterBand(i).ReadAsArray())
                    # png = None
                    # geotiff = None
                    self.point_list.clear()
    def update_selection(self, event):
        pass
    def finish_selection(self, event):
        self.start_pos = None

    """    def open_dir_old(self):
        # load config
        self.label.setVisible(True)
        conf_file = open(self.lineEdit_config.text(), 'r')
        self.conf = yaml.safe_load(conf_file)

        start_time = time.time()
        print(f"CPU 核心數: {os.cpu_count()}")
        self.step_flag = 1
        self.renew()
        self.img_fd = QFileDialog.getExistingDirectory(self, "Select Directory", "./")
        if not self.img_fd:
            print("未選擇任何資料夾")
            return
        img_list = sorted(listdir(self.img_fd))
        print(f'img_list ={img_list}')
        self.flag_first_valid = True
        self.label.setScaledContents(True)
        for i, img_name in enumerate((img_list)):
            # save_fd = join('result/round', str(i))
            # pathlib.Path(save_fd).mkdir(parents=True, exist_ok=True)
            img_fullpath = join(self.img_fd, img_name)
            # 1. read GPS
            gps_exif = read_exif(img_fullpath)
            # 是否過濾巡檢範圍
            # judgement = func_range(gps_exif, self.conf['valid_scope'][0], self.conf['valid_scope'][1],self.conf['valid_scope'][2], self.conf['valid_scope'][3])
            judgement = True
            self.qt_table_color(img_fullpath, gps_exif, judgement)
            self.img_fullpath_all_list.append(img_fullpath)
            self.GPS_all_list.append(gps_exif)
            self.judgement_list.append(judgement)
            if judgement == True:
                self.GPS_valid_list.append(gps_exif)
                self.img_fullpath_valid_list.append(img_fullpath)
                if self.flag_first_valid == True:
                    self.flag_first_valid  = False
                    # 12/27  計數crop_count、第一個first_true、有效範圍show_limit
                    self.crop_count = i
                    self.first_true = i
                    image = cv2.imread(img_fullpath)
                    self.original_img_h, self.original_img_w = image.shape[:2]
                    self.original_center = [self.original_img_w * 0.5, self.original_img_h * 0.5]
                    valid_area = self.conf['valid_area']['rect']
                    (v_x1, v_y1) = (valid_area['x1'], valid_area['y1'])
                    (v_x2, v_y2) = (valid_area['x2'], valid_area['y2'])
                    # if (v_x2 - v_x1) == self.original_img_w and (v_y2 - v_y1) == self.original_img_h:self.flag_same_size = True
                    self.flag_same_size = True
                    img_draw = image.copy()
                    cv2.rectangle(img_draw,
                                  (self.conf['template_area']['temp_up']['x1'],
                                   self.conf['template_area']['temp_up']['y1']),
                                  (self.conf['template_area']['temp_up']['x2'],
                                   self.conf['template_area']['temp_up']['y2']), (0, 255, 255), 3)
                    cv2.rectangle(img_draw,
                                  (self.conf['template_area']['temp_down']['x1'],
                                   self.conf['template_area']['temp_down']['y1']),
                                  (self.conf['template_area']['temp_down']['x2'],
                                   self.conf['template_area']['temp_down']['y2']), (0, 255, 255), 3)
                    cv2.rectangle(img_draw,
                                  (self.conf['template_area']['temp_left']['x1'],
                                   self.conf['template_area']['temp_left']['y1']),
                                  (self.conf['template_area']['temp_left']['x2'],
                                   self.conf['template_area']['temp_left']['y2']), (0, 255, 255), 3)
                    cv2.rectangle(img_draw,
                                  (self.conf['template_area']['temp_right']['x1'],
                                   self.conf['template_area']['temp_right']['y1']),
                                  (self.conf['template_area']['temp_right']['x2'],
                                   self.conf['template_area']['temp_right']['y2']), (0, 255, 255), 3)
                    draw_img = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
                    Ny, Nx, channels = draw_img.shape
                    self.image_view = QtGui.QImage(draw_img.data, Nx, Ny, QtGui.QImage.Format_RGB888)
                    self.label.setPixmap(QtGui.QPixmap.fromImage(self.image_view))

        # 04/01 方向判斷  # 06/25 轉向提醒
        self.direction_list = calculate_quadrants(self.GPS_valid_list)
        self.textEdit.append(str(self.direction_list))
        change_index = [i for i in range(1, len(self.direction_list)) if self.direction_list[i] != self.direction_list[i - 1]]
        for i in change_index:
            for col in range(self.tableWidget.columnCount()):
                item = self.tableWidget.item(i, col)
                if item is not None:
                    item.setBackground(QtGui.QColor('#FFB6C1'))
                    item.setForeground(QtGui.QBrush(QtGui.QColor('red')))

        with concurrent.futures.ThreadPoolExecutor (max_workers=6) as executor:
            print(f"max_workers: {executor._max_workers}")
            # self.valid_images = list(executor.map(cv2.imread, self.img_fullpath_valid_list))
            self.valid_images = list(executor.map(read_with_roi, self.img_fullpath_valid_list))

        # 調整平均亮度
        self.valid_images = adjust_images_to_average_brightness(self.valid_images)

        # self.btn_crop.setEnabled(True)
        # self.btn_stitch.setEnabled(False)
        # self.btn_open.setEnabled(False)
        self.tableWidget.setEnabled(True)
        print(f'self.GPS_all_list = {self.GPS_all_list}')
        print(f'self.judgement_list = {self.judgement_list}')
        print(f'self.GPS_valid_list = {self.GPS_valid_list}')
        print(f'self.img_fullpath_valid_list = {self.img_fullpath_valid_list}')
        self.show_limit = self.crop_count + len(self.img_fullpath_valid_list)
        print(f'show_limit = {self.show_limit}')
        end_time = time.time()
        print(end_time -start_time)
    def show_temp(self):
        temp_up_w = self.conf['template_area']['temp_up']['x2'] - self.conf['template_area']['temp_up']['x1']
        temp_up_h = self.conf['template_area']['temp_up']['y2'] - self.conf['template_area']['temp_up']['y1']
        temp_down_w = self.conf['template_area']['temp_down']['x2'] - self.conf['template_area']['temp_down']['x1']
        temp_down_h = self.conf['template_area']['temp_down']['y2'] - self.conf['template_area']['temp_down']['y1']
        temp_left_w = self.conf['template_area']['temp_left']['x2'] - self.conf['template_area']['temp_left']['x1']
        temp_left_h = self.conf['template_area']['temp_left']['y2'] - self.conf['template_area']['temp_left']['y1']
        temp_right_w = self.conf['template_area']['temp_right']['x2'] - self.conf['template_area']['temp_right']['x1']
        temp_right_h = self.conf['template_area']['temp_right']['y2'] - self.conf['template_area']['temp_right']['y1']
        if self.step_flag == 2:
            if self.crop_count + 1 < self.show_limit:
                print(f'self.crop_count = {self.crop_count}')
                self.tableWidget.setCurrentCell(self.crop_count, 0)
                item = self.tableWidget.item(self.crop_count + 1, 0)
                item.setSelected(True)
                # 左圖
                if self.flag_same_size == True:
                    # crop_image = cv2.imread(self.img_fullpath_valid_list[self.crop_count - self.first_true])
                    crop_image = self.valid_images[self.crop_count - self.first_true].copy()
                else:
                    crop_image = cv2.imread(f'./result/crop_images/{str(self.crop_count - self.first_true)}.jpg')
                if self.direction_list[self.crop_count+1] == int(self.lineEdit_up.text()) :
                    draw_img = cv2.rectangle(crop_image, (self.template_list[self.crop_count - self.first_true][0],
                                                          self.template_list[self.crop_count - self.first_true][1]),
                                             (self.template_list[self.crop_count - self.first_true][0] + temp_up_w,
                                              self.template_list[self.crop_count - self.first_true][1] + temp_up_h),
                                             (12, 31, 242), 10)
                if self.direction_list[self.crop_count+1] == int(self.lineEdit_down.text()):
                    draw_img = cv2.rectangle(crop_image, (self.template_list[self.crop_count - self.first_true][0],
                                                      self.template_list[self.crop_count - self.first_true][1]),
                                             (self.template_list[self.crop_count - self.first_true][0] + temp_down_w,
                                              self.template_list[self.crop_count - self.first_true][1] + temp_down_h),
                                             (12, 31, 242), 10)
                if self.direction_list[self.crop_count+1] == int(self.lineEdit_left.text()):
                    draw_img = cv2.rectangle(crop_image, (self.template_list[self.crop_count - self.first_true][0],
                                                      self.template_list[self.crop_count - self.first_true][1]),
                                             (self.template_list[self.crop_count - self.first_true][0] + temp_left_w,
                                              self.template_list[self.crop_count - self.first_true][1] + temp_left_h),
                                             (12, 31, 242), 10)
                if self.direction_list[self.crop_count+1] == int(self.lineEdit_right.text()):
                    draw_img = cv2.rectangle(crop_image, (self.template_list[self.crop_count - self.first_true][0],
                                                          self.template_list[self.crop_count - self.first_true][1]),
                                             (self.template_list[self.crop_count - self.first_true][0] + temp_right_w,
                                              self.template_list[self.crop_count - self.first_true][1] + temp_right_h),
                                             (12, 31, 242), 10)
                draw_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
                Ny, Nx, channels = draw_img.shape
                self.image_view_2 = QtGui.QImage(draw_img.data, Nx, Ny, QtGui.QImage.Format_RGB888)
                self.label_6.setPixmap(QtGui.QPixmap.fromImage(self.image_view_2))
                self.crop_count = self.crop_count + 1
                # 右圖
                if self.flag_same_size == True:
                    # crop_image = cv2.imread(self.img_fullpath_valid_list[self.crop_count - self.first_true])
                    crop_image = self.valid_images[self.crop_count - self.first_true].copy()
                else:
                    crop_image = cv2.imread(f'./result/crop_images/{str(self.crop_count - self.first_true)}.jpg')
                if self.direction_list[self.crop_count] == int(self.lineEdit_up.text()):
                    cv2.rectangle(crop_image,
                                  (self.conf['template_area']['temp_up']['x1'],
                                   self.conf['template_area']['temp_up']['y1']),
                                  (self.conf['template_area']['temp_up']['x2'],
                                   self.conf['template_area']['temp_up']['y2']), (0, 255, 255), 10)
                if self.direction_list[self.crop_count] == int(self.lineEdit_down.text()):
                    cv2.rectangle(crop_image,
                                  (self.conf['template_area']['temp_down']['x1'],
                                   self.conf['template_area']['temp_down']['y1']),
                                  (self.conf['template_area']['temp_down']['x2'],
                                   self.conf['template_area']['temp_down']['y2']), (0, 255, 255), 10)
                if self.direction_list[self.crop_count] == int(self.lineEdit_left.text()):
                    cv2.rectangle(crop_image,
                                  (self.conf['template_area']['temp_left']['x1'],
                                   self.conf['template_area']['temp_left']['y1']),
                                  (self.conf['template_area']['temp_left']['x2'],
                                   self.conf['template_area']['temp_left']['y2']), (0, 255, 255), 10)
                if self.direction_list[self.crop_count] == int(self.lineEdit_right.text()):
                    cv2.rectangle(crop_image,
                                  (self.conf['template_area']['temp_right']['x1'],
                                   self.conf['template_area']['temp_right']['y1']),
                                  (self.conf['template_area']['temp_right']['x2'],
                                   self.conf['template_area']['temp_right']['y2']), (0, 255, 255), 10)
                draw_img = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
                Ny, Nx, channels = draw_img.shape
                self.image_view_3 = QtGui.QImage(draw_img.data, Nx, Ny, QtGui.QImage.Format_RGB888)
                self.label_7.setPixmap(QtGui.QPixmap.fromImage(self.image_view_3))
        if self.step_flag == 3:  # 畫框
            if self.crop_count + 1 < self.show_limit:
                self.tableWidget.setCurrentCell(self.crop_count, 0)
                item = self.tableWidget.item(self.crop_count + 1, 0)
                item.setSelected(True)
                # 左圖
                if self.flag_same_size == True:
                    # crop_image = cv2.imread(self.img_fullpath_valid_list[self.crop_count - self.first_true])
                    crop_image = self.valid_images[self.crop_count - self.first_true].copy()
                else:
                    crop_image = cv2.imread(f'./result/crop_images/{str(self.crop_count - self.first_true)}.jpg')
                draw_img = cv2.rectangle(crop_image, (
                self.template_list[self.crop_count- self.first_true][0], self.template_list[self.crop_count- self.first_true][1]), (
                                             self.template_list[self.crop_count- self.first_true][0] + temp_up_w,
                                             self.template_list[self.crop_count- self.first_true][1] + temp_up_h), (12, 31, 242), 10)
                draw_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
                Ny, Nx, channels = draw_img.shape
                self.image_view_2 = QtGui.QImage(draw_img.data, Nx, Ny, QtGui.QImage.Format_RGB888)
                self.label_6.setPixmap(QtGui.QPixmap.fromImage(self.image_view_2))
                self.crop_count = self.crop_count + 1
                # 右圖
                if self.flag_same_size == True:
                    # crop_image = cv2.imread(self.img_fullpath_valid_list[self.crop_count - self.first_true])
                    crop_image = self.valid_images[self.crop_count - self.first_true].copy()
                else:
                    crop_image = cv2.imread(f'./result/crop_images/{str(self.crop_count - self.first_true)}.jpg')
                if self.direction_list[self.crop_count] == int(self.lineEdit_up.text()):
                    cv2.rectangle(crop_image,
                                  (self.conf['template_area']['temp_up']['x1'],
                                   self.conf['template_area']['temp_up']['y1']),
                                  (self.conf['template_area']['temp_up']['x2'],
                                   self.conf['template_area']['temp_up']['y2']), (0, 255, 255), 10)
                if self.direction_list[self.crop_count] == int(self.lineEdit_down.text()):
                    cv2.rectangle(crop_image,
                                  (self.conf['template_area']['temp_down']['x1'],
                                   self.conf['template_area']['temp_down']['y1']),
                                  (self.conf['template_area']['temp_down']['x2'],
                                   self.conf['template_area']['temp_down']['y2']), (0, 255, 255), 10)
                if self.direction_list[self.crop_count] == int(self.lineEdit_left.text()):
                    cv2.rectangle(crop_image,
                                  (self.conf['template_area']['temp_left']['x1'],
                                   self.conf['template_area']['temp_left']['y1']),
                                  (self.conf['template_area']['temp_left']['x2'],
                                   self.conf['template_area']['temp_left']['y2']), (0, 255, 255), 10)
                if self.direction_list[self.crop_count] == int(self.lineEdit_right.text()):
                    cv2.rectangle(crop_image,
                                  (self.conf['template_area']['temp_right']['x1'],
                                   self.conf['template_area']['temp_right']['y1']),
                                  (self.conf['template_area']['temp_right']['x2'],
                                   self.conf['template_area']['temp_right']['y2']), (0, 255, 255), 10)
                # 12/24 選取範圍
                cv2.rectangle(crop_image, (self.rect_start[0],self.rect_start[1]),(self.rect_end[0],self.rect_end[1]), (22,103,242), 10)
                draw_img = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
                Ny, Nx, channels = draw_img.shape
                self.image_view_3 = QtGui.QImage(draw_img.data, Nx, Ny, QtGui.QImage.Format_RGB888)
                self.label_7.setPixmap(QtGui.QPixmap.fromImage(self.image_view_3))
    def start_stitching(self):
        print(f'直接給答案的:{self.revise_match_list}')
        img_h, img_w = self.valid_images[0].shape[:2]

        # 先用方向分類
        segments = []
        current_segment = [0]
        for i in range(1, len(self.direction_list)):
            if self.direction_list[i] == self.direction_list[i - 1]:
                current_segment.append(i)
            else:
                segments.append(current_segment)
                current_segment = [i]
        segments.append(current_segment)
        print(f'原始 segments: {segments}')

        # 要再把轉向的加進後面的list
        revised_segments = []
        stitch_offset_list = [[0,0]]
        i = 0
        while i < len(segments):
            if len(segments[i]) == 1 and i + 1 < len(segments):
                # 單點 + 下一段合併
                stitch_offset_list.append(self.stitch_offset_list[segments[i][0]])
                merged = segments[i] + segments[i + 1]
                revised_segments.append(merged)
                i += 2  # skip next
            else:
                revised_segments.append(segments[i])
                i += 1
        print(f'合併轉向後 segments: {revised_segments}')

        stitch_list = []
        count = 0
        for seg in revised_segments:
            print(count)
            sub_offsets = [self.stitch_offset_list[i] for i in seg]
            sub_images = [self.valid_images[i] for i in seg]
            if count > 0:
                sub_offsets[0] = [0,0]
            final_sti_img,final_coordinate_list,center_coordinate_list = stitch_seg_images(sub_offsets,sub_images)

            # 柔化融合  (轉彎出問題、正負正也會出問題)
            print(f'final_coordinate_list = {final_coordinate_list}')
            # final_sti_img = soften_seams(final_sti_img, final_coordinate_list, blend_height=100)

            # 透視轉換
            # 分割塊數，避免過長error
            height, width = final_sti_img.shape[:2]
            split_height = 30000
            num_splits = (height + split_height - 1) // split_height

            all_corners = []
            valid_segments = []

            # Step 1: 找出每塊的四個角點
            for i in range(num_splits):
                start_y = i * split_height
                end_y = min(start_y + split_height, height)
                segment = final_sti_img[start_y:end_y, :]

                corners = find_corners(segment)
                if None in corners:
                    print(f"Segment {i}: All white image")
                    continue

                all_corners.append(corners)
                valid_segments.append(segment)
            # Step 2: 找最小寬度
            widths = [np.linalg.norm(np.array(tr) - np.array(tl)) for tl, tr, _, _ in all_corners]
            target_width = int(min(widths))
            print(f"Unified width = {target_width}")

            # Step 3: warp 所有 segment，並儲存
            warped_list = []
            for i, (segment, corners) in enumerate(zip(valid_segments, all_corners)):
                warped = warp_segment(segment, corners, target_width)
                warped_list.append(warped)
                # cv2.imwrite(f'./test_img_save/segment_{i}_warped_fixed.jpg', warped)
                print(f"Segment {i} warped and saved.")

            # Step 4: 拼接 (透視轉換後的上下塊拼接)
            stitched = cv2.vconcat(warped_list)

            stitch_list.append(stitched)
            cv2.imwrite(f"./test_img_save/seg_{count}.jpg", stitched)
            count = count+1

        # 要再依照左右的offset，把兩條併起來
        canvas_width = 0
        canvas_height = 0

        min_x = min(x for x, y in stitch_offset_list)
        min_y = min(y for x, y in stitch_offset_list)

        shift_x = -min_x if min_x < 0 else 0
        shift_y = -min_y if min_y < 0 else 0

        for img, (x, y) in zip(stitch_list, stitch_offset_list):
            h, w = img.shape[:2]
            canvas_width = max(canvas_width, x + shift_x + w)
            canvas_height = max(canvas_height, y + shift_y + h)
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
        for i, (img, (x, y)) in enumerate(zip(stitch_list, stitch_offset_list)):
            h, w = img.shape[:2]
            new_x = x + shift_x
            new_y = y + shift_y
            if i == 0:
                # 第1張圖直接貼
                canvas[new_y:new_y + h, new_x:new_x + w] = img
            else:
                # 建立遮罩 消除白邊
                white_edge = 300
                mask = np.zeros((h, w), dtype=bool)
                left_strip = img[:, :white_edge]
                near_white = (left_strip >= 240).all(axis=2)
                mask[:, :white_edge] = ~near_white
                mask[:, white_edge:] = True
                # 貼上
                canvas[new_y:new_y + h, new_x:new_x + w][mask] = img[mask]
        # 儲存或顯示結果
        # cv2.imwrite('./test_img_save/final_merged_stitched.jpg', canvas)
        print("多張圖片合併完成，儲存為 final_merged_stitched.jpg")

        # # 在大圖上加上每張圖的左上角編號(也可標GPS、座標等資訊)
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # font_scale = 5
        # font_color = (0, 0, 255)  # Red
        # thickness = 3
        # for i, (x, y) in enumerate(final_coordinate_list):
        #     cv2.putText(final_sti_img, f'{i}', (x + 10, y + 30), font, font_scale, font_color, thickness, cv2.LINE_AA)
        # stitch

        # 建立 QImage
        rgb_image = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        bytes_per_line = channel * width
        q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)

        self.cur_img = canvas.copy()

        self.label.setVisible(False)
        self.label_6.setVisible(False)
        self.label_7.setVisible(False)

        # 轉為 QPixmap
        self.original_pixmap = QPixmap.fromImage(q_image)
        self.graphicsView.setVisible(True)
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)
        self.scale_factor = 0.25
        self.thumbnail_pixmap = self.original_pixmap.scaled(int(self.original_pixmap.width() * self.scale_factor),int(self.original_pixmap.height() * self.scale_factor),Qt.KeepAspectRatio)
        self.pixmap_item = self.scene.addPixmap(self.thumbnail_pixmap)

        self.step_flag = 4
    def update_progress(self, value):
        # 更新進度條
        self.progressBar.setValue(value)
    def end_stitching(self, GPS_sti_offset_list,crop_point_list):
        self.GPS_sti_offset_list = GPS_sti_offset_list
        self.crop_point_list = crop_point_list
        # 12/05 拼接完顯示長直圖，可與Label互動，縮放、拖曳、點擊定位(旋轉、裁切)
        self.label.setScaledContents(False)
        self.label.setVisible(False)
        self.label_6.setVisible(False)
        self.label_7.setVisible(False)
        self.tableWidget.setVisible(False)
        self.progressBar.setVisible(False)
        # self.btn_crop.setEnabled(False)
        # self.btn_open.setEnabled(False)
        # self.btn_stitch.setEnabled(False)
        # stitch_res = cv2.imread(f'./result/stitch/stitch_{len(self.img_fullpath_valid_list)-1}.png', cv2.IMREAD_UNCHANGED)
        stitch_res = cv2.imread(f'./result/stitch/stitch_{len(self.img_fullpath_valid_list) - 1}.png', cv2.IMREAD_COLOR)
        self.cur_img = stitch_res
        self.save_img = self.cur_img.copy()
        self.img_h, self.img_w, channels = self.cur_img.shape
        self.cur_img = cv2.cvtColor(self.cur_img, cv2.COLOR_BGRA2RGBA)
        self.image_view = QtGui.QImage(self.cur_img.data, self.img_w, self.img_h, QtGui.QImage.Format_RGBA8888)
        self.label.setPixmap(QtGui.QPixmap.fromImage(self.image_view))
        self.label.setCursor(QtCore.Qt.PointingHandCursor)
        self.mouse_flag = True
    def start_crop(self):
        self.graphicsView.setVisible(False)
        self.step_flag = 2
        self.direction_label = [int(self.lineEdit_up.text()),int(self.lineEdit_down.text()),int(self.lineEdit_left.text()),int(self.lineEdit_right.text())]
        self.crop_thread = CropThread(self.img_fullpath_valid_list,self.conf,self.valid_images,self.direction_list,self.direction_label,self.revise_match_list)
        self.crop_thread.progress_signal.connect(self.update_progress)
        self.crop_thread.list_signal.connect(self.end_crop)
        self.crop_thread.start()
    def end_crop(self,stitch_offset_list,template_list):
        # 4/29  stitch_offset_list  給stitching
        self.stitch_offset_list = stitch_offset_list
        self.template_list = template_list
        self.label.clear()
        self.label_6.setVisible(True)
        self.label_7.setVisible(True)
        self.graphicsView.setVisible(False)
        self.btn_stitch.setEnabled(True)
        # self.btn_open.setEnabled(True)
        self.show_temp()
        meters_per_pixel_list = []
        for i in range (1,len(self.stitch_offset_list)):
            meters_per_pixel = calculate_meters_per_pixel(self.GPS_valid_list[i],self.GPS_valid_list[i-1],self.stitch_offset_list[i])
            meters_per_pixel_list.append(meters_per_pixel)
        meters_per_pixel = np.mean(meters_per_pixel_list) # 單位公尺
        print(meters_per_pixel)

        self.image_crop_width = self.conf['valid_area']['rect']['x2'] - self.conf['valid_area']['rect']['x1']
        self.image_crop_height = self.conf['valid_area']['rect']['y2'] - self.conf['valid_area']['rect']['y1']
"""

    def yaml_write(self):
        print(f'self.step_flag == {self.step_flag}')
        config_data = {
            'valid_area': {
                'rect': {
                    'x1': self.conf['valid_area']['rect']['x1'],
                    'y1': self.conf['valid_area']['rect']['y1'],
                    'x2': self.conf['valid_area']['rect']['x2'],
                    'y2': self.conf['valid_area']['rect']['y2']
                },
                'polygon': None  # 設置為 None 表示空值
            },
            'template_area': {
                'temp_up':{
                        'x1': self.conf['template_area']['temp_up']['x1'],
                        'y1': self.conf['template_area']['temp_up']['y1'],
                        'x2': self.conf['template_area']['temp_up']['x2'],
                        'y2': self.conf['template_area']['temp_up']['y2']
                    },
                    'temp_down': {
                        'x1': self.conf['template_area']['temp_down']['x1'],
                        'y1': self.conf['template_area']['temp_down']['y1'],
                        'x2': self.conf['template_area']['temp_down']['x2'],
                        'y2': self.conf['template_area']['temp_down']['y2']
                    },
                    'temp_left': {
                        'x1': self.conf['template_area']['temp_left']['x1'],
                        'y1': self.conf['template_area']['temp_left']['y1'],
                        'x2': self.conf['template_area']['temp_left']['x2'],
                        'y2': self.conf['template_area']['temp_left']['y2']
                    },
                    'temp_right': {
                        'x1': self.conf['template_area']['temp_right']['x1'],
                        'y1': self.conf['template_area']['temp_right']['y1'],
                        'x2': self.conf['template_area']['temp_right']['x2'],
                        'y2': self.conf['template_area']['temp_right']['y2']
                    }
            },
            'valid_scope': [[24.77730084,121.04272265], [24.77721331,121.04282717],[24.77565692,121.04134659], [24.775753739456956,121.041227]],
            'direction': {
                "up": int(self.lineEdit_up.text()),
                "down": int(self.lineEdit_down.text()),
                "left": int(self.lineEdit_left.text()),
                "right": int(self.lineEdit_right.text())
            }
        }
        with open(self.lineEdit_config.text(), 'w', encoding='utf-8') as file:
            yaml.dump(config_data, file, allow_unicode=True, sort_keys=False)
        conf_file = open(self.lineEdit_config.text(), 'r')
        self.conf = yaml.safe_load(conf_file)
        # 讀完圖
        if self.step_flag == 1:
            image = cv2.imread(self.img_fullpath_all_list[0])
            img_draw = image.copy()
            cv2.rectangle(img_draw, (self.conf['template_area']['temp_up']['x1'], self.conf['template_area']['temp_up']['y1']),
                          (self.conf['template_area']['temp_up']['x2'], self.conf['template_area']['temp_up']['y2']), (0, 255, 255),3)
            cv2.rectangle(img_draw,
                          (self.conf['template_area']['temp_down']['x1'], self.conf['template_area']['temp_down']['y1']),
                          (self.conf['template_area']['temp_down']['x2'], self.conf['template_area']['temp_down']['y2']),(0, 255, 255), 3)
            cv2.rectangle(img_draw,
                          (self.conf['template_area']['temp_left']['x1'], self.conf['template_area']['temp_left']['y1']),
                          (self.conf['template_area']['temp_left']['x2'], self.conf['template_area']['temp_left']['y2']),(0, 255, 255), 3)
            cv2.rectangle(img_draw,
                          (self.conf['template_area']['temp_right']['x1'], self.conf['template_area']['temp_right']['y1']),
                          (self.conf['template_area']['temp_right']['x2'], self.conf['template_area']['temp_right']['y2']),(0, 255, 255), 3)
            draw_img = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
            Ny, Nx, channels = draw_img.shape
            self.image_view = QtGui.QImage(draw_img.data, Nx, Ny, QtGui.QImage.Format_RGB888)
            self.label.setPixmap(QtGui.QPixmap.fromImage(self.image_view))
        if self.step_flag == 300:
            self.step_flag = 200
            curr = self.watchdog_info_list[self.show_count]
            if curr.direction == 'up':
                config_data['template_area']['temp_up']['x1'] = int(self.rect_start[0])
                config_data['template_area']['temp_up']['y1'] = int(self.rect_start[1])
                config_data['template_area']['temp_up']['x2'] = int(self.rect_end[0])
                config_data['template_area']['temp_up']['y2'] = int(self.rect_end[1])
            if curr.direction == 'down':
                config_data['template_area']['temp_down']['x1'] = int(self.rect_start[0])
                config_data['template_area']['temp_down']['y1'] = int(self.rect_start[1])
                config_data['template_area']['temp_down']['x2'] = int(self.rect_end[0])
                config_data['template_area']['temp_down']['y2'] = int(self.rect_end[1])
            if curr.direction == 'left':
                config_data['template_area']['temp_left']['x1'] = int(self.rect_start[0])
                config_data['template_area']['temp_left']['y1'] = int(self.rect_start[1])
                config_data['template_area']['temp_left']['x2'] = int(self.rect_end[0])
                config_data['template_area']['temp_left']['y2'] = int(self.rect_end[1])
            if curr.direction == 'right':
                config_data['template_area']['temp_right']['x1'] = int(self.rect_start[0])
                config_data['template_area']['temp_right']['y1'] = int(self.rect_start[1])
                config_data['template_area']['temp_right']['x2'] = int(self.rect_end[0])
                config_data['template_area']['temp_right']['y2'] = int(self.rect_end[1])
            with open(self.lineEdit_config.text(), 'w', encoding='utf-8') as file:
                yaml.dump(config_data, file, allow_unicode=True, sort_keys=False)
            conf_file = open(self.lineEdit_config.text(), 'r')
            self.conf = yaml.safe_load(conf_file)
            self.show_count = self.show_count - 1
            self.show_template()
        if self.step_flag == 2:
            self.crop_count = self.crop_count - 1
            self.show_temp()
        # 12/24 將劃出的框位置，修改template位置
        if self.step_flag == 3:
            self.step_flag = 2
            if self.direction_list[self.crop_count] == int(self.lineEdit_up.text()):
                config_data['template_area']['temp_up']['x1'] = int(self.rect_start[0])
                config_data['template_area']['temp_up']['y1'] = int(self.rect_start[1])
                config_data['template_area']['temp_up']['x2'] = int(self.rect_end[0])
                config_data['template_area']['temp_up']['y2'] = int(self.rect_end[1])
            if self.direction_list[self.crop_count] == int(self.lineEdit_down.text()):
                config_data['template_area']['temp_down']['x1'] = int(self.rect_start[0])
                config_data['template_area']['temp_down']['y1'] = int(self.rect_start[1])
                config_data['template_area']['temp_down']['x2'] = int(self.rect_end[0])
                config_data['template_area']['temp_down']['y2'] = int(self.rect_end[1])
            if self.direction_list[self.crop_count] == int(self.lineEdit_left.text()):
                config_data['template_area']['temp_left']['x1'] = int(self.rect_start[0])
                config_data['template_area']['temp_left']['y1'] = int(self.rect_start[1])
                config_data['template_area']['temp_left']['x2'] = int(self.rect_end[0])
                config_data['template_area']['temp_left']['y2'] = int(self.rect_end[1])
            if self.direction_list[self.crop_count] == int(self.lineEdit_right.text()):
                config_data['template_area']['temp_right']['x1'] = int(self.rect_start[0])
                config_data['template_area']['temp_right']['y1'] = int(self.rect_start[1])
                config_data['template_area']['temp_right']['x2'] = int(self.rect_end[0])
                config_data['template_area']['temp_right']['y2'] = int(self.rect_end[1])
            with open(self.lineEdit_config.text(), 'w', encoding='utf-8') as file:
                yaml.dump(config_data, file, allow_unicode=True, sort_keys=False)
            conf_file = open(self.lineEdit_config.text(), 'r')
            self.conf = yaml.safe_load(conf_file)
            self.crop_count = self.crop_count - 1
            self.show_temp()
            # 20250117  在第一步驟畫框，修改template位置
        if self.step_flag == 11:
            self.step_flag = 1
            config_data = {
                'valid_area': {
                    'rect': {
                        'x1': int(self.lineEdit_crop_x1.text()),
                        'y1': int(self.lineEdit_crop_y1.text()),
                        'x2': int(self.lineEdit_crop_x2.text()),
                        'y2': int(self.lineEdit_crop_y2.text())
                    },
                    'polygon': None  # 設置為 None 表示空值
                },
                'template_area': {
                    'temp_up':{
                        'x1': int(self.rect_start[0]),
                        'y1': int(self.rect_start[1]),
                        'x2': int(self.rect_end[0]),
                        'y2': int(self.rect_end[1])
                    },
                    'temp_down': {
                        'x1': int(1992),
                        'y1': int(396),
                        'x2': int(2772),
                        'y2': int(784)
                    },
                    'temp_left': {
                        'x1': int(2324),
                        'y1': int(2172),
                        'x2': int(2576),
                        'y2': int(2600)
                    },
                    'temp_right': {
                        'x1': int(548),
                        'y1': int(1220),
                        'x2': int(1032),
                        'y2': int(1988)
                    }
                },
                'valid_scope': [[24.77730084, 121.04272265], [24.77721331, 121.04282717],
                                [24.77565692, 121.04134659], [24.775753739456956, 121.041227]],
                'direction': {
                    "up": int(self.lineEdit_up.text()),
                    "down": int(self.lineEdit_down.text()),
                    "left": int(self.lineEdit_left.text()),
                    "right": int(self.lineEdit_right.text())

                }
            }
            with open(self.lineEdit_config.text(), 'w', encoding='utf-8') as file:
                yaml.dump(config_data, file, allow_unicode=True, sort_keys=False)
            conf_file = open(self.lineEdit_config.text(), 'r')
            self.conf = yaml.safe_load(conf_file)
            # 20250117_重新顯示影像
            image = cv2.imread(self.img_fullpath_valid_list[0])
            img_draw = image.copy()
            cv2.rectangle(img_draw,
                          (self.conf['template_area']['temp_up']['x1'], self.conf['template_area']['temp_up']['y1']),
                          (self.conf['template_area']['temp_up']['x2'], self.conf['template_area']['temp_up']['y2']),(0, 255, 255), 3)
            cv2.rectangle(img_draw,
                          (self.conf['template_area']['temp_down']['x1'], self.conf['template_area']['temp_down']['y1']),
                          (self.conf['template_area']['temp_down']['x2'], self.conf['template_area']['temp_down']['y2']),(0, 255, 255), 3)
            cv2.rectangle(img_draw,
                          (self.conf['template_area']['temp_left']['x1'], self.conf['template_area']['temp_left']['y1']),
                          (self.conf['template_area']['temp_left']['x2'], self.conf['template_area']['temp_left']['y2']),(0, 255, 255), 3)
            cv2.rectangle(img_draw,
                          (self.conf['template_area']['temp_right']['x1'],self.conf['template_area']['temp_right']['y1']),
                          (self.conf['template_area']['temp_right']['x2'],self.conf['template_area']['temp_right']['y2']), (0, 255, 255), 3)
            draw_img = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
            Ny, Nx, channels = draw_img.shape
            self.image_view = QtGui.QImage(draw_img.data, Nx, Ny, QtGui.QImage.Format_RGB888)
            self.label.setPixmap(QtGui.QPixmap.fromImage(self.image_view))
        self.set_lineedit()
        self.show_message()
    def yaml_load(self):
        config_path, _ = QFileDialog.getOpenFileName(self,"Select YAML File","./","YAML Files (*.yaml *.yml)")
        if config_path:
            self.lineEdit_config.setText(str(config_path))
            conf_file = open(self.lineEdit_config.text(), 'r')
            self.conf = yaml.safe_load(conf_file)
            self.set_lineedit()
    def show_message(self):
        # 創建彈出視窗
        msg_box = QtWidgets.QMessageBox()
        msg_box.setIcon(QtWidgets.QMessageBox.Information)
        msg_box.setWindowTitle("Prompt")
        msg_box.setText("Success")
        msg_box.setStandardButtons(QtWidgets.QMessageBox.Ok)
        # 顯示視窗
        msg_box.exec_()
    def renew(self):
        conf_file = open(self.lineEdit_config.text(), 'r')
        self.conf = yaml.safe_load(conf_file)
        # self.log.info(str(self.conf))

        self.tableWidget.setRowCount(0)

        # 宣告
        self.img_fullpath_all_list = []
        self.GPS_all_list = []

        # 20250117
        self.gps_exif = []
        self.img_fullpath = []

        self.img_fullpath_valid_list = []
        self.GPS_valid_list = []

        self.judgement_list = []
        self.GPS_sti_offset_list = [[0, 0]]

        self.coord_gps_temp = []
        self.coord_gps_temp.append({'coord': [None, None], 'gps': [None, None]})
        self.GPS_crop_center_list = []
        self.XY_crop_center_list = []

        self.label_6.setVisible(False)
        self.label_7.setVisible(False)

        # 12/05 Label互動功能參數
        self.mouse_mv_y = ""
        self.mouse_mv_x = ""
        self.label_x = 0
        self.label_y = 0
        self.x1 = 0
        self.y1 = 0
        self.resize_point = 10
        self.has_been_chgd = False
        self.has_been_moved = False
        self.point_list = []
        self.mouse_flag = False
        self.rotate_flag = True

        # 12/19
        self.template_list = []
        # self.label.setGeometry(QtCore.QRect(10, 110, 1152, 864))
        self.label.setScaledContents(True)
        self.label.clear()
        self.tableWidget.setVisible(True)
        self.progressBar.setVisible(True)
        self.flag_same_size = False
        self.valid_images = []
        self.revise_match_list = []
        self.graphicsView.setVisible(False)
    def qt_table_color(self,img,coord,judgement):
        self.tableWidget.setColumnCount(1)
        self.tableWidget.setHorizontalHeaderLabels(['檔名'])
        current_row = self.tableWidget.rowCount()
        self.tableWidget.insertRow(current_row)
        # 添加新項目到第一列
        item = QtWidgets.QTableWidgetItem(os.path.basename(img))
        self.tableWidget.setItem(current_row, 0, item)
        # 設定背景顏色
        if judgement:
            color = QtGui.QColor('#06C8FC')  # 判斷為 True，設置為藍色
        else:
            color = QtGui.QColor('#FFAF60')  # 判斷為 False，設置為橙色
        # 確保單元格項目已經初始化
        for column in range(self.tableWidget.columnCount()):
            cell_item = self.tableWidget.item(current_row, column)
            if cell_item is None:
                cell_item = QtWidgets.QTableWidgetItem()
                self.tableWidget.setItem(current_row, column, cell_item)
            # 設置背景顏色和文字置中
            cell_item.setBackground(color)
            cell_item.setTextAlignment(Qt.AlignCenter)
    def cell_double_clicked(self, row, column):
        if self.step_flag == 1 :
            # 在行點擊時，取得該行的所有資訊
            row_data = [self.tableWidget.item(row, col).text() for col in range(self.tableWidget.columnCount())]
            print(f"雙擊了第 {row + 1} 行，第 {column + 1} 列的資料：", row_data)
            # 修改self.judgement_list
            if self.judgement_list[row] == True:
               self.judgement_list[row] = False
            if self.judgement_list[row] == False:
                self.judgement_list[row] = True
            print(self.judgement_list)

            # 清空重填
            self.GPS_valid_list = []
            self.img_fullpath_valid_list = []
            self.gps_exif = []
            self.tableWidget.setRowCount(0)
            self.flag_first_valid = True
            img_list = sorted(listdir(self.img_fd))
            for i, img_name in enumerate(reversed(img_list)):
                self.img_fullpath = join(self.img_fd,img_name)
                self.gps_exif = read_exif(self.img_fullpath)
                self.qt_table_color(self.img_fullpath, self.gps_exif, self.judgement_list[i])
                if self.judgement_list[i] == True:
                    self.GPS_valid_list.append(self.gps_exif)
                    self.img_fullpath_valid_list.append(self.img_fullpath)
                    if self.flag_first_valid == True:
                        self.flag_first_valid = False
                        # 12/27  計數crop_count、第一個first_true、有效範圍show_limit
                        self.crop_count = i
                        self.first_true = i
                        image = cv2.imread(self.img_fullpath)
                        self.original_img_h, self.original_img_w = image.shape[:2]
                        self.original_center = [self.original_img_w * 0.5, self.original_img_h * 0.5]
                        img_draw = image.copy()
                        cv2.rectangle(img_draw,
                                      (self.conf['valid_area']['rect']['x1'], self.conf['valid_area']['rect']['y1']),
                                      (self.conf['valid_area']['rect']['x2'], self.conf['valid_area']['rect']['y2']),
                                      (0, 255, 255), 5)
                        cv2.rectangle(img_draw, (self.conf['template_area']['x1'], self.conf['template_area']['y1']),
                                      (self.conf['template_area']['x2'], self.conf['template_area']['y2']),
                                      (217, 216, 18), 10)
                        draw_img = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
                        Ny, Nx, channels = draw_img.shape
                        self.image_view = QtGui.QImage(draw_img.data, Nx, Ny, QtGui.QImage.Format_RGB888)
                        self.label.setPixmap(QtGui.QPixmap.fromImage(self.image_view))
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                self.valid_images = list(executor.map(cv2.imread, self.img_fullpath_valid_list))
            print(f'self.GPS_all_list = {self.GPS_all_list}')
            print(f'self.judgement_list = {self.judgement_list}')
            print(f'self.GPS_valid_list = {self.GPS_valid_list}')
            print(f'self.img_fullpath_valid_list = {self.img_fullpath_valid_list}')
            self.show_limit = self.crop_count + len(self.img_fullpath_valid_list)

    # 0704
    def update_table(self, info):
        header = self.tableWidget.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)  # 自動調寬度
        self.tableWidget.setColumnCount(5)
        self.tableWidget.setHorizontalHeaderLabels(['選取', '檔名', '方向', '比對結果', '距離 (m)'])

        filename = self.short_filename(info.path)
        # filename = full_filename[6:]  # 從第 7 個字元開始（含索引 6）
        direction = info.direction if info.direction else "-"
        match_result = str(info.match_coordinate) if info.match_coordinate else "-"
        # 如果沒有值的話，距離先填入-1
        distance_value = info.distance if info.distance is not None else -1
        distance_str = f"{distance_value:.2f}" if distance_value >= 0 else "-"
        # hasattr 是否已有該參數
        if not hasattr(self, 'reference_distance') and distance_value >= 0:
            self.reference_distance = distance_value

        existing_row = -1
        for row in range(self.tableWidget.rowCount()):
            item = self.tableWidget.item(row, 1)  # 改成 column 1 是檔名
            if item and item.text() == filename:
                existing_row = row
                break

        # 建立距離 QTableWidgetItem（紅字邏輯）
        distance_item = QtWidgets.QTableWidgetItem(distance_str)
        if hasattr(self, 'reference_distance') and distance_value >= 0:
            lower_bound = self.reference_distance * 0.5
            upper_bound = self.reference_distance * 1.5
            if distance_value < lower_bound or distance_value > upper_bound:
                # 異常距離，紅字
                distance_item.setForeground(QtGui.QColor(200, 0, 0))

        if existing_row == -1:
            current_row = self.tableWidget.rowCount()
            self.tableWidget.insertRow(current_row)

            # 建立 checkbox 並放到第 0 欄
            checkbox = QtWidgets.QCheckBox()
            checkbox.clicked.connect(lambda checked, row=current_row: self.handle_checkbox_clicked(row, checked))
            checkbox_widget = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout(checkbox_widget)
            layout.addWidget(checkbox)
            layout.setAlignment(QtCore.Qt.AlignCenter)
            layout.setContentsMargins(0, 0, 0, 0)
            self.tableWidget.setCellWidget(current_row, 0, checkbox_widget)

            # 填入資料到其他欄
            self.tableWidget.setItem(current_row, 1, QtWidgets.QTableWidgetItem(filename))
            self.tableWidget.setItem(current_row, 2, QtWidgets.QTableWidgetItem(direction))
            self.tableWidget.setItem(current_row, 3, QtWidgets.QTableWidgetItem(match_result))
            self.tableWidget.setItem(current_row, 4, distance_item)

            # 設定背景
            for col in range(1, 5):  # 跳過 checkbox 欄
                self.tableWidget.item(current_row, col).setBackground(QtGui.QColor(220, 220, 220))
        else:
            # 更新方向與比對結果
            self.tableWidget.setItem(existing_row, 2, QtWidgets.QTableWidgetItem(direction))
            self.tableWidget.setItem(existing_row, 3, QtWidgets.QTableWidgetItem(match_result))
            self.tableWidget.setItem(existing_row, 4, distance_item)
            # 設定綠背景
            for col in range(1, 5):  # 跳過 checkbox 欄
                self.tableWidget.item(existing_row, col).setBackground(QtGui.QColor(80, 160, 80))
    def update_table_row(self, index: int, info: ImageInfo):
        if 0 <= index < self.tableWidget.rowCount():
            filename = self.short_filename(info.path)
            # filename = full_filename[6:]
            direction = info.direction if info.direction else "-"
            match_result = str(info.match_coordinate) if info.match_coordinate else "-"

            item0 = QtWidgets.QTableWidgetItem(filename)
            item0.setBackground(QtGui.QColor(80, 160, 80))  # 若要保持一致也可設背景
            self.tableWidget.setItem(index, 1, item0)

            item1 = QtWidgets.QTableWidgetItem(direction)
            item1.setBackground(QtGui.QColor(80, 160, 80))  # 淡綠色背景
            self.tableWidget.setItem(index, 2, item1)

            item2 = QtWidgets.QTableWidgetItem(match_result)
            item2.setBackground(QtGui.QColor(180, 80, 80))  # 淡紅色背景
            self.tableWidget.setItem(index, 3, item2)
    def rebuild_table(self):
        logging.info("開始重建表格...")
        self.rebuilding = True
        for i in range(1,len(self.watchdog_info_list)):
            prev = self.watchdog_info_list[i-1]
            curr = self.watchdog_info_list[i]
            direction_quadrants = calculate_direction(prev.gps, curr.gps)
            if direction_quadrants == int(self.lineEdit_up.text()):
                curr.direction = 'up'
            elif direction_quadrants == int(self.lineEdit_down.text()):
                curr.direction = 'down'
            elif direction_quadrants == int(self.lineEdit_left.text()):
                curr.direction = 'left'
            elif direction_quadrants == int(self.lineEdit_right.text()):
                curr.direction = 'right'
            curr.distance = gps_distance(prev.gps, curr.gps)

        header = self.tableWidget.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.tableWidget.setRowCount(0)  # 清空所有資料列
        self.tableWidget.setColumnCount(5)
        self.tableWidget.setHorizontalHeaderLabels(['選取', '檔名', '方向', '比對結果','距離 (m)'])
        for info in self.watchdog_info_list:
            filename = self.short_filename(info.path)
            # filename = full_filename[6:]  # 從第 7 個字元開始（含索引 6）
            direction = info.direction if info.direction else "-"
            match_result = str(info.match_coordinate) if info.match_coordinate else "-"
            # 如果沒有值的話，距離先填入-1
            distance_value = info.distance if info.distance is not None else -1
            distance_str = f"{distance_value:.2f}" if distance_value >= 0 else "-"
            # hasattr 是否已有該參數
            if not hasattr(self, 'reference_distance') and distance_value >= 0:
                self.reference_distance = distance_value

            distance_item = QtWidgets.QTableWidgetItem(distance_str)
            if hasattr(self, 'reference_distance') and distance_value >= 0:
                lower_bound = self.reference_distance * 0.5
                upper_bound = self.reference_distance * 1.5
                if distance_value < lower_bound or distance_value > upper_bound:
                    # 異常距離，紅字
                    distance_item.setForeground(QtGui.QColor(200, 0, 0))

            current_row = self.tableWidget.rowCount()
            self.tableWidget.insertRow(current_row)

            # 建立 checkbox 並放到第 0 欄
            checkbox = QtWidgets.QCheckBox()
            checkbox_widget = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout(checkbox_widget)
            layout.addWidget(checkbox)
            layout.setAlignment(QtCore.Qt.AlignCenter)
            layout.setContentsMargins(0, 0, 0, 0)
            self.tableWidget.setCellWidget(current_row, 0, checkbox_widget)

            # 填入其他資料
            self.tableWidget.setItem(current_row, 1, QtWidgets.QTableWidgetItem(filename))
            self.tableWidget.setItem(current_row, 2, QtWidgets.QTableWidgetItem(direction))
            self.tableWidget.setItem(current_row, 3, QtWidgets.QTableWidgetItem(match_result))
            self.tableWidget.setItem(current_row, 4, distance_item)
            # 預設灰底（你可改為記錄之前的顏色狀態）
            for col in range(1, 5):
                self.tableWidget.item(current_row, col).setBackground(QtGui.QColor(80, 160, 80))
        self.rebuilding = False
        logging.info("表格重建完成")
    def cell_clicked(self, row, column):
        if self.step_flag == 1 :
            # 在行點擊時，取得該行的所有資訊
            row_data = [self.tableWidget.item(row, col).text() for col in range(self.tableWidget.columnCount())]
            logging.info(f"點擊了第 {row + 1} 行，第 {column + 1} 列的資料：", row_data)
            # image = cv2.imread(self.img_fullpath_all_list[row])
            image = self.valid_images[row]
            img_draw = image.copy()
            cv2.rectangle(img_draw,
                          (self.conf['template_area']['temp_up']['x1'],
                           self.conf['template_area']['temp_up']['y1']),
                          (self.conf['template_area']['temp_up']['x2'],
                           self.conf['template_area']['temp_up']['y2']), (0, 255, 255), 3)
            cv2.rectangle(img_draw,
                          (self.conf['template_area']['temp_down']['x1'],
                           self.conf['template_area']['temp_down']['y1']),
                          (self.conf['template_area']['temp_down']['x2'],
                           self.conf['template_area']['temp_down']['y2']), (0, 255, 255), 3)
            cv2.rectangle(img_draw,
                          (self.conf['template_area']['temp_left']['x1'],
                           self.conf['template_area']['temp_left']['y1']),
                          (self.conf['template_area']['temp_left']['x2'],
                           self.conf['template_area']['temp_left']['y2']), (0, 255, 255), 3)
            cv2.rectangle(img_draw,
                          (self.conf['template_area']['temp_right']['x1'],
                           self.conf['template_area']['temp_right']['y1']),
                          (self.conf['template_area']['temp_right']['x2'],
                           self.conf['template_area']['temp_right']['y2']), (0, 255, 255), 3)
            draw_img = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
            Ny, Nx, channels = draw_img.shape
            self.image_view = QtGui.QImage(draw_img.data, Nx, Ny, QtGui.QImage.Format_RGB888)
            self.label.setPixmap(QtGui.QPixmap.fromImage(self.image_view))
        if self.step_flag == 2:
            row_data = [self.tableWidget.item(row, col).text() for col in range(self.tableWidget.columnCount())]
            logging.info(f"點擊了第 {row + 1} 行，第 {column + 1} 列的資料：", row_data)
            self.crop_count = row
            self.show_temp()
        if self.step_flag == 200 or self.step_flag == 300:
            # row_data = [self.tableWidget.item(row, col).text() for col in range(self.tableWidget.columnCount())]
            # print(f"點擊了第 {row + 1} 行，第 {column + 1} 列的資料：", row_data)
            if row > 0 :
                self.show_count = row - 1
                self.show_template()
        if self.step_flag == 500:
            if not hasattr(self, 'reorder_selected_row'):
                self.reorder_selected_row = None  # 初始化狀態
            if self.reorder_selected_row is None:
                # 第一次點擊：紀錄要移動的來源列
                self.reorder_selected_row = row
                logging.info(f"已選取第 {row} 行作為來源，請點目標位置")
                # 可以額外標記選取行的背景色
                for col in range(self.tableWidget.columnCount()):
                    item = self.tableWidget.item(row, col)
                    if item:
                        item.setBackground(QtGui.QColor(200, 180, 100))  # 黃色背景
            else:
                # 第二次點擊：進行交換
                from_row = self.reorder_selected_row
                to_row = row
                if from_row != to_row:
                    self.watchdog_info_list[from_row], self.watchdog_info_list[to_row] = \
                        self.watchdog_info_list[to_row], self.watchdog_info_list[from_row]
                    logging.info(f"已交換第 {from_row} 與第 {to_row} 行的資料")
                    self.rebuild_table()
                else:
                    logging.info("點選了同一列，取消交換")
                    for col in range(self.tableWidget.columnCount()):
                        item = self.tableWidget.item(row, col)
                        if item:
                            item.setBackground(QtGui.QColor(80, 160, 80))  # 綠色背景
                self.reorder_selected_row = None
    def handle_checkbox_clicked(self, row, checked):
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if modifiers == QtCore.Qt.ShiftModifier and self.last_checked_row is not None:
            # Shift 被按下，而且已經有先前點過的 row
            start = min(self.last_checked_row, row)
            end = max(self.last_checked_row, row)
            for r in range(start, end + 1):
                checkbox_widget = self.tableWidget.cellWidget(r, 0)
                if checkbox_widget is not None:
                    checkbox = checkbox_widget.findChild(QtWidgets.QCheckBox)
                    if checkbox:
                        checkbox.setChecked(checked)
        else:
            # 沒按 Shift，就只是單純紀錄
            self.last_checked_row = row

    # 0801
    def short_filename(self,path):
        filename = os.path.basename(path)
        name, _ = os.path.splitext(filename)

        # 格式1：時間戳_緯度_經度.jpg
        match = re.match(r'^(\d{14})_', name)
        if match:
            return match.group(1)[8:]  # 回傳時間的 HHMMSS 部分作為識別（最後6碼）

        # 格式2：SkyCAM_0004.JPG 類型
        match = re.search(r'_(\d{4,})$', name)
        if match:
            return f"#{match.group(1)}"  # 加 # 來代表流水編號

        # 格式3：P2835052.JPG 或其他類似
        match = re.search(r'(\d{4,})$', name)
        if match:
            return match.group(1)  # 回傳後幾碼編號即可

        # fallback：前8字
        return name[:8]

    def mousePressEvent(self, event):
        if self.step_flag == 1:
            if event.button() == Qt.LeftButton:
                self.flag_draw = True
                mouse_pos = event.pos()
                label_geometry = self.label.geometry()
                if label_geometry.contains(mouse_pos):
                    label_pos = self.label.mapFromParent(mouse_pos)
                    label_width = self.label.width()
                    label_height = self.label.height()
                    pixmap = self.label.pixmap()
                    if pixmap is not None:
                        image_width = pixmap.width()
                        image_height = pixmap.height()
                        scale_x = image_width / label_width
                        scale_y = image_height / label_height
                        image_x = int(label_pos.x() * scale_x)
                        image_y = int(label_pos.y() * scale_y)
                        print(f"滑鼠在 Label 範圍內按下，相對影像的座標為: ({image_x}, {image_y})")
                        self.rect_start = [image_x, image_y]
        if self.step_flag == 2 or self.step_flag == 3 or self.step_flag == 200 :
            if event.button() == Qt.LeftButton:
                self.flag_draw = True
                mouse_pos = event.pos()
                if self.label_7.geometry().contains(mouse_pos):
                    label_pos = self.label_7.mapFromParent(mouse_pos)
                    pixmap = self.label_7.pixmap()
                    if pixmap is not None:
                        scale_x = pixmap.width() / self.label_7.width()
                        scale_y = pixmap.height() / self.label_7.height()
                        image_x = int(label_pos.x() * scale_x)
                        image_y = int(label_pos.y() * scale_y)
                        print(f"滑鼠在 Label_7 範圍內按下，相對影像的座標為: ({image_x}, {image_y})")
                        self.rect_start = [image_x, image_y]
                elif self.label_6.geometry().contains(mouse_pos):
                    label_pos = self.label_6.mapFromParent(mouse_pos)
                    pixmap = self.label_6.pixmap()
                    if pixmap is not None:
                        scale_x = pixmap.width() / self.label_6.width()
                        scale_y = pixmap.height() / self.label_6.height()
                        image_x = int(label_pos.x() * scale_x)
                        image_y = int(label_pos.y() * scale_y)
                        print(f"點擊 label_6，座標為: ({image_x}, {image_y})")
                        self.rect_start = [image_x, image_y]
        if self.step_flag == 10:
            if event.button() == Qt.LeftButton:
                self.flag_draw = True
                mouse_pos = event.pos()
                if self.label_6.geometry().contains(mouse_pos):
                    label_pos = self.label_6.mapFromParent(mouse_pos)
                    pixmap = self.label_6.pixmap()
                    if pixmap is not None:
                        scale_x = pixmap.width() / self.label_6.width()
                        scale_y = pixmap.height() / self.label_6.height()
                        image_x = int(label_pos.x() * scale_x)
                        image_y = int(label_pos.y() * scale_y)
                        print(f"點擊 label_6，座標為: ({image_x}, {image_y})")
                        self.rect_start = [image_x, image_y]
        if self.mouse_flag:
            if event.buttons() == QtCore.Qt.LeftButton:
                self.left_flag = True
                self.x1 = event.x()
                self.y1 = event.y()
                if self.has_been_chgd == True or self.has_been_moved == True:
                    mv_x = self.x1  - self.label_x  # 減去拖曳座標
                    mv_y = self.y1  - self.label_y
                    # chg_x = mv_x / self.resize_point*10  # 除以縮放倍率
                    # chg_y = mv_y / self.resize_point*10
                    chg_x = mv_x * (self.cur_img.shape[1] / self.label_wid)
                    chg_y = mv_y * (self.cur_img.shape[0] / self.label_hig)
                    self.point_list.append([int(chg_x),int(chg_y)])
                    print(int(chg_x),int(chg_y))
                    cv2.circle(self.cur_img,(int(chg_x),int(chg_y)),10,(0,0,0),-1)
                    # img2 = cv2.cvtColor(self.cur_img,cv2.COLOR_BGRA2RGBA)
                    q_image_format = QtGui.QImage.Format_RGBA8888
                    bytes_per_line = 4 * self.cur_img.shape[1]
                    q_image = QtGui.QImage(self.cur_img, self.cur_img.shape[1], self.cur_img.shape[0], bytes_per_line,q_image_format)
                    pixmap = QtGui.QPixmap(q_image).scaled(self.label.width(), self.label.height())
                    self.label.setPixmap(pixmap)
                # if len(self.point_list) == 2 and self.rotate_flag == True:
                #     dx = self.point_list[1][0] - self.point_list[0][0]
                #     dy = self.point_list[1][1] - self.point_list[0][1]
                #     angle_rad = math.atan2(dy, dx)  # 弧度
                #     angle_deg = math.degrees(angle_rad)  # 轉為角度
                #     print(angle_deg)
                #     (h, w) = self.cur_img.shape[:2]
                #     center = (w // 2, h // 2)
                #     # 計算旋轉後的圖片大小
                #     angle_to_rotate = 270 + angle_deg
                #     rotation_matrix = cv2.getRotationMatrix2D(center, angle_to_rotate, 1.0)
                #     # 旋轉後的邊界尺寸
                #     cos = abs(rotation_matrix[0, 0])
                #     sin = abs(rotation_matrix[0, 1])
                #     new_w = int(h * sin + w * cos)
                #     new_h = int(h * cos + w * sin)
                #     # 12/06 計算新畫布的偏移量
                #     new_canvas_offset = ((new_w - w) / 2, (new_h - h) / 2)
                #     # 調整旋轉矩陣以將圖片中心移到新的畫布中心
                #     rotation_matrix[0, 2] += (new_w / 2) - center[0]
                #     rotation_matrix[1, 2] += (new_h / 2) - center[1]
                #     # 旋轉圖片並調整大小
                #     self.rotated_image = cv2.warpAffine(self.save_img, rotation_matrix, (new_w, new_h), flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
                #     cv2.imwrite('rotated_image.png', self.rotated_image)
                #     self.rotated_image = cv2.cvtColor(self.rotated_image, cv2.COLOR_BGRA2RGBA)
                #     self.cur_img = self.rotated_image.copy()
                #     self.img_h, self.img_w, channels = self.cur_img.shape
                #     self.image_view = QtGui.QImage(self.cur_img.data, self.img_w, self.img_h, QtGui.QImage.Format_RGBA8888)
                #     self.label.setPixmap(QtGui.QPixmap.fromImage(self.image_view))
                #     self.point_list = []
                #     self.rotate_flag = False
                #
                #     # 12/06 用相同的旋轉矩陣，計算xy座標點
                #     rotated_point_list = rotate_points_with_matrix(self.crop_point_list, rotation_matrix)
                #     # 12/09 排序 左上、右上、右下、左下
                #     sort_key = create_sort_key(rotated_point_list)
                #     sorted_points = sorted(rotated_point_list, key=sort_key)
                #     # 最大外接矩形
                #     min_x = min(point[0] for point in sorted_points)
                #     max_x = max(point[0] for point in sorted_points)
                #     min_y = min(point[1] for point in sorted_points)
                #     max_y = max(point[1] for point in sorted_points)
                #     min_x, max_x = max(0, min_x), min(self.cur_img.shape[1], max_x)
                #     min_y, max_y = max(0, min_y), min(self.cur_img.shape[0], max_y)
                #     cropped_image = self.rotated_image[int(min_y):int(max_y), int(min_x):int(max_x)]
                #     cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGRA2RGBA)
                #     cv2.imwrite('cropped_image_auto.png', cropped_image)
                #
                #     # 0113  png2tiff
                #     gps_geotransform(f'cropped_image_auto.png',f'cropped_image_auto.tiff')
                #
                #     print(f"裁剪完成，結果已儲存")
                #     # 12/11 取出指定角點 - 左上
                #     index = rotated_point_list.index(sorted_points[0])
                #     left_top = self.conf['valid_scope'][index]
                #     index = rotated_point_list.index(sorted_points[1])
                #     right_top = self.conf['valid_scope'][index]
                #     index = rotated_point_list.index(sorted_points[3])
                #     left_bottom = self.conf['valid_scope'][index]

                if len(self.point_list) == 4:
                    # 12/09 排序 左上、右上、右下、左下
                    sort_key = create_sort_key(self.point_list)
                    sorted_points = sorted(self.point_list, key=sort_key)
                    # ?取矩形的?界
                    min_x = min(point[0] for point in sorted_points)
                    max_x = max(point[0] for point in sorted_points)
                    min_y = min(point[1] for point in sorted_points)
                    max_y = max(point[1] for point in sorted_points)
                    # 确保坐?合法
                    min_x, max_x = max(0, min_x), min(self.cur_img.shape[1], max_x)
                    min_y, max_y = max(0, min_y), min(self.cur_img.shape[0], max_y)

                    start_time = time.time()
                    # 20250116  透視轉換展平
                    sorted_points_np = np.array(sorted_points, dtype=np.float32)
                    dst_points = np.array([
                        [0, 0],
                        [max_x - min_x, 0],
                        [max_x - min_x, max_y - min_y],
                        [0, max_y - min_y]
                    ], dtype=np.float32)
                    M = cv2.getPerspectiveTransform(sorted_points_np, dst_points)
                    output_shape = (max_x - min_x, max_y - min_y)
                    output_image = np.zeros((output_shape[1], output_shape[0], 3), dtype=np.uint8)
                    warped_image = cv2.warpPerspective(self.save_img, M, (output_shape[0], output_shape[1]),
                                                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                    output_image[:warped_image.shape[0], :warped_image.shape[1]] = warped_image
                    cv2.imwrite('output_image.jpg', output_image)
                    end_time = time.time()
                    print(f'time = {end_time - start_time}')

                    # 0113  png2tiff
                    # gps_geotransform(f'output_image.jpg', f'/data/stitch.tiff')
                    self.show_message()
                    print(f"裁剪完成，結果已儲存")
                    # 12/11 取出指定角點 - 左上
                    left_top = self.conf['valid_scope'][1]
                    right_top = self.conf['valid_scope'][2]
                    left_bottom = self.conf['valid_scope'][0]
                    print(f'left_top={left_top},right_top={right_top},left_bottom={left_bottom}')
                    self.point_list = []
            if event.buttons() == QtCore.Qt.RightButton:
                self.right_flag = True
    def mouseReleaseEvent(self, event):
        if self.step_flag == 1 :
            if self.flag_draw == True:
                mouse_pos = event.pos()
                label_geometry = self.label.geometry()
                if label_geometry.contains(mouse_pos):
                    # 將滑鼠位置從主視窗座標系轉換為 label_7 的本地座標
                    label_pos = self.label.mapFromParent(mouse_pos)
                    # 獲取 label_7 的實際顯示大小
                    label_width = self.label.width()
                    label_height = self.label.height()
                    # 獲取影像的實際大小
                    pixmap = self.label.pixmap()
                    if pixmap is not None:
                        image_width = pixmap.width()
                        image_height = pixmap.height()
                        # 計算縮放比例
                        scale_x = image_width / label_width
                        scale_y = image_height / label_height
                        # 回推到原始影像的實際座標
                        image_x = int(label_pos.x() * scale_x)
                        image_y = int(label_pos.y() * scale_y)
                        print(f"滑鼠在 Label 範圍內放開，相對影像的座標為: ({image_x}, {image_y})")
                        self.rect_end = [image_x, image_y]
                    self.flag_draw == False
                    image = cv2.imread(self.img_fullpath_valid_list[0])
                    img_draw = image.copy()
                    cv2.rectangle(img_draw,
                                  (self.conf['template_area']['temp_up']['x1'],
                                   self.conf['template_area']['temp_up']['y1']),
                                  (self.conf['template_area']['temp_up']['x2'],
                                   self.conf['template_area']['temp_up']['y2']), (0, 255, 255), 3)
                    cv2.rectangle(img_draw,
                                  (self.conf['template_area']['temp_down']['x1'],
                                   self.conf['template_area']['temp_down']['y1']),
                                  (self.conf['template_area']['temp_down']['x2'],
                                   self.conf['template_area']['temp_down']['y2']), (0, 255, 255), 3)
                    cv2.rectangle(img_draw,
                                  (self.conf['template_area']['temp_left']['x1'],
                                   self.conf['template_area']['temp_left']['y1']),
                                  (self.conf['template_area']['temp_left']['x2'],
                                   self.conf['template_area']['temp_left']['y2']), (0, 255, 255), 3)
                    cv2.rectangle(img_draw,
                                  (self.conf['template_area']['temp_right']['x1'],
                                   self.conf['template_area']['temp_right']['y1']),
                                  (self.conf['template_area']['temp_right']['x2'],
                                   self.conf['template_area']['temp_right']['y2']), (0, 255, 255), 3)
                    cv2.rectangle(img_draw,(self.rect_start[0],self.rect_start[1]),(self.rect_end[0],self.rect_end[1]), (22,103,242),10)
                    draw_img = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
                    Ny, Nx, channels = draw_img.shape
                    self.image_view = QtGui.QImage(draw_img.data, Nx, Ny, QtGui.QImage.Format_RGB888)
                    self.label.setPixmap(QtGui.QPixmap.fromImage(self.image_view))
                    self.step_flag = 11
        if self.step_flag == 2 or self.step_flag == 3 :
            if self.flag_draw == True:
                mouse_pos = event.pos()
                if self.label_7.geometry().contains(mouse_pos):  # 修改template位置
                    label_pos = self.label_7.mapFromParent(mouse_pos)
                    pixmap = self.label_7.pixmap()
                    if pixmap is not None:
                        scale_x = pixmap.width() / self.label_7.width()
                        scale_y = pixmap.height() / self.label_7.height()
                        image_x = int(label_pos.x() * scale_x)
                        image_y = int(label_pos.y() * scale_y)
                        print(f"點擊 label_7，座標為: ({image_x}, {image_y})")
                        self.rect_end = [image_x, image_y]
                    self.flag_draw == False
                    self.step_flag = 3
                    self.crop_count = self.crop_count - 1
                    self.show_temp()
                elif self.label_6.geometry().contains(mouse_pos):  # 修改match結果
                    index_to_add = self.crop_count - 1
                    new_data = [index_to_add, self.rect_start]
                    # 先檢查是否已有相同 index 的項目
                    found = False
                    for i, item in enumerate(self.revise_match_list):
                        if item[0] == index_to_add:
                            self.revise_match_list[i] = new_data  # 覆蓋舊值
                            found = True
                            break
                    if not found:
                        self.revise_match_list.append(new_data)
                    # 根據 revise_match_list 更新 template_list
                    for index, new_value in self.revise_match_list:
                        if 0 <= index < len(self.template_list):
                            self.template_list[index] = tuple(new_value)
                    self.flag_draw == False
                    self.step_flag = 2
                    self.crop_count = self.crop_count -1
                    self.show_temp()
        if self.step_flag == 200 :
            if self.flag_draw == True:
                mouse_pos = event.pos()
                if self.label_7.geometry().contains(mouse_pos):  # 修改template位置
                    label_pos = self.label_7.mapFromParent(mouse_pos)
                    pixmap = self.label_7.pixmap()
                    if pixmap is not None:
                        scale_x = pixmap.width() / self.label_7.width()
                        scale_y = pixmap.height() / self.label_7.height()
                        image_x = int(label_pos.x() * scale_x)
                        image_y = int(label_pos.y() * scale_y)
                        print(f"點擊 label_7，座標為: ({image_x}, {image_y})")
                        self.rect_end = [image_x, image_y]
                    self.flag_draw = False
                    self.step_flag = 300   # 讓show會顯示新畫的template
                    self.show_count = self.show_count - 1
                    self.show_template()
                elif self.label_6.geometry().contains(mouse_pos):  # 修改match結果
                    index_to_add = self.show_count
                    new_data = [index_to_add, self.rect_start]
                    # 先檢查是否已有相同 index 的項目
                    found = False
                    for i, item in enumerate(self.revise_match_list):
                        if item[0] == index_to_add:
                            self.revise_match_list[i] = new_data  # 覆蓋舊值
                            found = True
                            break
                    if not found:
                        self.revise_match_list.append(new_data)
                    # 根據 revise_match_list 更新 template_list
                    # for index, new_value in self.revise_match_list:
                    #     if 0 <= index < len(self.template_list):
                    #         self.template_list[index] = tuple(new_value)
                    for index, new_value in self.revise_match_list:
                        if 0 <= index < len(self.watchdog_info_list):
                            print(f'修改info第{index}張圖')
                            print(f'修改前{self.watchdog_info_list[index].match_coordinate},修改後{list(new_value)}')
                            self.watchdog_info_list[index].match_coordinate = list(new_value)
                            self.watchdog_info_list[index].revise = True
                            self.update_table_row(index, self.watchdog_info_list[index])
                    self.flag_draw = False
                    self.show_count = self.show_count - 1
                    self.show_template()
        if self.step_flag == 10:
            if self.flag_draw == True:
                mouse_pos = event.pos()
                if self.label_6.geometry().contains(mouse_pos):
                    prev_show = self.prev_cropped.copy()
                    cv2.rectangle(prev_show, (self.rect_start[0], self.rect_start[1]),
                                  (self.rect_start[0] + 400, self.rect_start[1] + 400), (12, 31, 242), 10)
                    prev_show = cv2.cvtColor(prev_show, cv2.COLOR_BGR2RGB)
                    Ny, Nx, channels = prev_show.shape
                    self.prev_view = QtGui.QImage(prev_show.data, Nx, Ny, QtGui.QImage.Format_RGB888)
                    self.label_6.setPixmap(QtGui.QPixmap.fromImage(self.prev_view))
                    h, w = self.curr_img.shape[:2]
                    cx, cy = w // 2, h // 2  # 中心點
                    x1, x2 = cx - 200, cx + 200
                    y1, y2 = cy - 200, cy + 200
                    self.big_stitch_offset = [self.rect_start[0] - x1, self.rect_start[1] - y1]
        self.left_flag = False
        self.right_flag = False
        self.mouse_mv_y = ""
        self.mouse_mv_x = ""
    def mouseMoveEvent(self, event):
        if self.mouse_flag:
            if self.right_flag:
                self.x1 = event.x()
                self.y1 = event.y()
                if self.mouse_mv_x !="" and self.mouse_mv_y !="":
                    self.label_x = self.label_x + (self.x1 - self.mouse_mv_x)
                    self.label_y = self.label_y + (self.y1 - self.mouse_mv_y)
                self.mouse_mv_x = self.x1
                self.mouse_mv_y = self.y1
                self.label.setGeometry(QtCore.QRect(self.label_x, self.label_y, self.label.width(), self.label.height()))
                self.has_been_moved = True
    def wheelEvent(self, event):
        if self.mouse_flag:
            self.angle = event.angleDelta() / 8
            self.angleY = self.angle.y()
            if self.angleY > 0:
                if self.resize_point >= 1 and self.resize_point <= 20:
                    self.resize_point += 1
            elif self.angleY < 0:
                if self.resize_point >= 2 and self.resize_point <= 21:
                    self.resize_point -= 1
            # 計算縮放比例
            scale = self.resize_point / 10
            # 取得目前視窗的寬和高（UI中心點）
            ui_center_x = self.width() // 2
            ui_center_y = self.height() // 2
            # 計算縮放後影像的寬和高
            new_width = int(self.cur_img.shape[1] * scale)
            new_height = int(self.cur_img.shape[0] * scale)
            # 計算縮放後影像左上角的位置，使影像中心對齊UI中心
            self.label_x = ui_center_x - (new_width // 2)
            self.label_y = ui_center_y - (new_height // 2)
            # 進行影像縮放
            self.cur_resimg = cv2.resize(self.cur_img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            # 將影像轉換為QPixmap並顯示
            q_image_format = QtGui.QImage.Format_RGBA8888
            bytes_per_line = 4 * self.cur_resimg.shape[1]
            q_image = QtGui.QImage(self.cur_resimg, self.cur_resimg.shape[1], self.cur_resimg.shape[0], bytes_per_line,q_image_format)
            pixmap = QtGui.QPixmap(q_image).scaled(new_width, new_height)
            self.label_wid = self.cur_resimg.shape[1]
            self.label_hig = self.cur_resimg.shape[0]
            # 更新label的位置和大小
            self.label.setGeometry(QtCore.QRect(self.label_x, self.label_y, new_width, new_height))
            self.label.setPixmap(pixmap)
            # 標記狀態已改變
            self.has_been_chgd = True
    def keyPressEvent(self, event):
        self.Press_key = event.key()
        if self.Press_key == 16777249:
            if self.step_flag == 4:
                self.step_flag = 2
                self.end_crop(self.stitch_offset_list,self.template_list)
        #     stitch_res = cv2.imread(f'./result/stitch/stitch_{len(self.img_fullpath_valid_list) - 1}.png',cv2.IMREAD_UNCHANGED)
        #     self.cur_img = stitch_res
        #     self.save_img = self.cur_img.copy()
        #     self.img_h, self.img_w, channels = self.cur_img.shape
        #     self.cur_img = cv2.cvtColor(self.cur_img, cv2.COLOR_BGRA2RGBA)
        #     self.image_view = QtGui.QImage(self.cur_img.data, self.img_w, self.img_h, QtGui.QImage.Format_RGBA8888)
        #     self.label.setPixmap(QtGui.QPixmap.fromImage(self.image_view))
        #     self.point_list = []
        #     self.rotate_flag = True
    def set_lineedit(self):
        if self.conf == None:
            conf_file = open(self.lineEdit_config.text(), 'r')
            self.conf = yaml.safe_load(conf_file)
        self.lineEdit_up_x1.setText(str(self.conf['template_area']['temp_up']['x1']))
        self.lineEdit_up_y1.setText(str(self.conf['template_area']['temp_up']['y1']))
        self.lineEdit_up_x2.setText(str(self.conf['template_area']['temp_up']['x2']))
        self.lineEdit_up_y2.setText(str(self.conf['template_area']['temp_up']['y2']))

        self.lineEdit_down_x1.setText(str(self.conf['template_area']['temp_down']['x1']))
        self.lineEdit_down_y1.setText(str(self.conf['template_area']['temp_down']['y1']))
        self.lineEdit_down_x2.setText(str(self.conf['template_area']['temp_down']['x2']))
        self.lineEdit_down_y2.setText(str(self.conf['template_area']['temp_down']['y2']))

        self.lineEdit_left_x1.setText(str(self.conf['template_area']['temp_left']['x1']))
        self.lineEdit_left_y1.setText(str(self.conf['template_area']['temp_left']['y1']))
        self.lineEdit_left_x2.setText(str(self.conf['template_area']['temp_left']['x2']))
        self.lineEdit_left_y2.setText(str(self.conf['template_area']['temp_left']['y2']))

        self.lineEdit_right_x1.setText(str(self.conf['template_area']['temp_right']['x1']))
        self.lineEdit_right_y1.setText(str(self.conf['template_area']['temp_right']['y1']))
        self.lineEdit_right_x2.setText(str(self.conf['template_area']['temp_right']['x2']))
        self.lineEdit_right_y2.setText(str(self.conf['template_area']['temp_right']['y2']))

        self.lineEdit_up.setText(str(self.conf['direction']['up']))
        self.lineEdit_down.setText(str(self.conf['direction']['down']))
        self.lineEdit_left.setText(str(self.conf['direction']['left']))
        self.lineEdit_right.setText(str(self.conf['direction']['right']))
if __name__=='__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())