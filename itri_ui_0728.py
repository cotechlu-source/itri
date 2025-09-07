# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'itri_ui_layout.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1845, 1000)
        font = QtGui.QFont()
        font.setFamily("Microsoft JhengHei")
        font.setBold(True)
        font.setWeight(75)
        MainWindow.setFont(font)
        MainWindow.setStyleSheet("QMainWindow {\n"
"     background-color: #1e1e1e;\n"
"     color: #f0f0f0;\n"
" }\n"
"\n"
" QWidget {\n"
"     background-color: #2d2d2d;\n"
"     color: #ffffff;\n"
" }\n"
"\n"
" QLabel {\n"
"     color: #ffffff;\n"
"     background-color: transparent;\n"
" }\n"
"\n"
" QPushButton {\n"
"     background-color: #3a3a3a;\n"
"     color: #ffffff;\n"
"     border-radius: 10px;\n"
"     padding: 5px;\n"
" }\n"
"\n"
" QPushButton:hover {\n"
"     background-color: #505050;\n"
" }\n"
"\n"
" QPushButton:pressed {\n"
"     background-color: #2d2d2d;\n"
" }\n"
"\n"
" QLineEdit, QTextEdit {\n"
"     background-color: #1e1e1e;\n"
"     color: #ffffff;\n"
"     border: 1px solid #555555;\n"
" }\n"
"\n"
" QTableWidget {\n"
"     background-color: #1e1e1e;\n"
"     color: #ffffff;\n"
"     gridline-color: #444444;\n"
" }\n"
"\n"
" QHeaderView::section {\n"
"     background-color: #3a3a3a;\n"
"     color: #ffffff;\n"
"     padding: 4px;\n"
" }\n"
"\n"
" QProgressBar {\n"
"     border: 1px solid #555;\n"
"     border-radius: 5px;\n"
"     text-align: center;\n"
"     color: #ffffff;\n"
"     background-color: #2d2d2d;\n"
" }\n"
"\n"
" QProgressBar::chunk {\n"
"     background-color: #06C8FC;\n"
" }\n"
"\n"
" QGraphicsView {\n"
"     background-color: #1e1e1e;\n"
" }\n"
"\n"
" QStatusBar {\n"
"     background-color: #2d2d2d;\n"
"     color: #ffffff;\n"
" }")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(110, 220, 361, 571))
        self.widget.setAutoFillBackground(False)
        self.widget.setStyleSheet("border: 2px solid black;\n"
"")
        self.widget.setObjectName("widget")
        self.label_up_view = QtWidgets.QLabel(self.widget)
        self.label_up_view.setGeometry(QtCore.QRect(90, 150, 31, 31))
        self.label_up_view.setObjectName("label_up_view")
        self.label_right_view = QtWidgets.QLabel(self.widget)
        self.label_right_view.setGeometry(QtCore.QRect(300, 150, 31, 31))
        self.label_right_view.setObjectName("label_right_view")
        self.label_down_view = QtWidgets.QLabel(self.widget)
        self.label_down_view.setGeometry(QtCore.QRect(160, 150, 31, 31))
        self.label_down_view.setObjectName("label_down_view")
        self.label_left_view = QtWidgets.QLabel(self.widget)
        self.label_left_view.setGeometry(QtCore.QRect(230, 150, 31, 31))
        self.label_left_view.setObjectName("label_left_view")
        self.lineEdit_up = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_up.setGeometry(QtCore.QRect(90, 200, 31, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.lineEdit_up.setFont(font)
        self.lineEdit_up.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.lineEdit_up.setObjectName("lineEdit_up")
        self.lineEdit_left = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_left.setGeometry(QtCore.QRect(230, 200, 31, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.lineEdit_left.setFont(font)
        self.lineEdit_left.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.lineEdit_left.setObjectName("lineEdit_left")
        self.lineEdit_right = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_right.setGeometry(QtCore.QRect(300, 200, 31, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.lineEdit_right.setFont(font)
        self.lineEdit_right.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.lineEdit_right.setObjectName("lineEdit_right")
        self.lineEdit_down = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_down.setGeometry(QtCore.QRect(160, 200, 31, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.lineEdit_down.setFont(font)
        self.lineEdit_down.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.lineEdit_down.setObjectName("lineEdit_down")
        self.label_x1 = QtWidgets.QLabel(self.widget)
        self.label_x1.setGeometry(QtCore.QRect(90, 260, 31, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.label_x1.setFont(font)
        self.label_x1.setObjectName("label_x1")
        self.label_y2 = QtWidgets.QLabel(self.widget)
        self.label_y2.setGeometry(QtCore.QRect(300, 260, 31, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.label_y2.setFont(font)
        self.label_y2.setObjectName("label_y2")
        self.label_x2 = QtWidgets.QLabel(self.widget)
        self.label_x2.setGeometry(QtCore.QRect(230, 260, 31, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.label_x2.setFont(font)
        self.label_x2.setObjectName("label_x2")
        self.label_y1 = QtWidgets.QLabel(self.widget)
        self.label_y1.setGeometry(QtCore.QRect(160, 260, 31, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.label_y1.setFont(font)
        self.label_y1.setObjectName("label_y1")
        self.lineEdit_up_x1 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_up_x1.setGeometry(QtCore.QRect(70, 300, 61, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.lineEdit_up_x1.setFont(font)
        self.lineEdit_up_x1.setObjectName("lineEdit_up_x1")
        self.lineEdit_up_y1 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_up_y1.setGeometry(QtCore.QRect(140, 300, 61, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.lineEdit_up_y1.setFont(font)
        self.lineEdit_up_y1.setObjectName("lineEdit_up_y1")
        self.lineEdit_up_x2 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_up_x2.setGeometry(QtCore.QRect(220, 300, 61, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.lineEdit_up_x2.setFont(font)
        self.lineEdit_up_x2.setObjectName("lineEdit_up_x2")
        self.lineEdit_up_y2 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_up_y2.setGeometry(QtCore.QRect(290, 300, 61, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.lineEdit_up_y2.setFont(font)
        self.lineEdit_up_y2.setObjectName("lineEdit_up_y2")
        self.label_up = QtWidgets.QLabel(self.widget)
        self.label_up.setGeometry(QtCore.QRect(10, 300, 41, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.label_up.setFont(font)
        self.label_up.setObjectName("label_up")
        self.label_down = QtWidgets.QLabel(self.widget)
        self.label_down.setGeometry(QtCore.QRect(10, 340, 41, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.label_down.setFont(font)
        self.label_down.setObjectName("label_down")
        self.label_left = QtWidgets.QLabel(self.widget)
        self.label_left.setGeometry(QtCore.QRect(10, 380, 41, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.label_left.setFont(font)
        self.label_left.setObjectName("label_left")
        self.label_right = QtWidgets.QLabel(self.widget)
        self.label_right.setGeometry(QtCore.QRect(10, 420, 41, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.label_right.setFont(font)
        self.label_right.setObjectName("label_right")
        self.lineEdit_down_x2 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_down_x2.setGeometry(QtCore.QRect(220, 340, 61, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.lineEdit_down_x2.setFont(font)
        self.lineEdit_down_x2.setObjectName("lineEdit_down_x2")
        self.lineEdit_down_y2 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_down_y2.setGeometry(QtCore.QRect(290, 340, 61, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.lineEdit_down_y2.setFont(font)
        self.lineEdit_down_y2.setObjectName("lineEdit_down_y2")
        self.lineEdit_down_x1 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_down_x1.setGeometry(QtCore.QRect(70, 340, 61, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.lineEdit_down_x1.setFont(font)
        self.lineEdit_down_x1.setObjectName("lineEdit_down_x1")
        self.lineEdit_down_y1 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_down_y1.setGeometry(QtCore.QRect(140, 340, 61, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.lineEdit_down_y1.setFont(font)
        self.lineEdit_down_y1.setObjectName("lineEdit_down_y1")
        self.lineEdit_left_x2 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_left_x2.setGeometry(QtCore.QRect(220, 380, 61, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.lineEdit_left_x2.setFont(font)
        self.lineEdit_left_x2.setObjectName("lineEdit_left_x2")
        self.lineEdit_left_y2 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_left_y2.setGeometry(QtCore.QRect(290, 380, 61, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.lineEdit_left_y2.setFont(font)
        self.lineEdit_left_y2.setObjectName("lineEdit_left_y2")
        self.lineEdit_left_x1 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_left_x1.setGeometry(QtCore.QRect(70, 380, 61, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.lineEdit_left_x1.setFont(font)
        self.lineEdit_left_x1.setObjectName("lineEdit_left_x1")
        self.lineEdit_left_y1 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_left_y1.setGeometry(QtCore.QRect(140, 380, 61, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.lineEdit_left_y1.setFont(font)
        self.lineEdit_left_y1.setObjectName("lineEdit_left_y1")
        self.lineEdit_right_x2 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_right_x2.setGeometry(QtCore.QRect(220, 420, 61, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.lineEdit_right_x2.setFont(font)
        self.lineEdit_right_x2.setObjectName("lineEdit_right_x2")
        self.lineEdit_right_y2 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_right_y2.setGeometry(QtCore.QRect(290, 420, 61, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.lineEdit_right_y2.setFont(font)
        self.lineEdit_right_y2.setObjectName("lineEdit_right_y2")
        self.lineEdit_right_x1 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_right_x1.setGeometry(QtCore.QRect(70, 420, 61, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.lineEdit_right_x1.setFont(font)
        self.lineEdit_right_x1.setObjectName("lineEdit_right_x1")
        self.lineEdit_right_y1 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_right_y1.setGeometry(QtCore.QRect(140, 420, 61, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.lineEdit_right_y1.setFont(font)
        self.lineEdit_right_y1.setObjectName("lineEdit_right_y1")
        self.btn_write = QtWidgets.QPushButton(self.widget)
        self.btn_write.setGeometry(QtCore.QRect(10, 20, 161, 40))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btn_write.setFont(font)
        self.btn_write.setStyleSheet("QPushButton {\n"
"background-color: rgb(0, 70, 176);\n"
"border-radius: 15px;}\n"
"QPushButton:hover {\n"
"    background-color: #D2E9FF; \n"
"}\n"
"QPushButton:pressed {\n"
"    background-color: rgb(0, 70, 176);\n"
"}")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("icon2/edit_square_24dp_5F6368_FILL0_wght400_GRAD0_opsz24.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_write.setIcon(icon)
        self.btn_write.setIconSize(QtCore.QSize(30, 30))
        self.btn_write.setObjectName("btn_write")
        self.btn_load = QtWidgets.QPushButton(self.widget)
        self.btn_load.setGeometry(QtCore.QRect(190, 20, 161, 40))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btn_load.setFont(font)
        self.btn_load.setStyleSheet("QPushButton {\n"
"background-color: rgb(0, 70, 176);\n"
"border-radius: 15px;}\n"
"QPushButton:hover {\n"
"    background-color: #D2E9FF; \n"
"}\n"
"QPushButton:pressed {\n"
"    background-color: rgb(0, 70, 176);\n"
"}")
        self.btn_load.setIcon(icon)
        self.btn_load.setIconSize(QtCore.QSize(30, 30))
        self.btn_load.setObjectName("btn_load")
        self.lineEdit_config = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_config.setGeometry(QtCore.QRect(10, 90, 341, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.lineEdit_config.setFont(font)
        self.lineEdit_config.setObjectName("lineEdit_config")
        self.checkBox_brightness = QtWidgets.QCheckBox(self.widget)
        self.checkBox_brightness.setGeometry(QtCore.QRect(10, 460, 73, 16))
        self.checkBox_brightness.setObjectName("checkBox_brightness")
        self.checkBox_vignetting = QtWidgets.QCheckBox(self.widget)
        self.checkBox_vignetting.setGeometry(QtCore.QRect(10, 490, 73, 16))
        self.checkBox_vignetting.setObjectName("checkBox_vignetting")
        self.checkBox_enblend = QtWidgets.QCheckBox(self.widget)
        self.checkBox_enblend.setGeometry(QtCore.QRect(10, 520, 73, 16))
        self.checkBox_enblend.setObjectName("checkBox_enblend")
        self.lineEdit_vignetting = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_vignetting.setGeometry(QtCore.QRect(100, 490, 61, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.lineEdit_vignetting.setFont(font)
        self.lineEdit_vignetting.setObjectName("lineEdit_vignetting")
        self.lineEdit_enblend = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_enblend.setGeometry(QtCore.QRect(100, 520, 61, 20))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.lineEdit_enblend.setFont(font)
        self.lineEdit_enblend.setObjectName("lineEdit_enblend")
        self.pushButton_test = QtWidgets.QPushButton(self.widget)
        self.pushButton_test.setGeometry(QtCore.QRect(240, 460, 75, 23))
        self.pushButton_test.setObjectName("pushButton_test")
        self.pushButton_test_2 = QtWidgets.QPushButton(self.widget)
        self.pushButton_test_2.setGeometry(QtCore.QRect(240, 490, 75, 23))
        self.pushButton_test_2.setObjectName("pushButton_test_2")
        self.pushButton_test_3 = QtWidgets.QPushButton(self.widget)
        self.pushButton_test_3.setGeometry(QtCore.QRect(240, 520, 75, 23))
        self.pushButton_test_3.setObjectName("pushButton_test_3")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(490, 20, 250, 41))
        self.label_2.setMinimumSize(QtCore.QSize(250, 0))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(22)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("")
        self.label_2.setScaledContents(False)
        self.label_2.setObjectName("label_2")
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(1180, 10, 651, 951))
        self.graphicsView.setObjectName("graphicsView")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(480, 10, 1152, 864))
        self.label.setAutoFillBackground(False)
        self.label.setStyleSheet("")
        self.label.setText("")
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(480, 10, 684, 456))
        self.label_6.setAutoFillBackground(False)
        self.label_6.setStyleSheet("border: 2px solid white;\n"
"")
        self.label_6.setText("")
        self.label_6.setScaledContents(True)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(480, 500, 684, 456))
        self.label_7.setAutoFillBackground(False)
        self.label_7.setStyleSheet("border: 2px solid white;")
        self.label_7.setText("")
        self.label_7.setScaledContents(True)
        self.label_7.setObjectName("label_7")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(490, 510, 251, 41))
        self.label_3.setMinimumSize(QtCore.QSize(250, 0))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(22)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("")
        self.label_3.setObjectName("label_3")
        self.tableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget.setEnabled(False)
        self.tableWidget.setGeometry(QtCore.QRect(10, 220, 461, 741))
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.tableWidget.setFont(font)
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setRowCount(0)
        self.tableWidget.horizontalHeader().setStretchLastSection(True)
        self.tableWidget.verticalHeader().setStretchLastSection(False)
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 20, 463, 191))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.btn_open_dir = QtWidgets.QPushButton(self.verticalLayoutWidget)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 70, 176))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 70, 176))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 70, 176))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 70, 176))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 70, 176))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 70, 176))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 70, 176))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 70, 176))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 70, 176))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        self.btn_open_dir.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.btn_open_dir.setFont(font)
        self.btn_open_dir.setStyleSheet("QPushButton {\n"
"background-color: rgb(0, 70, 176);\n"
"border-radius: 15px;}\n"
"QPushButton:hover {\n"
"    background-color: #D2E9FF; \n"
"}\n"
"QPushButton:pressed {\n"
"    background-color: rgb(0, 70, 176);\n"
"}")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("icon2/folder_open_24dp_5F6368_FILL0_wght400_GRAD0_opsz24.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_open_dir.setIcon(icon1)
        self.btn_open_dir.setIconSize(QtCore.QSize(50, 50))
        self.btn_open_dir.setObjectName("btn_open_dir")
        self.horizontalLayout.addWidget(self.btn_open_dir)
        self.pushButton_rematching = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_rematching.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_rematching.setFont(font)
        self.pushButton_rematching.setStyleSheet("QPushButton {\n"
"background-color: rgb(0, 70, 176);\n"
"border-radius: 15px;}\n"
"QPushButton:hover {\n"
"    background-color: #D2E9FF; \n"
"}\n"
"QPushButton:pressed {\n"
"    background-color: rgb(0, 70, 176);\n"
"}")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("icon2/branding_watermark_24dp_5F6368_FILL0_wght400_GRAD0_opsz24.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_rematching.setIcon(icon2)
        self.pushButton_rematching.setIconSize(QtCore.QSize(50, 50))
        self.pushButton_rematching.setObjectName("pushButton_rematching")
        self.horizontalLayout.addWidget(self.pushButton_rematching)
        self.pushButton_stitch = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_stitch.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_stitch.setFont(font)
        self.pushButton_stitch.setStyleSheet("QPushButton {\n"
"background-color: rgb(0, 70, 176);\n"
"border-radius: 15px;}\n"
"QPushButton:hover {\n"
"    background-color: #D2E9FF; \n"
"}\n"
"QPushButton:pressed {\n"
"    background-color: rgb(0, 70, 176);\n"
"}")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("icon2/stack_24dp_5F6368_FILL0_wght400_GRAD0_opsz24.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_stitch.setIcon(icon3)
        self.pushButton_stitch.setIconSize(QtCore.QSize(50, 50))
        self.pushButton_stitch.setObjectName("pushButton_stitch")
        self.horizontalLayout.addWidget(self.pushButton_stitch)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButton_show = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_show.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_show.setFont(font)
        self.pushButton_show.setStyleSheet("QPushButton {\n"
"background-color: rgb(0, 70, 176);\n"
"border-radius: 15px;}\n"
"QPushButton:hover {\n"
"    background-color: #D2E9FF; \n"
"}\n"
"QPushButton:pressed {\n"
"    background-color: rgb(0, 70, 176);\n"
"}")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("icon2/exit_to_app_24dp_5F6368_FILL0_wght400_GRAD0_opsz24.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_show.setIcon(icon4)
        self.pushButton_show.setIconSize(QtCore.QSize(30, 30))
        self.pushButton_show.setObjectName("pushButton_show")
        self.horizontalLayout_2.addWidget(self.pushButton_show)
        self.textEdit = QtWidgets.QTextEdit(self.verticalLayoutWidget)
        self.textEdit.setObjectName("textEdit")
        self.horizontalLayout_2.addWidget(self.textEdit)
        self.pushButton_config = QtWidgets.QPushButton(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_config.setFont(font)
        self.pushButton_config.setStyleSheet("QPushButton {\n"
"background-color: rgb(0, 70, 176);\n"
"border-radius: 15px;}\n"
"QPushButton:hover {\n"
"    background-color: #D2E9FF; \n"
"}\n"
"QPushButton:pressed {\n"
"    background-color: rgb(0, 70, 176);\n"
"}")
        self.pushButton_config.setObjectName("pushButton_config")
        self.horizontalLayout_2.addWidget(self.pushButton_config)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.progressBar = QtWidgets.QProgressBar(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.progressBar.setFont(font)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.horizontalLayout_3.addWidget(self.progressBar)
        self.btn_change = QtWidgets.QPushButton(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btn_change.setFont(font)
        self.btn_change.setStyleSheet("QPushButton {\n"
"background-color: rgb(0, 70, 176);\n"
"border-radius: 15px;}\n"
"QPushButton:hover {\n"
"    background-color: #D2E9FF; \n"
"}\n"
"QPushButton:pressed {\n"
"    background-color: rgb(0, 70, 176);\n"
"}")
        self.btn_change.setIconSize(QtCore.QSize(30, 30))
        self.btn_change.setObjectName("btn_change")
        self.horizontalLayout_3.addWidget(self.btn_change)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.label_7.raise_()
        self.label.raise_()
        self.label_3.raise_()
        self.graphicsView.raise_()
        self.label_6.raise_()
        self.label_2.raise_()
        self.tableWidget.raise_()
        self.verticalLayoutWidget.raise_()
        self.widget.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_up_view.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:18pt; font-weight:600;\">↑</span></p></body></html>"))
        self.label_right_view.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:18pt; font-weight:600;\">→</span></p></body></html>"))
        self.label_down_view.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:18pt; font-weight:600;\">↓</span></p></body></html>"))
        self.label_left_view.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:18pt; font-weight:600;\">←</span></p></body></html>"))
        self.lineEdit_up.setText(_translate("MainWindow", "2"))
        self.lineEdit_left.setText(_translate("MainWindow", "3"))
        self.lineEdit_right.setText(_translate("MainWindow", "1"))
        self.lineEdit_down.setText(_translate("MainWindow", "4"))
        self.label_x1.setText(_translate("MainWindow", "x1:"))
        self.label_y2.setText(_translate("MainWindow", "y2:"))
        self.label_x2.setText(_translate("MainWindow", "x2:"))
        self.label_y1.setText(_translate("MainWindow", "y1:"))
        self.label_up.setText(_translate("MainWindow", "up"))
        self.label_down.setText(_translate("MainWindow", "down"))
        self.label_left.setText(_translate("MainWindow", "left"))
        self.label_right.setText(_translate("MainWindow", "right"))
        self.btn_write.setText(_translate("MainWindow", "Config Update"))
        self.btn_load.setText(_translate("MainWindow", "Config Load"))
        self.lineEdit_config.setText(_translate("MainWindow", "config.yaml"))
        self.checkBox_brightness.setText(_translate("MainWindow", "Brightness"))
        self.checkBox_vignetting.setText(_translate("MainWindow", "Vignetting"))
        self.checkBox_enblend.setText(_translate("MainWindow", "Enblend"))
        self.lineEdit_vignetting.setText(_translate("MainWindow", "1.5"))
        self.lineEdit_enblend.setText(_translate("MainWindow", "10"))
        self.pushButton_test.setText(_translate("MainWindow", "test"))
        self.pushButton_test_2.setText(_translate("MainWindow", "Rematching"))
        self.pushButton_test_3.setText(_translate("MainWindow", "Stitch"))
        self.label_2.setText(_translate("MainWindow", "Previous Match"))
        self.label_3.setText(_translate("MainWindow", "Current Template"))
        self.btn_open_dir.setText(_translate("MainWindow", "Step 1.\n"
"Select Dir"))
        self.pushButton_rematching.setText(_translate("MainWindow", "Step 2.\n"
"Match"))
        self.pushButton_stitch.setText(_translate("MainWindow", "Step 3.\n"
"Stitching"))
        self.pushButton_show.setText(_translate("MainWindow", "Show Next Pair"))
        self.textEdit.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">test</p></body></html>"))
        self.pushButton_config.setText(_translate("MainWindow", "Config"))
        self.btn_change.setText(_translate("MainWindow", "Change Mode"))

