# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'demofront.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1582, 694)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(0, 120, 1581, 51))
        self.pushButton.setObjectName("pushButton")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(130, 0, 181, 41))
        self.textEdit.setObjectName("textEdit")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 5, 151, 31))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(130, 50, 181, 16))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(320, 0, 121, 41))
        self.label_3.setObjectName("label_3")
        self.textEdit_2 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_2.setGeometry(QtCore.QRect(410, 0, 201, 41))
        self.textEdit_2.setObjectName("textEdit_2")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(410, 45, 181, 20))
        self.label_4.setObjectName("label_4")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(-10, 170, 1591, 51))
        self.pushButton_2.setObjectName("pushButton_2")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(0, 70, 791, 41))
        self.label_5.setObjectName("label_5")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(640, 0, 941, 121))
        self.textBrowser.setObjectName("textBrowser")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(610, 10, 31, 21))
        self.label_6.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.label_6.setObjectName("label_6")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(0, 220, 141, 101))
        self.pushButton_3.setObjectName("pushButton_3")
        self.textBrowser_2 = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_2.setGeometry(QtCore.QRect(0, 320, 1581, 351))
        self.textBrowser_2.setObjectName("textBrowser_2")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(140, 220, 141, 101))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(280, 220, 141, 101))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_6 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_6.setGeometry(QtCore.QRect(420, 220, 141, 101))
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_7 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_7.setGeometry(QtCore.QRect(1380, 220, 201, 21))
        self.pushButton_7.setObjectName("pushButton_7")
        self.pushButton_8 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_8.setGeometry(QtCore.QRect(1380, 240, 201, 21))
        self.pushButton_8.setObjectName("pushButton_8")
        self.pushButton_9 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_9.setGeometry(QtCore.QRect(1380, 260, 201, 21))
        self.pushButton_9.setObjectName("pushButton_9")
        self.pushButton_10 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_10.setGeometry(QtCore.QRect(1380, 280, 201, 21))
        self.pushButton_10.setObjectName("pushButton_10")
        self.pushButton_11 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_11.setGeometry(QtCore.QRect(1380, 300, 201, 21))
        self.pushButton_11.setObjectName("pushButton_11")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.pushButton.clicked.connect(MainWindow.getinputandstart)
        self.pushButton_2.clicked.connect(MainWindow.preprocessing)
        self.pushButton_3.clicked.connect(MainWindow.getmodelandpredict)
        self.pushButton_4.clicked.connect(MainWindow.EURAUDmodel)
        self.pushButton_5.clicked.connect(MainWindow.GBPCHFmodel)
        self.pushButton_6.clicked.connect(MainWindow.GBPJPYmodel)
        self.pushButton_7.clicked.connect(MainWindow.dispalay0)
        self.pushButton_8.clicked.connect(MainWindow.dispalay1)
        self.pushButton_9.clicked.connect(MainWindow.dispalay2)
        self.pushButton_10.clicked.connect(MainWindow.dispalay3)
        self.pushButton_11.clicked.connect(MainWindow.dispalay4)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "宏观预测demo"))
        self.pushButton.setText(_translate("MainWindow", "启动爬虫"))
        self.label.setText(_translate("MainWindow", "     开始时间："))
        self.label_2.setText(_translate("MainWindow", "格式：2021-01-24 早"))
        self.label_3.setText(_translate("MainWindow", "结束时间："))
        self.label_4.setText(_translate("MainWindow", "格式：2021-3-25   晚"))
        self.pushButton_2.setText(_translate("MainWindow", "数据预处理"))
        self.label_5.setText(_translate("MainWindow", "类别介绍：“0”：<100；“1”:100~200；“2”：200~300；“3”：300~400；“4”：>400"))
        self.label_6.setText(_translate("MainWindow", "状态"))
        self.pushButton_3.setText(_translate("MainWindow", "EURUSD模型"))
        self.pushButton_4.setText(_translate("MainWindow", "EURAUD模型"))
        self.pushButton_5.setText(_translate("MainWindow", "GBPCHF模型"))
        self.pushButton_6.setText(_translate("MainWindow", "GBPJPY模型"))
        self.pushButton_7.setText(_translate("MainWindow", "显示0级别数据"))
        self.pushButton_8.setText(_translate("MainWindow", "显示1级别数据"))
        self.pushButton_9.setText(_translate("MainWindow", "显示2级别数据"))
        self.pushButton_10.setText(_translate("MainWindow", "显示3级别数据"))
        self.pushButton_11.setText(_translate("MainWindow", "显示4级别数据"))
