from demofront import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication,QMainWindow
import jinshishuju
import re
import pandas as pd
import predict_new
from tqdm import tqdm
import codecs, gc
import numpy as np
from sklearn.model_selection import KFold
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.metrics import top_k_categorical_accuracy
from keras.layers import *
from keras.callbacks import *
from keras.models import Model
import keras.backend as K

from keras_bert import get_custom_objects
import keras
from keras.optimizers import Adam
from keras.utils import to_categorical
maxlen = 50
pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 200)
config_path = 'chinese_L-12_H-768_A-12\chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'chinese_L-12_H-768_A-12\chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'chinese_L-12_H-768_A-12\chinese_L-12_H-768_A-12/vocab.txt'
class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)
        self.news=''
        self.news_pre=''
        self.newstime=''
        self.datarecent=''
    def getinputandstart(self):
        textstart=self.textEdit.toPlainText()
        textend=self.textEdit_2.toPlainText()
        self.news=jinshishuju.spyder(str(textend).replace(' ',''),str(textstart.replace(' ','')))
        self.textBrowser.insertPlainText('爬取结束')
    def preprocessing(self):
        def check_cinese(content):
            zhmodel = re.compile(u'[\u4e00-\u9fa5]')
            match = zhmodel.search(content)
            if match:
                return True
            else:
                return False
        news = self.news
        news = news[['time', 'content']]
        self.newstime=news
        news['content'] = news['content'].apply(lambda x: x.replace('【', ''))
        news['content'] = news['content'].apply(lambda x: x.replace('】', ''))
        news['content'] = news['content'].apply(lambda x: x.replace('<b>', ''))
        news['content'] = news['content'].apply(lambda x: x.replace('</b>', ''))
        news['content'] = news['content'].apply(lambda x: x.replace('</a>', ''))
        news['content'] = news['content'].apply(lambda x: x.replace('<br />', ''))
        news['content'] = news['content'].apply(lambda x: x.replace('</br >', ''))
        news['content'] = news['content'].apply(lambda x: x.replace('<br/>', ''))
        news['content'] = news['content'].apply(lambda x: x.strip())
        news['content'] = news['content'].apply(lambda x: x.replace('<font color class=""h-note-text""><br/>', ''))
        news['content_zh'] = news['content'].apply(lambda x: check_cinese(x))
        news = news[news['content_zh'] == True]
        news = news[['content']]
        self.news_pre=news
        news.to_csv('news_pre.csv')
        self.textBrowser.insertPlainText('\n预处理成功')
    def getmodelandpredict(self):
        self.textBrowser_2.clear()
        fileplace='EURUSDmodel.h5'
        data_pred=predict_new.predict(self.news_pre,self.newstime,fileplace)
        self.datarecent=data_pred
        self.textBrowser_2.insertPlainText("预测结果：\n")
        for indexs in data_pred.index:
            self.textBrowser_2.insertPlainText(str(data_pred.loc[indexs])+'\n')
    def EURAUDmodel(self):
        self.textBrowser_2.clear()
        fileplace = 'EURAUDmodel.h5'
        data_pred =predict_new.predict(self.news_pre,self.newstime,fileplace)
        self.datarecent = data_pred
        self.textBrowser_2.insertPlainText("预测结果：\n")
        for indexs in data_pred.index:
            self.textBrowser_2.insertPlainText(str(data_pred.loc[indexs]) + '\n')
    def GBPCHFmodel(self):
        self.textBrowser_2.clear()
        fileplace = 'GBPCHFmodel.h5'
        data_pred = predict_new.predict(self.news_pre,self.newstime, fileplace)
        self.datarecent = data_pred
        self.textBrowser_2.insertPlainText("预测结果：\n")
        for indexs in data_pred.index:
            self.textBrowser_2.insertPlainText(str(data_pred.loc[indexs]) + '\n')
    def GBPJPYmodel(self):
        self.textBrowser_2.clear()
        fileplace = 'GBPJPYmodel.h5'
        data_pred = predict_new.predict(self.news_pre,self.newstime, fileplace)
        self.datarecent = data_pred
        self.textBrowser_2.insertPlainText("预测结果：\n")
        for indexs in data_pred.index:
            self.textBrowser_2.insertPlainText(str(data_pred.loc[indexs]) + '\n')
    def dispalay0(self):
        self.textBrowser_2.clear()
        data = self.datarecent[self.datarecent['pred_label'] == 0]
        for indexs in data.index:
            self.textBrowser_2.insertPlainText(str(data.loc[indexs]) + '\n')
    def dispalay1(self):
        self.textBrowser_2.clear()
        data=self.datarecent[self.datarecent['pred_label'] == 1]
        if len(data)==0:
            self.textBrowser_2.insertPlainText('NULL')
        else:
            for indexs in data.index:
                self.textBrowser_2.insertPlainText(str(data.loc[indexs]) + '\n')
    def dispalay2(self):
        self.textBrowser_2.clear()
        data = self.datarecent[self.datarecent['pred_label'] == 2]
        if len(data) == 0:
            self.textBrowser_2.insertPlainText('NULL')
        else:
            for indexs in data.index:
                self.textBrowser_2.insertPlainText(str(data.loc[indexs]) + '\n')
    def dispalay3(self):
        self.textBrowser_2.clear()
        data = self.datarecent[self.datarecent['pred_label'] == 3]
        if len(data) == 0:
            self.textBrowser_2.insertPlainText('NULL')
        else:
            for indexs in data.index:
                self.textBrowser_2.insertPlainText(str(data.loc[indexs]) + '\n')
    def dispalay4(self):
        self.textBrowser_2.clear()
        data = self.datarecent[self.datarecent['pred_label'] == 4]
        if len(data) == 0:
            self.textBrowser_2.insertPlainText('NULL')
        else:
            for indexs in data.index:
                self.textBrowser_2.insertPlainText(str(data.loc[indexs]) + '\n')
if __name__=='__main__':
    import sys
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())