import pandas as pd
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
import re
from keras.utils import to_categorical
maxlen = 50
def predict(file,rawfile,modelplace):
    pd.set_option('display.max_rows', 2000)
    pd.set_option('display.max_columns', 200)
    #读取数据
    news=file
    config_path = 'chinese_L-12_H-768_A-12\chinese_L-12_H-768_A-12/bert_config.json'
    checkpoint_path = 'chinese_L-12_H-768_A-12\chinese_L-12_H-768_A-12/bert_model.ckpt'
    dict_path = 'chinese_L-12_H-768_A-12\chinese_L-12_H-768_A-12/vocab.txt'


    # 将词表中的词编号转换为字典
    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)


    # 重写tokenizer
    class OurTokenizer(Tokenizer):
        def _tokenize(self, text):
            R = []
            for c in text:
                if c in self._token_dict:
                    R.append(c)
                elif self._is_space(c):
                    R.append('[unused1]')  # 用[unused1]来表示空格类字符
                else:
                    R.append('[UNK]')  # 不在列表的字符用[UNK]表示
            return R


    tokenizer = OurTokenizer(token_dict)
    def seq_padding(X, padding=0):
        L = [len(x) for x in X]
        ML = max(L)
        return np.array([
            np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
        ])
    class data_generator:
        def __init__(self, data, batch_size=32, shuffle=True):
            self.data = data
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.steps = len(self.data) // self.batch_size
            if len(self.data) % self.batch_size != 0:
                self.steps += 1

        def __len__(self):
            return self.steps

        def __iter__(self):
            while True:
                idxs = list(range(len(self.data)))

                if self.shuffle:
                    np.random.shuffle(idxs)

                X1, X2, Y = [], [], []
                for i in idxs:
                    d = self.data[i]
                    text = d[0][:maxlen]
                    x1, x2 = tokenizer.encode(first=text)
                    y = d[1]
                    X1.append(x1)
                    X2.append(x2)
                    Y.append([y])
                    if len(X1) == self.batch_size or i == idxs[-1]:
                        X1 = seq_padding(X1)
                        X2 = seq_padding(X2)
                        Y = seq_padding(Y)
                        yield [X1, X2], Y[:, 0]
                        [X1, X2, Y] = [], [], []

    def acc_top2(y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=2)
    #读取模型r
    custom_objects=get_custom_objects()
    my_object={'acc_top2':acc_top2}
    custom_objects.update(my_object)
    model=keras.models.load_model(modelplace,custom_objects=custom_objects)
    DATA_LIST_TEST = []
    for data_row in news.iloc[0:].itertuples():
        DATA_LIST_TEST.append((data_row.content, to_categorical(0, 5)))
    DATA_LIST_TEST = np.array(DATA_LIST_TEST)
    # print(DATA_LIST_TEST)
    test_new=data_generator(DATA_LIST_TEST, shuffle=False)
    preds_new=model.predict_generator(test_new.__iter__(), steps=len(test_new), verbose=1)
    preds_new = [np.argmax(x) for x in preds_new]
    news['pred_label']=preds_new
    # news_all=pd.merge(news,rawfile,on='content')
    # news=news_all[['content','time','pred_label']]
    news.to_csv(modelplace[0:6]+'_label.csv')
    return news