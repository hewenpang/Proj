import jinshishuju
import predict_new
import pandas as pd
fileplace=r'EURAUDmodel.h5'
filename=r'C:\Users\Computer of phw\PycharmProjects\bishe_front\news_pre.csv'
data=pd.read_csv(filename,index_col=0)
data.columns=['content']
predict_new.predict(data,'23',fileplace)
# data=pd.read_csv(r'C:\Users\Computer of phw\PycharmProjects\bishe_front\EURUSD_label.csv',index_col=0)
# print(data['ored'].value_counts())
# ! /usr/bin/env python
# coding=utf-8


# !/usr/bin/env python
# encoding: utf-8

# import os
#
#
# def clear(filepath):
#     files = os.listdir(filepath)
#     for fd in files:
#         cur_path = os.path.join(filepath, fd)
#         if os.path.isdir(cur_path):
#             if fd == "__pycache__":
#                 print("rm -rf {}".format(cur_path))
#                 os.system("rm -rf {}".format(cur_path))
#             else:
#                 clear(cur_path)
#         elif os.path.isfile(cur_path):
#             if ".pyc" in fd:
#                 print("rm -rf {}".format(cur_path))
#                 os.remove(cur_path)
#             elif ".gitignore" in fd:
#                 print("rm -rf {}".format(cur_path))
#                 os.remove(cur_path)
#
#
#
# if __name__ == "__main__":
#     clear("C:\\Users\\Computer of phw")