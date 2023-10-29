import requests
import datetime
from requests.adapters import HTTPAdapter
import time
from tqdm import tqdm
import pandas as pd
def spyder(starttime,endtime):
    ##爬虫获取页面数据
    url = "https://flash-api.jin10.com/get_flash_list"
    header = {
        "x-app-id": "SO1EJGmNgCtmpcPF",
        "x-version": "1.0.0",
    }
    queryParam = {
        "max_time": starttime,
        "channel": "-8200",
    }

    # 循环爬取并插入数据：结束条件是爬不到数据为止
    data = pd.DataFrame(columns=['id', 'time', 'type', 'pic', 'content', 'title'])

    totalCount = 0
    Data = requests.get(url, queryParam, headers=header).json()['data']
    length = len(Data)
    while (length >0):
        for i in tqdm(range(length)):
            try:
                id = Data[i]['id']
                time = Data[i]['time']
                create_time = datetime.datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
                type = Data[i]['type']
                if type == 0:
                    if len(Data[i]['data']) > 2:
                        pic = Data[i]['data']['pic']
                        content = Data[i]['data']['content']
                        title = Data[i]['data']['title']
                    elif len(Data[i]['data']) == 1:
                        pic = None
                        content = Data[i]['data']['content']
                        title = None
                    else:
                        pic = Data[i]['data']['pic']
                        content = Data[i]['data']['content']
                        title = None
                    newlist=[id, time, type, pic, content, title]
                    data.loc[len(data)]=newlist
                    print(id, time, type, pic, content, title)
                    print(data)
                    print(queryParam['max_time'] + '############')
                    # data.to_csv(r'C:\Users\Computer of phw\tttest/jinshi.csv', header=True)
                    # try:
                    #
                    #     sql = "insert into  jin10_data(id,create_time,type,pic,content,title) values(%s,%s,%s,%s,%s,%s)"
                    #     cursor = conn.cursor()
                    #     cursor.execute(sql, (id, create_time, type, pic, content, title))
                    #     conn.commit()
                    #     cursor.close()
                    # except Exception as e:
                    #     print(e)
                    #     continue
            except Exception as e:
                print(e)
                continue

        totalCount += length

        # 修正下一个查询时间
        queryParam['max_time'] = Data[length - 1]['time']
        # print('next queryParam is', queryParam['max_time'])

        # 再请求一次数据
        try:
            s = requests.Session()
            s.mount('http://', HTTPAdapter(max_retries=3))
            s.mount('https://', HTTPAdapter(max_retries=3))
            Data = requests.get(url, queryParam, timeout=5, headers=header).json()['data']
            length = len(Data)
        except Exception as e:
            print(e)
        if endtime in queryParam['max_time']:
            data.to_csv('news'+str(datetime.datetime.now().day),index=0)
            return data
    print('all ok,totalCount is:', totalCount)
if __name__=='__main__':
    spyder('2021-04-10','2021-04-09')