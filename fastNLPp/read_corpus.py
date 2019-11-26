# -*- coding:utf-8 -*-
# import pandas as pd
import os


# 对搜狗语料包读取
def read_data(data_path):
    # df_train = pd.DataFrame(columns=('category','text'))  # pandas的dataframe数据形式，可以转换为.csv文件方便保存
    # df_test = pd.DataFrame(columns=('category','text'))

    data_train = {'label':[], 'text':[]}   # 字典的数据形式
    data_test = {'label':[], 'text':[]}

    categories = os.listdir(data_path)
    for category in categories:
        category_path = os.path.join(data_path, category)
        if os.path.isdir(category_path):  # 确认该路径是文件夹 (排除掉readme)
            print(f'正在读取类别：{category}')
            text_files = os.listdir(category_path)
            n = len(text_files)
            train_set_shape = n * 0.8  # 80%的数据 用作训练模型
            test_set_shape = n * 0.2
            for i, text_file in enumerate(text_files):
                text_path = os.path.join(category_path, text_file)
                text = open(text_path, encoding='utf-8').read()
                # if i < train_set_shape:
                #     df_train = df_train.append({'category':int(category),'text':text}, ignore_index=True)
                # else:
                #     df_test = df_test.append({'category':int(category),'text':text}, ignore_index=True)
                if i < train_set_shape:
                    data_train['label'].append(int(category))
                    data_train['text'].append(text)
                else:
                    data_test['label'].append(int(category))
                    data_test['text'].append(text)

    return data_train, data_test





