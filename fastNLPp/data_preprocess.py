# -*- coding:utf-8 -*-
from fastNLP import DataSet, Instance, Vocabulary
from pyltpp.pyltp_segment import PyltpSegment


# 数据预处理， 以匹配fastNLP的trainner


class Preprocess:
    def __init__(self):
        self.segmentor = PyltpSegment()

    # 分词
    def segment(self, instance):
        return self.segmentor.cut(instance['text'], part_of_speech=False)

    # padding
    # 统计分词后的长度，得到最大长度，以此来添加'0',并且作为后面网络maxfeature的参数
    def padding_words(self, data, max_sentence_length):
        for i in range(len(data)):
            if data[i]['description_seq_len'] <= max_sentence_length:
                padding = [0] * (max_sentence_length - data[i]['description_seq_len'])
                data[i]['description_words'] += padding
            else:
                pass
        return data

    def data_preprocess(self, data_train, data_test):
        # 读取数据
        data_train = DataSet(data_train)
        data_test = DataSet(data_test)

        # 分词
        print('进入分词阶段...')
          # 使用pyltp预设模型

        print('开始切分训练数据')
        data_train.apply(self.segment, new_field_name='description_words')
        print('开始切分测试数据')
        data_test.apply(self.segment, new_field_name='description_words')

        print('计算分词长度...')
        data_train.apply(lambda x: len(x['description_words']), new_field_name='description_seq_len')
        data_test.apply(lambda x: len(x['description_words']), new_field_name='description_seq_len')

        max_seq_len_train = 0
        max_seq_len_test = 0
        for i in range(len(data_train)):
            if (data_train[i]['description_seq_len'] > max_seq_len_train):
                max_seq_len_train = data_train[i]['description_seq_len']
            else:
                pass
        for i in range(len(data_test)):
            if (data_test[i]['description_seq_len'] > max_seq_len_test):
                max_seq_len_test = data_test[i]['description_seq_len']
            else:
                pass
        max_sentence_length = max_seq_len_train
        if (max_seq_len_test > max_sentence_length):
            max_sentence_length = max_seq_len_test
        print('max_sentence_length:', max_sentence_length)

        # 根据训练集来建立词典
        print('开始建立词典...')
        vocab = Vocabulary(min_freq=2)
        data_train.apply(lambda x: [vocab.add(word) for word in x['description_words']])
        vocab.build_vocab()
        data_train.apply(lambda x: [vocab.to_index(word) for word in x['description_words']],
                         new_field_name='description_words')
        data_test.apply(lambda x: [vocab.to_index(word) for word in x['description_words']],
                        new_field_name='description_words')

        # padding
        print('进入padding阶段')

        data_train = self.padding_words(data_train, max_sentence_length)
        data_test = self.padding_words(data_test, max_sentence_length)
        data_train.apply(lambda x: len(x['description_words']), new_field_name='description_seq_len')
        data_test.apply(lambda x: len(x['description_words']), new_field_name='description_seq_len')

        # 重命名dataset的fields, 并标注出数据集中的输入和输出,确定input和target.
        data_train.rename_field("description_words", "description_word_seq")
        data_train.rename_field("label", "label_seq")
        data_test.rename_field("description_words", "description_word_seq")
        data_test.rename_field("label", "label_seq")

        data_train.set_input("description_word_seq")
        data_test.set_input("description_word_seq")
        data_train.set_target("label_seq")
        data_test.set_target("label_seq")
        print("dataset processed successfully!")

        return data_train, data_test, vocab, max_sentence_length

