# -*- coding:utf-8 -*-
from read_corpus import read_data
from data_preprocess import Preprocess
from model_train_save import train_and_save_model
from model_load_test import load_model, test_model






if __name__ == '__main__':
    data_path = r'/apps/data/ai_nlp_testing/raw/training_data_for_classification/sogou_text_classification_corpus_mini'
    # 读取搜狗语料
    data_train, data_test = read_data(data_path)
    preprocess = Preprocess()
    # 对语料进行处理，以匹配fastNLP的输入结构
    data_train, data_test, vocab, max_sentence_length = preprocess.data_preprocess(data_train, data_test)
    # 模型存储路径
    save_dir = '/apps/data/ai_nlp_testing/model/sogou_corpus_fastNLP'
    # 设置超参
    word_embedding_dimension = 300
    num_classes = 4
    # 训练模型
    train_and_save_model(data_train, data_test,vocab, max_sentence_length, save_dir=save_dir)
    # 读取模型
    load_model(save_dir)
    # 测试模型
    test_model(data_test, model)


