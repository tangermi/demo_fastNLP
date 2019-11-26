# -*- coding:utf-8 -*-
from fastNLP import Trainer
# from copy import deepcopy
from fastNLP.core.losses import CrossEntropyLoss
from fastNLP.core.optimizer import Adam
from fastNLP.core.utils import _save_model
from DPCNN import *

import os


# 训练模型
def train_and_save_model(data_train, data_test, vocab, max_sentence_length, save_dir):
    # 确认torch版本是否能与fastnlp兼容
    print(torch.__version__)

    # 读取神经网络
    model = DPCNN(max_features=len(vocab), word_embedding_dimension=word_embedding_dimension,
                  max_sentence_length=max_sentence_length, num_classes=num_classes)

    # 定义 loss 和 metric
    loss = CrossEntropyLoss(pred="output", target="label_seq")
    metric = AccuracyMetric(pred="predict", target="label_seq")

    # train model with train_data,and val model witst_data
    # embedding=300 gaussian init，weight_decay=0.0001, lr=0.001，epoch=5
    trainer = Trainer(model=model, train_data=data_train, dev_data=data_test, loss=loss, metrics=metric, save_path='CD',
                      batch_size=64, n_epochs=5, optimizer=Adam(lr=0.001, weight_decay=0.0001))
    trainer.train()
    # 存储模型
    _save_model(model, model_name='new_model.pkl', save_dir=save_dir)




# if __name__ == '__main__':
#     # 设置超参
#     word_embedding_dimension = 300
#     num_classes = 4
#
#     # 模型所在路径
#     save_dir = '/apps/data/ai_nlp_testing/model/sogou_corpus_fastNLP'
#
#     # 确认torch版本是否能与fastnlp兼容
#     print(torch.__version__)
#
#     # 读取神经网络
#     model = DPCNN(max_features=len(vocab), word_embedding_dimension=word_embedding_dimension, max_sentence_length = max_sentence_length, num_classes=num_classes)
#
#     #定义 loss 和 metric
#     loss = CrossEntropyLoss(pred="output", target="label_seq")
#     metric = AccuracyMetric(pred="predict", target="label_seq")
#
#     # train model with train_data,and val model witst_data
#     # embedding=300 gaussian init，weight_decay=0.0001, lr=0.001，epoch=5
#     trainer=Trainer(model=model, train_data=data_train, dev_data=data_test, loss=loss,metrics=metric, save_path='CD', batch_size=64, n_epochs=5, optimizer=Adam(lr=0.001, weight_decay=0.0001))
#     trainer.train()
#
#     # 保存模型
#     _save_model(model, model_name='new_model.pkl', save_dir=save_dir)


