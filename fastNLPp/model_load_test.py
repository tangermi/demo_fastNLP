from fastNLP import Tester
from fastNLP.core.metrics import AccuracyMetric

import torch
import os


#读取模型和测试


def ensure_model(model_name, save_dir):
    model_path = os.path.join(save_dir, model_name)
    if model_path.is_file():
        model = torch.load(model_path)
        return model
    else:   #如果模型不存在
        print('模型尚未训练')

def load_model(save_dir):
    return torch.load(save_dir)

def test_model(data_test, model):
    # 使用tester来进行测试
    tester = Tester(data=data_test, model=model, metrics=AccuracyMetric(pred="predict", target="label_seq"),
                    batch_size=4)
    acc = tester.test()
    print(acc)




# if __name__ == '__main__':
#     save_dir = '/apps/data/ai_nlp_testing/model/sogou_corpus_fastNLP'
#     model = ensure_model('new_model.pkl', save_dir=save_dir)

