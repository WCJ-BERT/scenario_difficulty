import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from functools import partial
# from dataset import Dataset, transform, collate_fn
from model import Transformer
from util import save_load as sl
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

exp_name = 'experiment_2'
model_path = 'checkpoint/' + exp_name


def paint_score(score, human_readable, pred):
    '''
    score: (n_head, T_dec, T_enc)
    human_readabe: string, length is T_enc
    pred: string, length is T_dec
    '''

    n_head = score.shape[0]
    f = plt.figure(figsize=(8, 8.5))
    ax = f.add_subplot(n_head, 1, 1)

    data = score[1]
    # 将30x30的数组分割成6x6的小块
    split_data = [data[i:i + 5, j:j + 5] for i in range(0, 30, 5) for j in range(0, 30, 5)]

    # 计算每个5x5块的和，得到6x6的结果数组
    result = np.array([block.sum() for block in split_data]).reshape(6, 6)
    ax.imshow(result[:, :] / 9.0, interpolation='nearest', cmap='Blues')

    plt.savefig('./attention.png')


def main():
    # dataset = Dataset(transform=transform, n_datas=10000, seed=None)  # 生成10000个数据，确保字符都出现
    model = Transformer(n_head=2)
    try:
        trained_epoch = sl.find_last_checkpoint(model_path)
        print('load model %d' % (trained_epoch))
    except Exception as e:
        print('no trained model found, {}'.format(e))
        return
    model = sl.load_model(model_path, -1, model)
    model.eval()

    data = pd.read_csv('/home/wcj/PycharmProjects/scenario_difficulty/data/all_tf.csv')
    data_value = data.values

    random_row_index = np.random.choice(data_value.shape[0])
    random_row_index = 200
    data_info = data_value[random_row_index, :]
    data_info[7] = 0.8
    print(data_info)
    x = torch.Tensor(data_info[2:8]).float().unsqueeze(dim=-1).unsqueeze(dim=0)  # 示例特征数据特征数据
    # labels = torch.Tensor(data_value[:, 8:10]).float().unsqueeze(dim=-1)  # 示例标签数据
    actual_data = data_value[:, 8:10][random_row_index]

    actual_data[0] = np.clip(actual_data[0], -1.0, 1.0) * (19 - 7) / 2 + (19 + 7) / 2
    actual_data[1] = np.clip(actual_data[1], -1.0, 1.0) * (0.25 - (-1)) / 2 + (0.25 + (-1)) / 2

    pred = model(x)
    predicted_data = [0,0]
    predicted_data[0] = float(pred[0][0])
    predicted_data[1] = float(pred[0][1])
    predicted_data[0] = np.clip(predicted_data[0], -1.0, 1.0) * (19 - 7) / 2 + (19 + 7) / 2
    predicted_data[1] = np.clip(predicted_data[1], -1.0, 1.0) * (0.25 - (-1)) / 2 + (0.25 + (-1)) / 2

    print("预测数据={}，真值={}".format(predicted_data,actual_data ))
    print("偏差={}".format(predicted_data-actual_data))
    # dec_scores = model.encoder.scores_for_paint
    # paint_score(dec_scores[0], x, pred)  # [0]是去batch中的第0个


if __name__ == '__main__':
    main()
