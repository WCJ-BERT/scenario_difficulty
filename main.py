import os
import torch
import numpy as np
from tqdm import tqdm
from model import Transformer
from util import save_load as sl
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter



def train(model, loss_fn, optimizer, dataloader, use_gpu=False):
    exp_name = 'experiment_1'
    writer = SummaryWriter('logs/' + exp_name)
    model_path = 'checkpoint/' + exp_name
    os.makedirs(model_path)
    epoch_num = 2000
    for epoch in range(0, epoch_num):

        pbar = tqdm(total=len(dataloader), bar_format='{l_bar}{r_bar}', dynamic_ncols=True)
        pbar.set_description(f'Epoch %d' % epoch)
        for step, (batch_x, batch_y) in enumerate(dataloader):
            pred = model(batch_x)
            # print(batch_y.shape)
            batch_y = batch_y.squeeze(-1) #####在此处，必须将pred和batch_y的维度齐平，前代码main中labels使用的np.即为这个功能
            loss = loss_fn(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(**{'loss': loss.detach().cpu().item()})
            pbar.update()

        sl.save_checkpoint(model_path, epoch, model, optimizer)
        pbar.close()
        writer.add_scalar('Loss', loss, epoch)

    writer.close()

def main(gpu_id=None):
    root_path = os.getcwd()
    data = pd.read_csv(root_path + '/' + 'data' + '/' + 'all_tf.csv')
    data_value = data.values

    features = torch.Tensor(data_value[:, 2:8]).float().unsqueeze(dim=-1) ##行全要 列取2:8

    normal_labels = (data_value[:, 8:10])  # 归一化 -1.1 ##行全要 列取1:2
    labels = torch.Tensor(normal_labels).float().unsqueeze(dim=-1)

    # 创建一个 TensorDataset，将特征和标签组合在一起
    dataset = TensorDataset(features, labels)
    # 创建 DataLoader，指定批量大小和是否进行数据随机化
    batch_size = 1024
    shuffle = True
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    model = Transformer(n_head=2)

    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters())

    train(model, loss_fn, optimizer, dataloader, use_gpu=True if gpu_id is not None else False)


if __name__ == '__main__':
    main(gpu_id=None)
    # model = Transformer(n_head=2)
    # ckpt_path = 'checkpoint/experiment_1/ckpt_epoch_999.pth'
    # checkpoint = torch.load(ckpt_path)
    # print(checkpoint)
    # model.load_state_dict(checkpoint['model'])
    # torch.save(model,'model.pkl')