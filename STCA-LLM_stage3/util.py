# -*- coding:utf-8 -*-
import copy
import os
import sys

import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from get_data import setup_seed

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from itertools import chain

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

from models import device, stcllm
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from einops import rearrange

setup_seed(122)


@torch.no_grad()
def get_val_loss(args, model, Val):
    model.eval()
    loss_function = nn.MSELoss().to(args.device)
    val_loss = []
    
    for seq, labels in Val:
        seq = seq.to(args.device)  # 确保数据在正确的设备上
        labels = labels.to(args.device)  # (batch_size, n_outputs, pred_step_size)
        preds = model(seq)  # 获取模型预测的输出
        
        # 处理 preds 的维度
        preds = preds[:, :, :1]  # 保留最后一维的第一个元素
        preds = preds.mean(dim=-1, keepdim=True)  # 在最后一维上取平均值
        total_loss = 0
        batch_size = seq.shape[0]  # 获取批次大小
        
        for k in range(args.input_size):
            # 确保 preds 和 labels 的维度匹配
            total_loss += loss_function(preds[:, k, :], labels[:, k, :])  # 比较每个输出
        
        total_loss /= batch_size  # 计算平均损失
        val_loss.append(total_loss.item())  # 记录每个批次的损失值

    return np.mean(val_loss)  # 返回平均验证损失



def train(args, Dtr, Val):
    adj  = pd.read_csv(r'D:\pythonproject\STCA-LLM\STCA-LLM_stage3\dataset\DTW_normalized.csv',header=None)
    adj = np.array(adj)
    model = stcllm(args, device, adj).to(device)
    model.load_state_dict(torch.load(r'D:\pythonproject\STCA-LLM\STCA-LLM_stage2\models\gpt.pkl')['model'])
    loss_function = nn.MSELoss().to(device)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.9, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # training
    min_epochs = 2
    best_model = None
    min_val_loss = 5
    for epoch in tqdm(range(args.epochs)):
        train_loss = []
        for (seq, labels) in Dtr:
            seq = seq.to(device)
            labels = labels.to(device)  # (batch_size, n_outputs, pred_step_size)
            preds = model(seq)
            total_loss = 0
            for k in range(args.input_size):
                total_loss = total_loss + loss_function(preds[:, k, :], labels[:, k, :])

            total_loss = total_loss / preds.shape[0]
            # total_loss.requires_grad_(True)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            train_loss.append(total_loss.item())

        scheduler.step()
        # validation
        val_loss = get_val_loss(args, model, Val)
        if epoch + 1 >= min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            #best_model = copy.deepcopy(model)
            #state = {'model': best_model.state_dict()}
            #torch.save(state, r'D:\pythonproject\stcllm\models\gpt.pkl')
            torch.save({'model': model.state_dict()}, r'D:\pythonproject\STCA-LLM\STCA-LLM_stage3\models\gpt.pkl')

        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
        model.train()

    #state = {'model': best_model.state_dict()}
    #torch.save(state, r'D:\pythonproject\stcllm\models\gpt.pkl')
    torch.save({'model': model.state_dict()}, r'D:\pythonproject\STCA-LLM\STCA-LLM_stage3\models\gpt.pkl')


def test(args, Dte, scaler):
    print('loading models...')
    adj  = pd.read_csv(r'D:\pythonproject\STCA-LLM\STCA-LLM_stage3\dataset\DTW_normalized.csv',header=None)
    adj = np.array(adj)
    model = stcllm(args, device, adj).to(device)
    model.load_state_dict(torch.load(r'D:\pythonproject\STCA-LLM\STCA-LLM_stage3\models\gpt.pkl')['model'])
    model.eval()
    print('predicting...')

    ys = [[] for _ in range(args.input_size)]  # Ensure preds list is correctly sized
    preds = [[] for _ in range(args.input_size)]  # Ensure preds list is correctly sized

    for (seq, targets) in tqdm(Dte):
        targets = np.array(targets.data.tolist())  # (batch_size, n_outputs, pred_step_size)

        # Flatten targets and add to ys
        for i in range(args.input_size):
            target = targets[:, i, :]
            target = list(chain.from_iterable(target))
            ys[i].extend(target)

        seq = seq.to(device)
        with torch.no_grad():
            _pred = model(seq)
            _pred = rearrange(_pred, "b m l -> m b l")
            for i in range(_pred.shape[0]):
                pred = _pred[i]
                pred = list(chain.from_iterable(pred.data.tolist()))
                preds[i].extend(pred)

    # Convert to arrays and apply inverse scaling
    ys = np.array(ys).T # Transpose after checking lengths
    preds = np.array(preds).T
    
   
   # Transpose after checking lengths

    ys = scaler.inverse_transform(ys).T
    preds = scaler.inverse_transform(preds).T

    pd.DataFrame(ys.T).to_csv(r'D:\pythonproject\STCA-LLM\STCA-LLM_stage3\results\ys.csv', index=False)
    pd.DataFrame(preds.T).to_csv(r'D:\pythonproject\STCA-LLM\STCA-LLM_stage3\results\preds.csv', index=False) 
    
    # Calculate and print evaluation metrics
    mses, rmses, maes, mapes, r2s = [], [], [], [], []
    for ind, (y, pred) in enumerate(zip(ys, preds), 0):
        print('--------------------------------')
        print('wind turbine:', str(ind+1))
        print('mse:', get_mse(y, pred))
        print('rmse:', get_rmse(y, pred))
        print('mae:', get_mae(y, pred))
        print('r2:', get_r2(y, pred))
        mses.append(get_mse(y, pred))
        rmses.append(get_rmse(y, pred))
        maes.append(get_mae(y, pred))
        r2s.append(get_r2(y, pred))
        print('--------------------------------')
        #plot(y, pred, ind + 1, label='GPT-2')

    # Save results to CSV
    df = pd.DataFrame({
        "mse": mses, 
        "rmse": rmses,
        "mae": maes, 
        "r2": r2s
    })
    df.to_csv(r'D:\pythonproject\STCA-LLM\STCA-LLM_stage3\results\results.csv', index=False)
    plt.show()


def plot(y, pred, ind, label):
    # plot
    plt.plot(y, color='red', label='true value')

    plt.plot(pred, color='blue', label='predicted value')
    plt.title('Wind turbine:' + str(ind) + 'forecasting value')
    plt.grid(True)
    plt.legend(loc='upper center', ncol=6)
    plt.show()








def get_mape(x, y):
    """Calculate MAPE"""
    return np.mean(np.abs((x - y) / x))

def get_r2(y, pred):
    """Calculate R² score"""
    return r2_score(y, pred)

def get_mae(y, pred):
    """Calculate MAE"""
    return mean_absolute_error(y, pred)

def get_mse(y, pred):
    """Calculate MSE"""
    return mean_squared_error(y, pred)

def get_rmse(y, pred):
    """Calculate RMSE"""
    return np.sqrt(mean_squared_error(y, pred))


