# -*- coding:utf-8 -*-
import argparse
import torch


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=20, help='training epochs')
    parser.add_argument('--input_size', type=int, default=20, help='input dimension')#有多少列
    parser.add_argument('--seq_len', type=int, default=10, help='seq len')#用于预测的时序长度
    parser.add_argument('--output_size', type=int, default=1, help='output dimension')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')
    parser.add_argument('--out_channels', type=int, default=64, help='out channels')
    parser.add_argument('--num_layers', type=int, default=2, help='num layers')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--add_att', type=bool, default=False, help='add attention?')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--bidirectional', type=bool, default=True, help='')
    parser.add_argument('--step_size', type=int, default=150, help='step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma')
    parser.add_argument('--is_gpt', type=int, default=1)
    parser.add_argument('--gpt_layers', type=int, default=6)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--pretrain', type=int, default=1)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--freeze', type=int, default=1)
    parser.add_argument('--dropout', type=int, default=0.1)
    parser.add_argument('--kernel_size', type=int, default=2)
    

    args = parser.parse_args()

    return args
