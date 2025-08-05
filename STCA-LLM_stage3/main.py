# -*- coding:utf-8 -*-

from args import args_parser
from get_data import nn_seq, setup_seed
from util import test,train

setup_seed(438)


def main():
    args = args_parser()
    Dtr, Val, Dte, scaler = nn_seq(args.input_size, args.seq_len,
                                               args.batch_size, args.output_size)
    print(len(Dtr), len(Val), len(Dte))
    # 如需训练请取消注释
    train(args, Dtr, Val)
    test(args, Dte, scaler)


if __name__ == '__main__':
    main()
