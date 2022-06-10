import argparse

import torch

from exp.experiment import Experiment


parser = argparse.ArgumentParser(description='FNet & Autoformer for EEG classification')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='DeepConvLSTM',
                    help='model name, options: [DeepConvLSTM, Transformer, FNet, Autoformer]')

# data loader
parser.add_argument('--data', type=str, required=True, default='PhysioNet32',
                    help='dataset , options: [PhysioNet32, PhysioNe640, PhysioNet32CV, PhysioNe640CV, PhysioNetCS]')
parser.add_argument('--root_path', type=str, default=r'G:\data', help='root path of the data file')
parser.add_argument('--label_path', type=str, default=r'G:\labels', help='label path')
parser.add_argument('--result_path', type=str, default=r'C:\Users\56349\OneDrive\test_results_0604', help='result path')
parser.add_argument('--data_path', type=str, default='', help='data file')
parser.add_argument('--seq_len', type=int, default=32, help='input sequence length')
parser.add_argument('--label_len', type=int, default=256, help='label length')
parser.add_argument('--checkpoints', type=str, default=r'C:\Users\56349\OneDrive\checkpoints', help='checkpoints path')

# model define
parser.add_argument('--c_out', type=int, default=4, help='output size')
parser.add_argument('--d_model', type=int, default=256, help='dimension of temporal model')
parser.add_argument('--d_spatial_model', type=int, default=32, help='dimension of spatial model')
parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--activation', type=str, default='relu', help='activation function')
parser.add_argument('--hidden_chn', type=int, default=128, help='hidden channels in pre-convolutional layers')
parser.add_argument('--kernel_size', type=int, default=5, help='kernel size in pre-convolutional layers')
parser.add_argument('--stride', type=int, default=1, help='stride in pre-convolutional layers')
parser.add_argument('--filter_num', type=int, default=32, help='filter number in feature extractors')
parser.add_argument('--filter_size', type=int, default=10, help='filter size in feature extractors')
parser.add_argument('--fe_stride', type=int, default=2, help='stride in feature extractors')
parser.add_argument('--window_size', type=int, default=20, help='window size')
parser.add_argument('--window_stride', type=int, default=10, help='window size')
parser.add_argument('--n_layers', type=int, default=2, help='num of lstm layers in DeepConvLSTM')
parser.add_argument('--n_hidden', type=int, default=512, help='hidden dimension in LSTM')
parser.add_argument('--pool', type=str, default='class_token',
                    help='pooling operation after transformer, options: [class_token, mean]')
parser.add_argument('--save_model', type=bool, default=False, help='save model graph')

# optimization
parser.add_argument('--itr', type=int, default=5, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=60, help='train epochs')
parser.add_argument('--batch_size', type=int, default=8, help='batch size of train input data')
parser.add_argument('--learning_rate', type=float, default=0.00001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='CrossEntropyLoss', help='loss function')
parser.add_argument('--lr_adjust', type=str, default='type1', help='learning rate adjust')
parser.add_argument('--patience', type=int, default=7, help='early stopping patience')

# gpu
parser.add_argument('--use_gpu', type=bool, default='True', help='use gpu')
parser.add_argument('--gpu', type=str, default='0', help='gpu')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

print('Args in experiment:')
print(args)

if args.save_model:
    exp = Experiment(args, None)
    exp.get_parameter_number()
    # exp.save_model_graph()

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_lr{}_lra{}_sq{}_ll{}_dm{}_nh{}_el{}_df{}_cv{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.learning_rate,
            args.lr_adjust,
            args.seq_len,
            args.label_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_ff,
            ii + 1
        )
        # set experiment
        exp = Experiment(args, ii)
        print('>>>>>>>>>>>>>>>>>>start training : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.train(setting)

        print('>>>>>>>>>>>>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_lr{}_lra{}_sq{}_ll{}_dm{}_nh{}_el{}_df{}_{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.learning_rate,
        args.lr_adjust,
        args.seq_len,
        args.label_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_ff,
        ii
    )
    exp = Experiment(args, ii)
    print('>>>>>>>>>>>>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)

    torch.cuda.empty_cache()
