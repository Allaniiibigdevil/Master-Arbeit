import os

import numpy as np
import torch
import onnx

from torch import nn, optim
from thop import profile

from data.data_loader import data_provider
from models import DeepConvLSTM, Transformer, FNet, Autoformer, CNNTransformer, GatedTransformer, S3T,\
    AttendDiscriminate, ProposedModel
from utils.tools import visualization, classification_result, adjust_learning_rate, EarlyStopping


class Experiment(object):
    def __init__(self, args, cv):
        super(Experiment, self).__init__()
        self.args = args
        self.device = torch.device('cuda:{}'.format(self.args.gpu))
        self.model = self._build_model().to(self.device)
        self.cv = cv

    def _build_model(self):
        model_dict = {
            'DeepConvLSTM': DeepConvLSTM,
            'Transformer': Transformer,
            'FNet': FNet,
            'Autoformer': Autoformer,
            'CNNTransformer': CNNTransformer,
            'GatedTransformer': GatedTransformer,
            'S3T': S3T,
            'AttendDiscriminate': AttendDiscriminate,
            'ProposedModel': ProposedModel
        }
        model = model_dict[self.args.model].Model(self.args)
        return model

    def get_parameter_number(self):
        dummy_input = torch.randn(self.args.batch_size, self.args.seq_len, self.args.label_len, device=self.device)
        flops, params = profile(self.model, inputs=(dummy_input,))
        print(flops, params)

    def save_model_graph(self):
        dummy_input = torch.randn(self.args.batch_size, self.args.seq_len, self.args.label_len, device=self.device)
        torch.onnx.export(self.model, dummy_input, self.args.model + '.onnx', verbose=False, input_names='Time Series',
                          output_names='Classification Result')
        model = onnx.load(self.args.model + '.onnx')
        onnx.checker.check_model(model)
        print(onnx.helper.printable_graph(model.graph))

    def _get_data(self, flag):
        dataset, data_loader = data_provider(self.args, flag, self.cv)
        return dataset, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(params=self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_loader, criterion):
        vali_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x).to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                vali_loss.append(loss)

        vali_loss_avg = np.average(vali_loss)
        self.model.train()
        return vali_loss_avg

    def train(self, setting):
        train_set, train_loader = self._get_data('train')
        test_set, test_loader = self._get_data('test')
        train_steps = len(train_loader)

        ckpt_path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

        early_stopping = EarlyStopping(patience=self.args.patience)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        train_loss_global, vali_loss_global = [], []

        for epoch in range(self.args.train_epochs):
            train_loss = []

            self.model.train()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.to(self.device)
                model_optim.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())
                if (i + 1) % 100 == 0:
                    print('\titer: {0:>4}, epoch: {1} | loss: {2:.7f}'.format(i + 1, epoch + 1, loss.item()))
                loss.backward()
                model_optim.step()

            train_loss_avg = np.average(train_loss)
            vali_loss_avg = self.vali(test_loader, criterion)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Test Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss_avg, vali_loss_avg))

            train_loss_global.append(train_loss_avg)
            vali_loss_global.append(vali_loss_avg)

            early_stopping(vali_loss_avg, self.model, ckpt_path)
            if early_stopping.stop:
                print('Early stopping')
                break

            adjust_learning_rate(optimizer=model_optim, epoch=epoch, args=self.args)

        visualization(train_loss_global, vali_loss_global, self.args.result_path, setting,
                      'Training Loss and Validation Loss')

        return self.model

    def test(self, setting, test=0):
        test_set, test_loader = self._get_data('test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints + '\\' + setting,
                                                               'checkpoint.pth')))

        self.model.eval()
        with torch.no_grad():
            ground_truth, prediction = [], []
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x).to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                result = torch.topk(pred, 1, dim=-1)[1][0]
                ground_truth.append(int(true))
                prediction.append(int(result))

            classification_result(ground_truth, prediction, self.args.result_path, setting)

        return


"""
1. 详细的实验setup 。 数据描述，数据处理方法， train-test分割。评估标准。对比模型以及模型细节。 如果实验结果来源于别的文章，标记citation
2. 检查实验setup，保证和别的文章以及数据集发布文章一致。
3. 检查一下数据是否 存在 intra-variability problem 
4. loss曲线存下来
5. 你读过文章，做成列表，发给我
6. 总结一下EEGtask 文章着重处理的问题。 预处理？ spatial-temporal 特征提取的？ train loss?  根据读过的文章，为模型的改动说明理由。
2413 56
"""
