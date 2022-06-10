import os

import onnx
import torch
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

from torch import nn
from sklearn.metrics import confusion_matrix, classification_report


def visualization(train_loss, vali_loss, result_path, setting, title: str, show=False, save=True):
    fig = go.Figure()
    trace1 = go.Scatter(y=train_loss, name='train_loss')
    trace2 = go.Scatter(y=vali_loss, name='vali_loss')
    fig.add_traces([trace1, trace2])
    fig.update_layout(
        title={'text': title, 'font': {'size': 30}, 'x': 0.5},
        xaxis_title={'text': 'Epoch', 'font': {'size': 17}},
        yaxis_title={'text': 'Loss', 'font': {'size': 17}}
    )
    if show:
        fig.show()
    if save:
        folder_path = result_path + '/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        pio.write_image(fig, folder_path + 'loss.pdf', format='pdf')
    return


def classification_result(y_true, y_pred, result_path, setting, verbose=False, show=False, save=True):
    pred_label = true_label = target_names = ['not move', 'left hand', 'right hand', 'feet']
    confusion_mat = confusion_matrix(y_true, y_pred)
    # classification_rep = classification_report(y_true, y_pred, target_names=target_names)
    classification_rep = classification_report(y_true, y_pred)
    if verbose:
        print(confusion_mat)
        print(classification_rep)

    '''
    fig = px.imshow(confusion_mat, x=pred_label, y=true_label, color_continuous_scale='PuBu', text_auto=True,
                    labels={'x': "Predict Label", 'y': "True Label"})
    '''
    fig = px.imshow(confusion_mat, color_continuous_scale='PuBu', text_auto=True,
                    labels={'x': "Predict Label", 'y': "True Label"})
    fig.update_layout(
        height=500,
        width=500,
        xaxis={'side': 'top'}
    )
    if show:
        fig.show()
    if save:
        folder_path = result_path + '/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        with open(folder_path + 'result.txt', 'w') as f:
            f.write(classification_rep)
        pio.write_image(fig, folder_path + 'result.pdf', format='pdf')
    return


def adjust_learning_rate(optimizer, epoch, args):
    if args.lr_adjust == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.1 ** (epoch // 10))}
    elif args.lr_adjust == 'type2':
        lr_adjust = {
            5: 1e-5, 10: 1e-5, 15: 1e-5, 20: 1e-6,
            25: 1e-6, 35: 1e-6, 45: 1e-7
        }
    else:
        print('Learning rate will not be updated.')
        return
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.stop = False
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_score = score
            torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
            self.counter = 0


if __name__ == '__main__':
    x = [0, 1, 2]
    y = [2, 1, 0]
    visualization(x, y, 'x and y', 'Setting', 'Title')
