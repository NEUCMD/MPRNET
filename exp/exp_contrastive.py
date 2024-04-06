from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, register_metric
from utils.metrics import metric
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler 
import os
import time
import warnings
import numpy as np
from sklearn.metrics import classification_report

warnings.filterwarnings('ignore')


class Exp_Contrastive(Exp_Basic):
    def __init__(self, args):
        super(Exp_Contrastive, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def simcse_unsup_loss(self, y_pred, device, temp=0.05):
        """无监督的损失函数
        y_pred (tensor): bert的输出, [batch_size * 2, 768]

        """
        # 得到y_pred对应的label, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
        y_true = torch.arange(y_pred.shape[0], device=device)
        y_true = (y_true - y_true % 2 * 2) + 1
        # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
        sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
        # 将相似度矩阵对角线置为很小的值, 消除自身的影响
        sim = sim - torch.eye(y_pred.shape[0], device=device) * 1e12
        # 相似度矩阵除以温度系数
        sim = sim / temp
        # 计算相似度矩阵与y_true的交叉熵损失
        # 计算交叉熵，每个case都会计算与其他case的相似度得分，得到一个得分向量，目的是使得该得分向量中正样本的得分最高，负样本的得分最低
        loss = F.cross_entropy(sim, y_true)
        return torch.mean(loss)

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        const_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, attributes) in enumerate(vali_loader):
                batch_size = self.args.batch_size

                batch_x = batch_x.float().to(self.device)
                batch_x_duplicated = torch.repeat_interleave(batch_x, repeats=2, dim=0)

                batch_y = batch_y.float().to(self.device)
                batch_y_duplicated = torch.repeat_interleave(batch_y, repeats=2, dim=0)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_x_mark_duplicated = torch.repeat_interleave(batch_x_mark, repeats=2, dim=0)

                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_y_mark_duplicated = torch.repeat_interleave(batch_y_mark, repeats=2, dim=0)

                attributes = attributes.float().to(self.device)
                attributes_duplicated = torch.repeat_interleave(attributes, repeats=2, dim=0)

                # decoder input
                dec_inp = batch_x_duplicated
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs, features = self.model(batch_x_duplicated, batch_x_mark_duplicated, dec_inp, batch_y_mark_duplicated, attributes_duplicated)[0]
                        else:
                            outputs, features = self.model(batch_x_duplicated, batch_x_mark_duplicated, dec_inp, batch_y_mark_duplicated, attributes_duplicated)

                else:
                    if self.args.output_attention:
                        outputs, features = self.model(batch_x_duplicated, batch_x_mark_duplicated, dec_inp, batch_y_mark_duplicated, attributes_duplicated)[0]
                    else:
                        outputs, features = self.model(batch_x_duplicated, batch_x_mark_duplicated, dec_inp, batch_y_mark_duplicated, attributes_duplicated)

                f_dim = -1 if self.args.features == 'MS' else 0

                features = features.detach().cpu()
                pred = outputs.detach().cpu()
                true = batch_y_duplicated.detach().cpu()

                loss = self.simcse_unsup_loss(features, 'cpu')
                const_loss.append(loss.item())

                loss = criterion(pred, true)

                total_loss.append(loss)
        const_loss = np.average(const_loss)      
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss, const_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        self.model = self._build_model().to(self.device)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            const_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, attributes) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_size = self.args.batch_size

                batch_x = batch_x.float().to(self.device)
                batch_x_duplicated = torch.repeat_interleave(batch_x, repeats=2, dim=0)

                batch_y = batch_y.float().to(self.device)
                batch_y_duplicated = torch.repeat_interleave(batch_y, repeats=2, dim=0)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_x_mark_duplicated = torch.repeat_interleave(batch_x_mark, repeats=2, dim=0)

                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_y_mark_duplicated = torch.repeat_interleave(batch_y_mark, repeats=2, dim=0)

                attributes = attributes.float().to(self.device)
                attributes_duplicated = torch.repeat_interleave(attributes, repeats=2, dim=0)

                # decoder input
                dec_inp = batch_x_duplicated

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs, features = self.model(batch_x_duplicated, batch_x_mark_duplicated, dec_inp, batch_y_mark_duplicated, attributes_duplicated)[0]
                        else:
                            outputs, features = self.model(batch_x_duplicated, batch_x_mark_duplicated, dec_inp, batch_y_mark_duplicated, attributes_duplicated)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = torch.sigmoid(outputs)
                        loss = criterion(outputs, batch_y_duplicated)
                        train_loss.append(loss.item())
                        loss = self.simcse_unsup_loss(features, self.device)
                        const_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs, features = self.model(batch_x_duplicated, batch_x_mark_duplicated, dec_inp, batch_y_mark_duplicated, attributes_duplicated)[0]
                    else:
                        outputs, features = self.model(batch_x_duplicated, batch_x_mark_duplicated, dec_inp, batch_y_mark_duplicated, attributes_duplicated)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = torch.sigmoid(outputs)
                    t_loss = criterion(outputs, batch_y_duplicated)
                    train_loss.append(t_loss.item())
                    c_loss = self.simcse_unsup_loss(features, self.device)
                    const_loss.append(c_loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                combined_loss = t_loss + c_loss

                if self.args.use_amp:
                    scaler.scale(combined_loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    combined_loss.backward()
                    model_optim.step()

                if self.args.lradj == 'OneCycleLR':
                    adjust_learning_rate(model_optim, epoch + 1, self.args, scheduler, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            train_const_loss = np.average(const_loss)
            vali_loss, vali_const_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_const_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} & {3:.7f} Vali Loss: {4:.7f} & {5:.7f} Test Loss: {6:.7f} & {7:.7f} ".format(epoch + 1, train_steps, train_loss, train_const_loss, vali_loss, vali_const_loss, test_loss, test_const_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'OneCycleLR':
                adjust_learning_rate(model_optim, epoch + 1, self.args, scheduler)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, attributes) in enumerate(test_loader):
                batch_size = self.args.batch_size

                batch_x = batch_x.float().to(self.device)
                batch_x_duplicated = torch.repeat_interleave(batch_x, repeats=2, dim=0)

                batch_y = batch_y.float().to(self.device)
                batch_y_duplicated = torch.repeat_interleave(batch_y, repeats=2, dim=0)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_x_mark_duplicated = torch.repeat_interleave(batch_x_mark, repeats=2, dim=0)

                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_y_mark_duplicated = torch.repeat_interleave(batch_y_mark, repeats=2, dim=0)

                attributes = attributes.float().to(self.device)
                attributes_duplicated = torch.repeat_interleave(attributes, repeats=2, dim=0)

                # decoder input
                dec_inp = batch_x_duplicated

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs, features = self.model(batch_x_duplicated, batch_x_mark_duplicated, dec_inp, batch_y_mark_duplicated, attributes_duplicated)[0]
                        else:
                            outputs, features = self.model(batch_x_duplicated, batch_x_mark_duplicated, dec_inp, batch_y_mark_duplicated, attributes_duplicated)

                else:
                    if self.args.output_attention:
                        outputs, features = self.model(batch_x_duplicated, batch_x_mark_duplicated, dec_inp, batch_y_mark_duplicated, attributes_duplicated)[0]
                    else:
                        outputs, features = self.model(batch_x_duplicated, batch_x_mark_duplicated, dec_inp, batch_y_mark_duplicated, attributes_duplicated)

                f_dim = -1 if self.args.features == 'MS' else 0
                features = features.detach().cpu().numpy()
                outputs = outputs.detach().cpu().numpy()
                batch_y_duplicated = batch_y_duplicated.detach().cpu().numpy()

                pred = outputs
                true = batch_y_duplicated

                preds.append(pred)
                trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)

        preds = preds.reshape(preds.shape[0] * preds.shape[1], preds.shape[2])
        trues = trues.reshape(trues.shape[0] * trues.shape[1], trues.shape[2])       
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        probs = torch.nn.functional.sigmoid(torch.tensor(preds))

        pred = (np.array(probs) > 0.1).astype(int)

        report = classification_report(trues, pred, digits=5)

        print('report:{}'.format(report))

        np.save(folder_path + 'report.npy', report)

        return
