from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, register_metric
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Reconstruct(Exp_Basic):
    def __init__(self, args):
        super(Exp_Reconstruct, self).__init__(args)

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

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, attributes) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = batch_x.float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, attributes)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, attributes)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, attributes)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, attributes)
                f_dim = -1 if self.args.features == 'MS' else 0

                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

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

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, attributes) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = batch_x.float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, attributes)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, attributes)

                        f_dim = -1 if self.args.features == 'MS' else 0

                        loss = criterion(outputs, batch_x)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, attributes)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, attributes)

                    f_dim = -1 if self.args.features == 'MS' else 0

                    loss = criterion(outputs, batch_x)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if self.args.lradj == 'OneCycleLR':
                    adjust_learning_rate(model_optim, epoch + 1, self.args, scheduler, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
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

        self.model = self._build_model().to(self.device)


        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./pretrain/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        zero_maes = []


        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, attributes) in enumerate(test_loader):
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp =  batch_x.float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, attributes)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, attributes)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, attributes)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, attributes)

                f_dim = -1 if self.args.features == 'MS' else 0

                pred = outputs.detach().cpu().numpy()
                true = batch_x.detach().cpu().numpy()

                
                # 检查 batch_y 第一列中哪些位置为1
                zero_indices = torch.where(batch_y[:, 0] == 0)[0]

                # 如果mask中至少有一个True
                if len(zero_indices) > 0:
                    # 首先移动batch_x回CPU
                    batch_x = batch_x.cpu()
                    outputs = outputs.cpu()
                    batch_x_mark = batch_x_mark.cpu()

                    # 然后使用 mask 来选择相应的行，最后使用 numpy() 转换
                    selected_batch_x = batch_x[zero_indices].numpy()
                    selected_outputs = outputs[zero_indices].numpy()
                    selected_mark = batch_x_mark[zero_indices][0].numpy()
                    
                    # 可以对 selected_batch_x 和 selected_outputs 进行进一步处理
                    # 例如，计算他们的MAE
                    zero_mae = np.abs(selected_outputs - selected_batch_x)
                    zero_mae = np.array(zero_mae)
                    zero_mae_avg = np.mean(zero_mae, axis=-1, keepdims=True)[0]
                    temp_mae = np.concatenate((zero_mae_avg, selected_mark), axis=1)
                    temp_mae = temp_mae[np.newaxis, :]

                    zero_indices_np = zero_indices[0].cpu().numpy()

                    for m in range(1,50):
                        index = i*self.args.batch_size + zero_indices_np - m
                        past_batch_x, past_batch_y, past_batch_x_mark, past_batch_y_mark, past_attributes = test_data[index]

                        past_batch_x = torch.from_numpy(past_batch_x).float().to(self.device).unsqueeze(0)
                        past_batch_y = past_batch_y.float().to(self.device).unsqueeze(0)

                        past_batch_x_mark = torch.from_numpy(past_batch_x_mark).float().to(self.device).unsqueeze(0)
                        past_batch_y_mark = torch.from_numpy(past_batch_y_mark).float().to(self.device).unsqueeze(0)

                        # decoder input
                        past_dec_inp =  past_batch_x.to(self.device).unsqueeze(0)
                        # encoder - decoder
                        if self.args.use_amp:
                            with torch.cuda.amp.autocast():
                                if self.args.output_attention:
                                    outputs = self.model(past_batch_x, past_batch_x_mark, past_dec_inp, past_batch_y_mark, past_attributes)[0]
                                else:
                                    outputs = self.model(past_batch_x, past_batch_x_mark, past_dec_inp, past_batch_y_mark, past_attributes)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(past_batch_x, past_batch_x_mark, past_dec_inp, past_batch_y_mark, past_attributes)[0]

                            else:
                                outputs = self.model(past_batch_x, past_batch_x_mark, past_dec_inp, past_batch_y_mark, past_attributes)

                        # 首先移动batch_x回CPU
                        selected_batch_x = past_batch_x.cpu().numpy()
                        selected_outputs = outputs.cpu().numpy()
                        selected_mark = past_batch_x_mark.cpu().numpy()
                        
                        # 可以对 selected_batch_x 和 selected_outputs 进行进一步处理
                        # 例如，计算他们的MAE
                        zero_mae = np.abs(selected_outputs - selected_batch_x)
                        zero_mae = np.array(zero_mae)
                        zero_mae_avg = np.mean(zero_mae, axis=-1, keepdims=True)
                        temp_past_mae = np.concatenate((zero_mae_avg, selected_mark), axis=2)
                        temp_mae =  np.concatenate((temp_past_mae, temp_mae), axis=1)

                    zero_maes.append(temp_mae)


                preds.append(pred)
                trues.append(true)

            # if i % 20 == 0:
            #     input = batch_x
            #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
            #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axi s=0)
            #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        # arr = np.array(zero_maes)
        zero_maes_np = np.concatenate(zero_maes, axis=0)  # axis=0 是批次大小的轴
        print(zero_maes_np.shape)

        import matplotlib.pyplot as plt

        for i in range(0, 300, 5):

            plt.figure(figsize=(20, 6))

            # 假定 zero_maes_np[0,:,:] 的最后一列是 X 坐标

            x_coords = zero_maes_np[i, :, 2]*24*7 + zero_maes_np[i, :, 3]*24 + zero_maes_np[i, :, 4] + zero_maes_np[i, :, 5]/60

            x_coords = x_coords - min(x_coords)

            # MAE值为最后一列之前的所有值，这里假设我们只关心第一列的MAE值
            y_vals = zero_maes_np[i, :, 0]  # 如果你想用其他或多个列，可以调整这里

            x_to_y_values = {}

            for x, y in zip(x_coords, y_vals):
                # 如果x坐标不存在，则初始化为空列表
                if x not in x_to_y_values:
                    x_to_y_values[x] = []
                    # 将y值添加到对应的x坐标下
                    x_to_y_values[x].append(y)

            # 计算每个x坐标下的y值平均值
            x_max = []
            y_max = []

            for x, ys in x_to_y_values.items():
                x_max.append(x)
                y_max.append(np.max(ys))

            # 将x_means和y_means按照x_means的值排序
            sorted_indices = np.argsort(x_max)
            sorted_x_means = np.array(x_max)[sorted_indices]
            sorted_y_means = np.array(y_max)[sorted_indices]

            # 绘制折线图
            plt.plot(sorted_x_means, sorted_y_means, c='blue', label='Average MAE')
            
            # # 绘制每个独特X坐标与其对应Y值均值的散点图
            # plt.scatter(x_max, y_max, c='blue', label='Average MAE')

            # 指定图形将被保存的文件路径和文件名
            # 例如: 'results/graph_' + str(i) + '.png'
            # 这会在名为 'results' 的文件夹中创建文件，文件名为 'graph_0.png', 'graph_1.png', 依此类推
            # 注意: 请确保事先创建了 'results' 文件夹，或者选择一个已存在的文件夹
            plt.savefig('pic/graph_' + str(i) + '.png', dpi=600)  # 使用 dpi 参数调整保存图像的分辨率

            # 清除当前图形，为下一轮保存做准备
            plt.clf()
       
        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
