import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.tools import data_StandardScaler
from sklearn.model_selection import train_test_split
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
from tqdm import tqdm
import warnings

import utils.global_var

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_studWg(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):

        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        print(cols)
        cols.remove(self.target);
        cols.remove('enqueuedTime');
        cols.remove('approximateTimes');
        cols.remove('line');
        cols.remove('deviceName');
        cols.remove('studId')
        df_raw = df_raw[['deviceName'] + ['approximateTimes'] + cols + [self.target]]


        df_raw['error'] =df_raw['error'].astype(str)
        temp_label = df_raw['error']
        temp_label = pd.Series(temp_label, dtype="category")
        class_names = temp_label.cat.categories
        utils.global_var.set_value('class_names', class_names)

        df_raw = df_raw.drop(columns = 'error')
        df_raw = df_raw.drop(columns='carbodyID')

        num_flag = 0

        df_raw = df_raw.groupby('deviceName')

        for gun_no in df_raw.size().index:
            if df_raw.get_group(gun_no).shape[0] >= 10000:

                num_train = int(len(df_raw.get_group(gun_no)) * 0.6)
                num_test = int(len(df_raw.get_group(gun_no)) * 0.2)
                num_vali = len(df_raw.get_group(gun_no)) - num_train - num_test
                border1s = [0, num_train - self.seq_len, len(df_raw.get_group(gun_no)) - num_test - self.seq_len]
                border2s = [num_train, num_train + num_vali, len(df_raw.get_group(gun_no))]
                border1 = border1s[self.set_type]
                border2 = border2s[self.set_type]

                if self.features == 'M' or self.features == 'MS':
                    cols_data = df_raw.get_group(gun_no).columns[2:-1]
                    df_data = df_raw.get_group(gun_no)[cols_data]
                elif self.features == 'S':
                    df_data = df_raw.get_group(gun_no)[[self.target]]

                if self.scale:
                    train_data = df_data[border1s[0]:border2s[0]]
                    self.scaler.fit(train_data.values)
                    data = self.scaler.transform(df_data.values)
                else:
                    data = df_data.values

                df_stamp = df_raw.get_group(gun_no)[['approximateTimes']].iloc[border1:border2]

                df_stamp['approximateTimes'] = pd.to_datetime(df_stamp.approximateTimes)
                if self.timeenc == 0:
                    df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                    df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                    df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                    df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                    data_stamp = df_stamp.drop(['approximateTimes'], 1).values
                elif self.timeenc == 1:
                    data_stamp = time_features(pd.to_datetime(df_stamp['approximateTimes'].values), freq=self.freq)
                    data_stamp = data_stamp.transpose(1, 0)

                temp_data_x = data[border1:border2]
                temp_data_y = df_raw.get_group(gun_no)[border1:border2][self.target]
                temp_data_stamp = data_stamp
                temp_gun_no = re.findall("\d+", gun_no) 
                temp_gun_no = int("".join(temp_gun_no))
                temp_attributes = pd.DataFrame([temp_gun_no])
                temp_attributes = pd.concat([temp_attributes]*(temp_data_x.shape[0]), ignore_index=True)

                targets = []

                ##TODO 取出故障类别
                for i in range(0, len(temp_data_y) - 1):

                    label_str = str(temp_data_y.iloc[i]).split(',')
                    multi_cls_lab = np.zeros((len(utils.global_var.get_value('class_names'))), np.float32)

                    if '0' in label_str and len(label_str) == 1:
                        multi_cls_lab[0] = 1.0
                    else:
                        for num in label_str:
                            multi_cls_lab[int(num)] = 1.0
                        multi_cls_lab[0] = 0.0

                    label = multi_cls_lab
                    targets.append(label)

                targets = pd.DataFrame(targets)

                if num_flag == 0:
                    self.data_x = temp_data_x
                    self.data_y = targets
                    self.data_stamp = temp_data_stamp
                    self.data_attributes = temp_attributes

                else:
                    self.data_x = np.concatenate((self.data_x, temp_data_x), axis=0)
                    self.data_y = np.concatenate((self.data_y, targets), axis=0)
                    self.data_stamp = np.concatenate((self.data_stamp, temp_data_stamp), axis=0)               
                    self.data_attributes = np.concatenate((self.data_attributes, temp_attributes), axis=0)

                num_flag = num_flag + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]

        #该部分好像有点问题
        # seq_x = seq_x[:,1:]
        seq_y = self.data_y[s_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[s_end]

        attributes =  self.data_attributes[index]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, attributes

    ##没搞清楚为什么缺了一部分
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1 - 100

class Dataset_studWg_v2(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):

        self.scaler = data_StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        print(cols)
        # cols.remove('Unnamed: 0');
        cols.remove(self.target);
        cols.remove('enqueuedTime');
        cols.remove('approximateTimes');
        # cols.remove('timeDifference');
        cols.remove('line');
        cols.remove('deviceName');
        cols.remove('studId')
        # df_raw = df_raw[['deviceName'] + ['timeDifference'] + ['approximateTimes'] + cols + [self.target]]
        df_raw = df_raw[['deviceName'] + ['approximateTimes'] + cols + [self.target]]

        df_raw['error'] =df_raw['error'].astype(str)
        temp_label = df_raw['error']
        temp_label = pd.Series(temp_label, dtype="category")
        class_names = temp_label.cat.categories
        utils.global_var.set_value('class_names', class_names)

        df_raw = df_raw.drop(columns = 'error')
        df_raw = df_raw.drop(columns='carbodyID')

        num_flag = 0

        df_raw = df_raw.groupby('deviceName')

        for gun_no in df_raw.size().index:
            if df_raw.get_group(gun_no).shape[0] >= 10000:

                temp_gun_no = re.findall("\d+", gun_no) 
                temp_gun_no = int("".join(temp_gun_no))
                
                train_dataset, train_labels, train_attributes = self.create_dataset(df_raw.get_group(gun_no), temp_gun_no, seq_len=self.seq_len,interval= int(self.seq_len))                            
                X_train, X_test, y_train, y_test, attr_train, attr_test = train_test_split(train_dataset, train_labels, train_attributes, test_size=0.2, random_state=17)
                X_train, X_val, y_train, y_val, attr_train, attr_val = train_test_split(X_train, y_train, attr_train, test_size=0.4, random_state=17)

                if num_flag == 0:
                    self.train_x = X_train
                    self.train_y = y_train
                    train_attr = attr_train
                    self.val_x = X_val
                    self.val_y = y_val
                    val_attr = attr_val
                    self.test_x = X_test
                    self.test_y = y_test
                    test_attr = attr_test

                else:
                    self.train_x = np.concatenate((self.train_x, X_train), axis=0)
                    self.train_y = np.concatenate((self.train_y, y_train), axis=0)
                    train_attr = np.concatenate((train_attr, attr_train), axis=0)
                    self.val_x = np.concatenate((self.val_x, X_val), axis=0)
                    self.val_y = np.concatenate((self.val_y, y_val), axis=0)
                    val_attr = np.concatenate((val_attr, attr_val), axis=0)
                    self.test_x = np.concatenate((self.test_x, X_test), axis=0)
                    self.test_y = np.concatenate((self.test_y, y_test), axis=0)
                    test_attr = np.concatenate((test_attr, attr_test), axis=0)

                num_flag += 1

        train_stamp =self.train_x[:, :, :6]
        val_stamp= self.val_x[:, :, :6]
        test_stamp = self.test_x[:, :, :6]

        self.train_x = np.array(self.train_x[:, :, 6:], dtype=np.float64)

        if self.scale:
            self.scaler.fit(self.train_x)
            self.train_x = self.scaler.transform(self.train_x)
            self.val_x = self.scaler.transform(np.array(self.val_x[:, :, 6:], dtype=np.float64))
            self.test_x = self.scaler.transform(np.array(self.test_x[:, :, 6:], dtype=np.float64))
            ##TODO 存在空值 需要删除
            # self.val_x = np.nan_to_num(self.val_x)
            # self.test_x = np.nan_to_num(self.test_x)
        else:
            self.train_x = np.array(self.train_x[:, :, 6:], dtype=np.float64)
            self.val_x = np.array(self.val_x[:, :, 6:], dtype=np.float64)
            self.test_x = np.array(self.test_x[:, :, 6:], dtype=np.float64)

        if self.set_type == 0:
            self.data_x = self.train_x
            self.data_y = self.train_y
            self.data_stamp = train_stamp
            self.data_attributes = train_attr
        elif self.set_type == 1:
            self.data_x = self.val_x
            self.data_y = self.val_y
            self.data_stamp = val_stamp
            self.data_attributes = val_attr
        elif self.set_type == 2:
            self.data_x = self.test_x
            self.data_y = self.test_y
            self.data_stamp = test_stamp
            self.data_attributes = test_attr
        

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[index]
        seq_x_mark = self.data_stamp[index]
        attributes = self.data_attributes[index]

        trues = torch.Tensor(np.array(self.data_y[index]))
        return seq_x, trues, seq_x_mark, seq_x_mark, attributes

    ##没搞清楚为什么缺了一部分
    def __len__(self):
        return len(self.data_x) - 10
    
    ##新数据集构建
    def create_dataset(self, data, gun_no, seq_len=600, interval=60):

        features = []
        targets = []
        attributes = []

        df_stamp = data['approximateTimes']
        data_stamp = pd.DataFrame()
        df_stamp = pd.to_datetime(df_stamp)
        if self.timeenc == 0:
            data_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            data_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            data_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            data_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = data_stamp.values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp.values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        data_stamp = pd.DataFrame(data_stamp)
        
        data = data.iloc[:, 2:]
        data = pd.concat([data_stamp, data.reset_index(drop=True)], axis=1)

        for i in range(0, len(data) - seq_len, interval):
            # 取大范围故障
            # temp_list = data.iloc[i:i + seq_len, -1]
            # temp_label = ",".join([str(i) for i in temp_list if i is not None])
            # label_str = np.unique(temp_label.split(','))
            # 取最后一点故障
            temp_label = data.iloc[i + seq_len, -1]
            # temp_label = ",".join([str(i) for i in temp_list if i is not None])
            label_str = np.unique(str(temp_label).split(','))

            temp_data = data.iloc[i:i + seq_len]
            temp_attribute = gun_no
            

            ##TODO 取出故障类别
            multi_cls_lab = np.zeros((len(utils.global_var.get_value('class_names'))), np.float32)

            if '0' in label_str and len(label_str) == 1:
                multi_cls_lab[0] = 1.0
            else:
                for num in label_str:
                    multi_cls_lab[int(num)] = 1.0
                multi_cls_lab[0] = 0.0

            label = multi_cls_lab

            temp_data = temp_data.iloc[:, :-1]

            features.append(temp_data)
            targets.append(label)
            attributes.append(temp_attribute)

        return np.array(features), np.array(targets), np.array(attributes)
    
    def label_split(self, string):

            ss=string.split(',')

            return ss

class Dataset_studWg_v3(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):

        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        print(cols)
        cols.remove(self.target);
        cols.remove('enqueuedTime');
        cols.remove('approximateTimes');
        cols.remove('line');
        cols.remove('deviceName');
        cols.remove('studId')
        df_raw = df_raw[['deviceName'] + ['approximateTimes'] + cols + [self.target]]


        df_raw['error'] =df_raw['error'].astype(str)
        temp_label = df_raw['error']
        temp_label = pd.Series(temp_label, dtype="category")
        class_names = temp_label.cat.categories
        utils.global_var.set_value('class_names', class_names)

        df_raw = df_raw.drop(columns = 'error')
        df_raw = df_raw.drop(columns='carbodyID')

        num_flag = 0

        df_raw = df_raw.groupby('deviceName')

        for gun_no in df_raw.size().index:
            if df_raw.get_group(gun_no).shape[0] >= 10000:

                num_train = int(len(df_raw.get_group(gun_no)) * 0.6)
                num_test = int(len(df_raw.get_group(gun_no)) * 0.3)
                num_vali = len(df_raw.get_group(gun_no)) - num_train - num_test
                border1s = [0, num_train - self.seq_len, len(df_raw.get_group(gun_no)) - num_test - self.seq_len]
                # border1s = [0, num_train - self.seq_len, num_train - self.seq_len]
                border2s = [num_train, num_train + num_vali, len(df_raw.get_group(gun_no))]
                # border2s = [num_train, len(df_raw.get_group(gun_no)), len(df_raw.get_group(gun_no))]
                border1 = border1s[self.set_type]
                border2 = border2s[self.set_type]

                if self.features == 'M' or self.features == 'MS':
                    df_data = df_raw.get_group(gun_no)
                elif self.features == 'S':
                    df_data = df_raw.get_group(gun_no)[[self.target]]

                temp_gun_no = re.findall("\d+", gun_no)
                temp_gun_no = int("".join(temp_gun_no))


                df_stamp = df_data['approximateTimes'].iloc[border1:border2]
                data_stamp = pd.DataFrame()
                df_stamp = pd.to_datetime(df_stamp)
                if self.timeenc == 0:
                    data_stamp['month'] = df_stamp.dt.month
                    data_stamp['day'] = df_stamp.dt.day
                    data_stamp['weekday'] = df_stamp.dt.weekday
                    data_stamp['hour'] = df_stamp.dt.hour
                    data_stamp['minute'] = df_stamp.dt.minute
                    data_stamp['second'] = df_stamp.dt.second

                    data_stamp = data_stamp.values
                elif self.timeenc == 1:
                    data_stamp = time_features(pd.to_datetime(df_stamp.values), freq=self.freq)
                    data_stamp = data_stamp.transpose(1, 0)

                data_stamp = pd.DataFrame(data_stamp)
                data_stamp = data_stamp.reset_index(drop=True)
                df_label = pd.DataFrame(df_data.iloc[border1:border2, -1])
                df_label = df_label.reset_index(drop=True)
                df_data = df_data.iloc[:, 2:-1]

                if self.scale:
                    train_data = df_data.iloc[border1s[0]:border2s[0]]
                    self.scaler.fit(train_data)
                    data = self.scaler.transform(df_data.iloc[border1:border2])
                else:
                    data = df_data.iloc[border1:border2]
                    data = data.reset_index(drop = True)

                data = pd.DataFrame(data)

                data = pd.concat([data_stamp, data, df_label], axis=1)

                temp_data_x, temp_data_stamp, temp_data_y, temp_attributes = self.create_dataset(data, temp_gun_no, seq_len=self.seq_len,interval=int(self.seq_len))

                if num_flag == 0:
                    self.data_x = temp_data_x
                    self.data_y = temp_data_y
                    self.data_stamp = temp_data_stamp
                    self.data_attributes = temp_attributes

                else:
                    self.data_x = np.concatenate((self.data_x, temp_data_x), axis=0)
                    self.data_y = np.concatenate((self.data_y, temp_data_y), axis=0)
                    self.data_stamp = np.concatenate((self.data_stamp, temp_data_stamp), axis=0)               
                    self.data_attributes = np.concatenate((self.data_attributes, temp_attributes), axis=0)

                num_flag = num_flag + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[index]
        seq_x_mark = self.data_stamp[index]
        attributes = self.data_attributes[index]

        trues = torch.Tensor(np.array(self.data_y[index]))
        return seq_x, trues, seq_x_mark, seq_x_mark, attributes

    ##没搞清楚为什么缺了一部分
    def __len__(self):
        return len(self.data_x) - 10

    ##新数据集构建
    def create_dataset(self, data, gun_no, seq_len=600, interval=60):

        features = []
        stamp = []
        targets = []
        attributes = []
        
        for i in range(0, len(data) - seq_len, interval):
            # 取大范围故障
            # temp_list = data.iloc[i:i + seq_len, -1]
            # temp_label = ",".join([str(i) for i in temp_list if i is not None])
            # label_str = np.unique(temp_label.split(','))
            # 取最后一点故障
            temp_label = data.iloc[i + seq_len, -1]
            # temp_label = ",".join([str(i) for i in temp_label if i is not None])
            label_str = np.unique(str(temp_label).split(','))
            # label_str = str(data.iloc[i + seq_len, -1])

            temp_data = data.iloc[i:i + seq_len, 6:-1]
            temp_stamp = data.iloc[i:i + seq_len, :6]
            temp_attribute = gun_no


            ##TODO 取出故障类别
            multi_cls_lab = np.zeros((len(utils.global_var.get_value('class_names'))), np.float32)

            if '0' in label_str and len(label_str) == 1:
                multi_cls_lab[0] = 1.0
            else:
                for num in label_str:
                    multi_cls_lab[int(num)] = 1.0
                multi_cls_lab[0] = 0.0

            label = multi_cls_lab

            features.append(temp_data)
            stamp.append(temp_stamp)
            targets.append(label)
            attributes.append(temp_attribute)

        return np.array(features), np.array(stamp), np.array(targets), np.array(attributes)

    def label_split(self, string):

            ss=string.split(',')

            return ss

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_cmapss(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):

        self.scaler = data_StandardScaler()
        df_train = pd.read_csv(os.path.join(self.root_path, 'C-MPASS', 'train' + self.data_path))       
        df_test = pd.read_csv(os.path.join(self.root_path, 'C-MPASS', 'test' + self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_train.columns)
        print(cols)
        cols.remove(self.target);
        df_train = df_train[cols + [self.target]]
        df_test = df_test[cols + [self.target]]

        num_flag = 0
        df_train = df_train.groupby('unit')

        for unit in df_train.size().index:
            if(df_train.get_group(unit).shape[0] >= self.seq_len):
            
                X_train, y_train, attr_train = self.create_dataset(df_train.get_group(unit), unit, seq_len=self.seq_len,interval= int(1))   

                if num_flag == 0:
                    self.train_x = X_train
                    self.train_y = y_train
                    train_attr = attr_train

                else:
                    self.train_x = np.concatenate((self.train_x, X_train), axis=0)
                    self.train_y = np.concatenate((self.train_y, y_train), axis=0)
                    train_attr = np.concatenate((train_attr, attr_train), axis=0)

                num_flag += 1

        num_flag = 0
        df_test = df_test.groupby('unit')

        for unit in df_test.size().index:

            if(df_test.get_group(unit).shape[0] >= self.seq_len + 1):
            
                # X_val, y_val, attr_val = self.create_dataset(df_test.get_group(unit), unit, seq_len=self.seq_len,interval= int(self.seq_len))      
                X_val, y_val, attr_val = self.create_test_dataset(df_test.get_group(unit), unit, seq_len=self.seq_len,interval= int(1))                          

                if num_flag == 0:
                    self.val_x = X_val
                    self.val_y = y_val
                    val_attr = attr_val
                    self.test_x = X_val
                    self.test_y = y_val
                    test_attr = attr_val

                elif self.val_x.shape[0] == 0:
                    self.val_x = X_val
                    self.val_y = y_val
                    val_attr = attr_val
                    self.test_x = X_val
                    self.test_y = y_val
                    test_attr = attr_val

                else:             
                    self.val_x = np.concatenate((self.val_x, X_val), axis=0)
                    self.val_y = np.concatenate((self.val_y, y_val), axis=0)
                    val_attr = np.concatenate((val_attr, attr_val), axis=0)
                    self.test_x = np.concatenate((self.test_x, X_val), axis=0)
                    self.test_y = np.concatenate((self.test_y, y_val), axis=0)
                    test_attr = np.concatenate((test_attr, attr_val), axis=0)

                num_flag += 1

        train_stamp =self.train_x[:, :, :1]
        val_stamp= self.val_x[:, :, :1]
        test_stamp = self.test_x[:, :, :1]

        self.train_x = np.array(self.train_x[:, :, :], dtype=np.float64)

        if self.scale:
            self.scaler.fit(self.train_x)
            self.train_x = self.scaler.transform(self.train_x)
            self.val_x = self.scaler.transform(np.array(self.val_x[:, :, :], dtype=np.float64))
            self.test_x = self.scaler.transform(np.array(self.test_x[:, :, :], dtype=np.float64))

        if self.set_type == 0:
            self.data_x = self.train_x
            self.data_y = self.train_y
            self.data_stamp = train_stamp
            self.data_attributes = train_attr
        elif self.set_type == 1:
            self.data_x = self.val_x
            self.data_y = self.val_y
            self.data_stamp = val_stamp
            self.data_attributes = val_attr
        elif self.set_type == 2:
            self.data_x = self.test_x
            self.data_y = self.test_y
            self.data_stamp = test_stamp
            self.data_attributes = test_attr
        

    def __getitem__(self, index):
        s_begin = index

        seq_x = self.data_x[index]
        seq_x_mark = self.data_stamp[index]
        attributes = self.data_attributes[index]

        trues = torch.Tensor(np.array(self.data_y[index]))
        trues = torch.clamp(trues, max=125)
        return seq_x, trues, seq_x_mark, seq_x_mark, attributes

    ##没搞清楚为什么缺了一部分
    def __len__(self):
        return len(self.data_x) - 10
    
    ##新数据集构建
    def create_dataset(self, data, unit, seq_len=600, interval=60):

        features = []
        targets = []
        attributes = []

        data = data.iloc[:, 2:]

        for i in range(0, len(data) - seq_len, interval):
            label = data.iloc[i:i + seq_len, -1]

            temp_data = data.iloc[i:i + seq_len, :-1]
            temp_attribute = unit

            features.append(temp_data)
            targets.append(label)
            attributes.append(temp_attribute)

        return np.array(features), np.array(targets), np.array(attributes)
    
    def create_test_dataset(self, data, unit, seq_len=600, interval=60):

        features = []
        targets = []
        attributes = []

        data = data.iloc[:, 2:]

        for i in range(0, len(data) - seq_len, interval):
            label = data.iloc[len(data) - seq_len -1 : len(data)-1 , -1]

            temp_data = data.iloc[len(data) - seq_len -1 : len(data)-1, :-1 ]
            temp_attribute = unit

            features.append(temp_data)
            targets.append(label)
            attributes.append(temp_attribute)

        return np.array(features), np.array(targets), np.array(attributes)
    
    def label_split(self, string):

            ss=string.split(',')

            return ss
    

class Dataset_IMS(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):

        self.scaler = data_StandardScaler()
        df_train = pd.read_csv(os.path.join(self.root_path, 'bearing', self.data_path+'_train.csv'))       
        df_test = pd.read_csv(os.path.join(self.root_path, 'bearing', self.data_path+'_test.csv'))

        cols = list(df_train.columns)
        print(cols)
        cols.remove('date');
        cols.remove('rul');

        df_train = df_train[['date'] + cols + ['rul']]
        df_test = df_test[['date'] + cols + ['rul']]

        num_flag = 0
        df_train = df_train.groupby('date')
       
        self.train_x = []  # 初始化为空列表
        self.train_y = []  # 初始化为空列表

        for date in tqdm(df_train.size().index):
            group = df_train.get_group(date)
            if group.shape[0] == self.seq_len:
                X_train = group.iloc[:, 1:-1].values  # 转换成numpy数组
                y_train = group.iloc[0, -1]           # 这里仅取值，未转换成numpy数组

                self.train_x.append(X_train)
                self.train_y.append(y_train)  # 附加到列表中

        # 循环结束后，将列表转换为numpy数组
        self.train_x = np.stack(self.train_x, axis=0)
        self.train_y = np.array(self.train_y).reshape(-1, 1)  # 确保是列向量


        num_flag = 0
        df_test = df_test.groupby('date')
        self.test_x = []
        self.val_x = []
        self.test_y = []
        self.val_y = []

        for date in tqdm(df_test.size().index):

            if(df_test.get_group(date).shape[0] == self.seq_len):  

                X_val = df_test.get_group(date).iloc[:, 1:-1]
                y_val = df_test.get_group(date).iloc[0, -1]
               

                if num_flag == 0:
                    self.val_x = [X_val.values]
                    self.val_y = y_val
                    self.test_x =[X_val.values] 
                    self.test_y = y_val

                else:             
                    self.val_x.append(X_val.values)
                    self.val_y = np.append(self.val_y, y_val)
                    self.test_x.append(X_val.values)
                    self.test_y = np.append(self.test_y, y_val)

                num_flag += 1

        self.val_x = np.stack(self.val_x, axis=0) 
        self.val_y = np.array(self.val_y).reshape(-1, 1)
        self.test_x = np.stack(self.test_x, axis=0) 
        self.test_y = np.array(self.test_y).reshape(-1, 1)

        self.train_x = np.array(self.train_x, dtype=np.float64)

        if self.scale:
            self.scaler.fit(self.train_x)
            self.train_x = self.scaler.transform(self.train_x)
            self.val_x = self.scaler.transform(np.array(self.val_x[:, :, :], dtype=np.float64))
            self.test_x = self.scaler.transform(np.array(self.test_x[:, :, :], dtype=np.float64))

        if self.set_type == 0:
            self.data_x = self.train_x
            self.data_y = self.train_y
        elif self.set_type == 1:
            self.data_x = self.val_x
            self.data_y = self.val_y
        elif self.set_type == 2:
            self.data_x = self.test_x
            self.data_y = self.test_y
        

    def __getitem__(self, index):
        s_begin = index

        seq_x = self.data_x[index]
        seq_x_mark = torch.Tensor(np.array([0]))
        attributes = torch.Tensor(np.array([0]))

        trues = torch.Tensor(np.array(self.data_y[index]))
        trues = torch.clamp(trues, max=125)
        return seq_x, trues, seq_x_mark, seq_x_mark, attributes

    ##没搞清楚为什么缺了一部分
    def __len__(self):
        return len(self.data_x) - 10

class Dataset_FEMTO(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):

        self.scaler = data_StandardScaler()
        df_train = pd.read_csv(os.path.join(self.root_path, 'bearing', self.data_path+'_train.csv'))       
        df_test = pd.read_csv(os.path.join(self.root_path, 'bearing', self.data_path+'_test.csv'))

        cols = list(df_train.columns)
        print(cols)
        cols.remove('bearing');
        cols.remove('rul');
        cols.remove('time');

        df_train = df_train[['bearing'] + ['time'] + cols + ['rul']]
        df_test = df_test[['bearing'] +  ['time'] + cols + ['rul']]

        num_flag = 0
        df_train = df_train.groupby('bearing')
       
        self.train_x = []  # 初始化为空列表
        self.train_y = []  # 初始化为空列表

        for bearing in tqdm(df_train.size().index):
            group = df_train.get_group(bearing)

            once = group.groupby('time')

            for time in once.size().index:

                if once.get_group(time).shape[0] == self.seq_len:
                    X_train = once.get_group(time).iloc[:, 2:-1].values  # 转换成numpy数组
                    y_train = once.get_group(time).iloc[0, -1]           # 这里仅取值，未转换成numpy数组

                    self.train_x.append(X_train)
                    self.train_y.append(y_train)  # 附加到列表中

        # 循环结束后，将列表转换为numpy数组
        self.train_x = np.stack(self.train_x, axis=0)
        self.train_y = np.array(self.train_y).reshape(-1, 1)  # 确保是列向量


        num_flag = 0
        df_test = df_test.groupby('bearing')
        self.test_x = []
        self.val_x = []
        self.test_y = []
        self.val_y = []

        for bearing in tqdm(df_test.size().index):
            
            group = df_test.get_group(bearing)
           
            once = group.groupby('time')

            for time in once.size().index:

                if(once.get_group(time).shape[0] == self.seq_len):  

                    X_val = once.get_group(time).iloc[:, 2:-1]
                    y_val = once.get_group(time).iloc[0, -1]
                

                    if num_flag == 0:
                        self.val_x = [X_val.values]
                        self.val_y = y_val
                        self.test_x =[X_val.values] 
                        self.test_y = y_val

                    else:             
                        self.val_x.append(X_val.values)
                        self.val_y = np.append(self.val_y, y_val)
                        self.test_x.append(X_val.values)
                        self.test_y = np.append(self.test_y, y_val)

                    num_flag += 1

        self.val_x = np.stack(self.val_x, axis=0) 
        self.val_y = np.array(self.val_y).reshape(-1, 1)
        self.test_x = np.stack(self.test_x, axis=0) 
        self.test_y = np.array(self.test_y).reshape(-1, 1)

        self.train_x = np.array(self.train_x, dtype=np.float64)

        if self.scale:
            self.scaler.fit(self.train_x)
            self.train_x = self.scaler.transform(self.train_x)
            self.val_x = self.scaler.transform(np.array(self.val_x[:, :, :], dtype=np.float64))
            self.test_x = self.scaler.transform(np.array(self.test_x[:, :, :], dtype=np.float64))

        if self.set_type == 0:
            self.data_x = self.train_x
            self.data_y = self.train_y
        elif self.set_type == 1:
            self.data_x = self.val_x
            self.data_y = self.val_y
        elif self.set_type == 2:
            self.data_x = self.test_x
            self.data_y = self.test_y
        

    def __getitem__(self, index):
        s_begin = index

        seq_x = self.data_x[index]
        seq_x_mark = torch.Tensor(np.array([0]))
        attributes = torch.Tensor(np.array([0]))

        trues = torch.Tensor(np.array(self.data_y[index]))
        trues = torch.clamp(trues, max=60)
        return seq_x, trues, seq_x_mark, seq_x_mark, attributes

    ##没搞清楚为什么缺了一部分
    def __len__(self):
        return len(self.data_x) - 10

class Dataset_M4(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        # size [seq_len, label_len, pred_len]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


class PSMSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        self.val = test_data
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, root_path, file_list=None, limit_size=None, flag=None):
        self.root_path = root_path
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        print(len(self.all_IDs))

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from csv files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_path, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_path, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_path, '*')))
        if flag is not None:
            data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
        input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith('.ts')]
        if len(input_paths) == 0:
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))

        all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath):
        df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                             replace_missing_vals_with='NaN')
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        return self.instance_norm(torch.from_numpy(self.feature_df.loc[self.all_IDs[ind]].values)), \
               torch.from_numpy(self.labels_df.loc[self.all_IDs[ind]].values)

    def __len__(self):
        return len(self.all_IDs)
