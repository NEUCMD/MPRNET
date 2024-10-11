import os
import torch
from models import MPRNET, Autoformer, Transformer, iTransformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, PatchTST_GAT, MICN, Crossformer, FiLM, \
    Tide, Stateformer, Stateformer_ms, Stateformer_GAT, retnet, retnet_gat, retnet_inverted, GRU, TCN, STGCN, LSTM, NoPatch, Explor, SSSM


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'MPRNET': MPRNET,
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'iTransformer': iTransformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'Tide': Tide,
            'LSTM': LSTM,
            'GRU': GRU,
            'TCN': TCN,
            'STGCN': STGCN,
            'Stateformer': Stateformer_ms,
            'retnet': retnet,
            'SinglePatch': TimesNet,
            'NoPatch': NoPatch,
            'Explor_2_1': Explor,
            'SSSM': SSSM,
        }
        self.device = self._acquire_device()


    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
