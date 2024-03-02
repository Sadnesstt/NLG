import argparse
import pprint

import torch
from torch import optim
import torch.nn as nn

import torch_optimizer as custom_optim

from simple_nmt.data_loader import DatraLoader
import simple_nmt.data_loader as data_loader

from simple_nmt.models.seq2seq import Seq2Seq
# from simple_nmt.models.transformer import transformer
# from simple_nmt.models.rnnlm import LanguageModel

from simple_nmt.trainer import SingleTrainer
# from simple_nmt.rl_trainer import MinimumRiskTrainingEngine
from simple_nmt.trainer import MaximumLikelihoodEstimationEngine

def define_argparser(is_continue=False):
    p = argparse.ArgumentParser()

    if is_continue:
        p.add_argument(
            '--load_fn',
            required=True,
            help = 'Model file name to continue.'
        )
    
    p.add_argument(
        '--model_fn',
        required=not is_continue,
        help = 'Model file name to save. Additional information would be annotated to the file name.'
    )
    p.add_argument(
        '--train',
        required=not is_continue,
        help = 'Training set file name except the extenison. (ex: train.en --. train)'
    )
    p.add_argument(
        '--valid',
        required=not is_continue,
        help = 'Set of extention represents language pair. (ex: en + ko --> enko)' 
    )
    p.add_argument(
        '--gpu_id',
        type = int,
        default = -1,
        help = 'GPU ID to train. Currently, GPU parallel is not supported. -1 for CPU. Default=%(default)s'
    )
    p.add_argument(
        '--off_autocast',
        action = 'store_true',
        help = 'Turn-off Automatic Mixed Precision (AMP), which speed-up training.',
    )
    p.add_argument(
        '--batch_size',
        type = int,
        default = 32,
        help = 'Mini batch size for gradient descent. Default = %(default)s'
    )
    p.add_argument(
        '--n_epochs',
        type = int,
        default = 20,
        help = 'Number of epochs to train. Default = %(default)s'
    )
    p.add_argument(
        '--verbose',
        type = int,
        default = 2,
        help = 'VERBOSE_SILENT, VERBOSE_EPOCH_WISE, VERBOSE_BATCH_WISE = 0, 1, 2. Default=%(default)s'
    )
    p.add_argument(
        '--init_epoch',
        required=is_continue,
        type= int,
        default = 1,
        help = 'Set initial epoch'
    )