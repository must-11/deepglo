#### OS and commanline arguments
import argparse
import gzip
import logging
import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path

sys.path.append('./')


import json
import pickle
import random

import numpy as np
import pandas as pd
from DeepGLO.DeepGLO import *
from DeepGLO.LocalModel import *
#### DeepGLO model imports
from DeepGLO.metrics import *

np.random.seed(111)
torch.cuda.manual_seed(111)
torch.manual_seed(111)
random.seed(111)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def bool2str(b):
    if b:
        return "true"
    else:
        return "false"

train = pd.read_csv("/kaggle/input/rossmann-store-sales/train.csv")
Ymat = train.pivot(index="Store", columns="Date", values="Sales").dropna(axis=0).values

vbsize = 128  ## vertical batch size
hbsize = 256  ## horizontal batch size
num_channels_X = [32, 32, 32, 32, 32, 1]  ## number of channels for local model
num_channels_Y = [32, 32, 32, 32, 32, 1]  ## number of channels for hybrid model
kernel_size = 7  ## kernel size for local models
dropout = 0.2  ## dropout during training
rank = 64  ## rank of global model
kernel_size_Y = 7  ## kernel size of hybrid model
lr = 0.0005  ## learning rate
val_len = 48  ## validation length
end_index = Ymat.shape[1]  ## models will not look beyond this during training
start_date = "2013-1-1"  ## start date time for the time-series
freq = "H"  ## frequency of data
covariates = None  ## no covraites specified
use_time = True  ## us time covariates
dti = None  ## no spcified time covariates (using default)
svd = True  ## factor matrices are initialized by NMF
period = 24  ## periodicity of 24 is expected, leave it out if not known
y_iters = 300  ## max. number of iterations while training Tconv models
init_epochs = 100  ## max number of iterations while initialiozing factors
forward_cov = False


logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main(args):
    print("LOCAL MODEL====================")
    TC = LocalModel(
        Ymat,
        num_inputs=1,
        num_channels=num_channels_Y,
        kernel_size=kernel_size,
        dropout=dropout,
        vbsize=vbsize,
        hbsize=hbsize,
        num_epochs=y_iters,
        lr=lr,
        val_len=val_len,
        test=True,
        end_index=end_index - val_len,
        normalize=False,
        start_date=start_date,
        freq=freq,
        covariates=covariates,
        use_time=use_time,
        dti=dti,
        Ycov=None,
    )
    TC.train_model(early_stop=True, tenacity=7)

    result_dic = LM.rolling_validation(
        Ymat=Ymat, tau=24, n=7, bsize=100, cpu=False, alpha=0.3
    )
    print(result_dic)

    print("GLOBAL MODEL====================")
    DG = DeepGLO(
        Ymat,
        vbsize=vbsize,
        hbsize=hbsize,
        num_channels_X=num_channels_X,
        num_channels_Y=num_channels_Y,
        kernel_size=kernel_size,
        dropout=dropout,
        rank=rank,
        kernel_size_Y=kernel_size_Y,
        lr=lr,
        val_len=val_len,
        end_index=end_index,
        normalize=normalize,
        start_date=start_date,
        freq=freq,
        covariates=covariates,
        use_time=use_time,
        dti=dti,
        svd=svd,
        period=period,
        forward_cov=forward_cov,
    )

    DG.train_all_models(y_iters=y_iters, init_epochs=init_epochs)

    result_dic = DG.rolling_validation(
        Ymat=Ymat, tau=24, n=7, bsize=100, cpu=False, alpha=0.3
    )
    print(result_dic)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--normalize",
        type=str2bool,
        required=True,
        help="normalize for training or not",
    )
    args = parser.parse_args()
    global normalize
    normalize = args.normalize
    main(args)
