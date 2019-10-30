import tensorflow as tf
from tensorflow.contrib import layers
import os
from utils import make_date_dir, find_latest_dir

import sys
from zoo.pipeline.api.net import TFNet
from zoo import init_nncontext, Sample
from zoo.tfpark import TFOptimizer, TFDataset
from bigdl.optim.optimizer import *
import tensorflow as tf
import numpy as np
from data_utils import load_agg_selected_data_mem
from ARMem.config import Config
from ARMem.model import Model

# to reproduce the results in test_mem_model.py
# please set PARALLELISM to 1 and BATCH_PER_THREAD to 1022
PARALLELISM=4
BATCH_PER_THREAD=3200


if __name__ == "__main__":
    config = Config()

    config.latest_model=False

    # init or get SparkContext
    sc = init_nncontext()

    # create test data
    train_x, dev_x, test_x, train_y, dev_y, test_y, train_m, dev_m, test_m, test_dt = \
        load_agg_selected_data_mem(data_path=config.data_path,
                                   x_len=config.x_len,
                                   y_len=config.y_len,
                                   foresight=config.foresight,
                                   cell_ids=config.test_cell_ids,
                                   dev_ratio=config.dev_ratio,
                                   test_len=config.test_len,
                                   seed=config.seed)

    model_dir = config.model_dir

    dataset = TFDataset.from_ndarrays([train_x, train_m, train_y], batch_size=2700, val_tensors=[dev_x, dev_m, dev_y],)

    model = Model(config, dataset.tensors[0], dataset.tensors[1], dataset.tensors[2])
    optimizer = TFOptimizer.from_loss(model.loss, Adam(config.lr), metrics={"rse": model.rse, "smape": model.smape, "mae": model.mae})

    optimizer.optimize(end_trigger=MaxEpoch(15000))








