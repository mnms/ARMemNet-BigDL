import tensorflow as tf
from tensorflow.contrib import layers
import os
from utils import make_date_dir, find_latest_dir

import sys
from zoo.pipeline.api.net import TFNet
from zoo import init_nncontext, Sample
import tensorflow as tf
import numpy as np
from data_utils import load_agg_selected_data_mem
from ARMem.config import Config
from ARMem.model import Model
from preprocess_zoo import *

import time

# to reproduce the results in test_mem_model.py
# please set PARALLELISM to 1 and BATCH_PER_THREAD to 1022
PARALLELISM=4
BATCH_PER_THREAD=32


if __name__ == "__main__":
    # preprocess config
    config_preprocess = PreprocessConfig()

    # resampled csv, we'll change this later
    resampled_csv_filename = "/user/nvkvs/data-sample/aggregated_resampled.csv"

    # load resmapled data with given CELL_IDs
    df = load_resampled_data(resampled_csv_filename)

    # normalize
    df_normed, df_minmax = norm_df(df, feat_minmax=config_preprocess.FEAT_MINMAX,
                                   cols_to_exclude=config_preprocess.COLS_TO_EXCLUDE)

    # filter valid cells after normalize (because of scaling factor)
    df_normed = filter_valid_cells(df_normed, config_preprocess.VALID_CELL_IDS)

    # assemble features in one column named 'features'
    df_assembled = assemble_features(df_normed, cols_to_exclude=config_preprocess.COLS_TO_EXCLUDE)

    # prepare test data
    # train_x, train_y, train_m, valid_x, valid_y, valid_m, test_x, test_y, test_m = generate_dataset(df_assembled,
    #                                                                                                 config_preprocess)

    _, _, _, _, _, _, test_x, test_y, test_m = generate_dataset(df_assembled, config_preprocess)


    config = Config()

    config.latest_model=False

    model = Model(config)

    # init or get SparkContext
    sc = init_nncontext()

    # model_dir = config.model_dir
    model_dir = find_latest_dir(os.path.join(config.model, 'model_save/'))

    #  export a TensorFlow model to frozen inference graph.
    time_start = time.time()
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(model_dir, config.model))

        tfnet = TFNet.from_session(sess,
                                   inputs=[model.input_x, model.memories], # dropout is never used
                                   outputs=[model.predictions])

    data_x_rdd = sc.parallelize(test_x, PARALLELISM)
    data_m_rdd = sc.parallelize(test_m, PARALLELISM)

    # create a RDD of Sample
    sample_rdd = data_x_rdd.zip(data_m_rdd).map(
        lambda x: Sample.from_ndarray(features=x,
                                      labels=np.zeros([1])))

    # distributed inference on Spark and return an RDD
    outputs = tfnet.predict(sample_rdd,
                            batch_per_thread=BATCH_PER_THREAD,
                            distributed=True)
    time_end = time.time()

    print("Elapsed Time in Inferencing: {}".format(time_end - time_start))

    result_dir = make_date_dir(os.path.join(config.model, 'zoo_results/'))

    outputs.saveAsTextFile(os.path.join(result_dir, "result.txt"))


