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

# to reproduce the results in test_mem_model.py
# please set PARALLELISM to 1 and BATCH_PER_THREAD to 1022
PARALLELISM=4
BATCH_PER_THREAD=32


if __name__ == "__main__":
    config = Config()

    config.latest_model=False

    model = Model(config)

    # init or get SparkContext
    sc = init_nncontext()

    # create test data
    _, _, test_x, _, _, test_y, _, _, test_m, test_dt = \
        load_agg_selected_data_mem(data_path=config.data_path,
                                   x_len=config.x_len,
                                   y_len=config.y_len,
                                   foresight=config.foresight,
                                   cell_ids=config.test_cell_ids,
                                   dev_ratio=config.dev_ratio,
                                   test_len=config.test_len,
                                   seed=config.seed)

    model_dir = config.model_dir

    #  export a TensorFlow model to frozen inference graph.
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

    result_dir = make_date_dir(os.path.join(config.model, 'zoo_results/'))

    outputs.saveAsTextFile(os.path.join(result_dir, "result.txt"))


