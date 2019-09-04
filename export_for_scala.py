import os

from zoo import init_nncontext, Sample
import tensorflow as tf
import numpy as np
from data_utils import load_agg_selected_data_mem
from ARMem.config import Config
from ARMem.model import Model

# to reproduce the results in test_mem_model.py
# please set PARALLELISM to 1 and BATCH_PER_THREAD to 1022
from zoo.util.tf import export_tf
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

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

    test_x = np.concatenate([test_x] * 200, axis=0)
    test_m = np.concatenate([test_m] * 200, axis=0)

    np.save(os.path.join(dir_path, "data/test_x.npy"), test_x)
    np.save(os.path.join(dir_path, "data/test_m.npy"), test_m)

    model_dir = config.model_dir

    #  export a TensorFlow model to frozen inference graph.
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(model_dir, config.model))

        export_tf(sess, os.path.join(dir_path, "tfnet"), inputs=[model.input_x, model.memories], outputs=[model.predictions])

