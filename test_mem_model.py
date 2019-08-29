import os
from utils import get_logger, make_date_dir, find_latest_dir
from data_utils import load_agg_selected_data_mem, batch_loader
import numpy as np
from time import time
from ARMem.config import Config
from ARMem.model import Model


def main():
    config = Config()

    logger, log_dir = get_logger(os.path.join(config.model, "logs/"))
    logger.info("=======Model Configuration=======")
    logger.info(config.desc)
    logger.info("=================================")

    try:
        _, _, test_x, _, _, test_y, _, _, test_m, test_dt = load_agg_selected_data_mem(data_path=config.data_path, \
            x_len=config.x_len, \
            y_len=config.y_len, \
            foresight=config.foresight, \
            cell_ids=config.test_cell_ids, \
            dev_ratio=config.dev_ratio, \
            test_len=config.test_len, \
            seed=config.seed)

        model = Model(config)
        if config.latest_model:
            model_dir = find_latest_dir(os.path.join(config.model, 'model_save/'))
        else:
            if not model_dir:
                raise Exception("model_dir or latest_model=True should be defined in config")
            model_dir = config.model_dir

        time_start = time()
        model.restore_session(model_dir)
        if len(test_y) > 100000:
            # Batch mode
            test_data = list(zip(test_x, test_m, test_y))
            test_batches = batch_loader(test_data, config.batch_size)
            total_pred = np.empty(shape=(0, test_y.shape[1]))

            for batch in test_batches:
                batch_x, batch_m, batch_y = zip(*batch)
                pred, _, _, _, _ = model.eval(batch_x, batch_m, batch_y)
                total_pred = np.r_[total_pred, pred]

        else:
            # Not batch mode
            total_pred, test_loss, test_rse, test_smape, test_mae = model.eval(test_x, test_m, test_y)
        time_end = time()
        print("Elapsed Time in Inferencing: {}".format(time_end - time_start))

        result_dir = make_date_dir(os.path.join(config.model, 'results/'))
        np.save(os.path.join(result_dir, 'pred.npy'), total_pred)
        np.save(os.path.join(result_dir, 'test_y.npy'), test_y)
        np.save(os.path.join(result_dir, 'test_dt.npy'), test_dt)
        logger.info("Saving results at {}".format(result_dir))
        logger.info("Testing finished, exit program")

    except:
        logger.exception("ERROR")

if __name__ == "__main__":
    main()
