# AR_mem_config


class Config(object):
    def __init__(self):
        # model params
        self.model = "ARMem"
        self.nsteps = 10   # equivalent to x_len
        self.msteps = 7
        self.attention_size = 16
        self.l2_lambda = 1e-3
        self.ar_lambda = 0.1
        self.ar_g = 1

        # data params
        self.data_path = '../data/aggregated_5min_scaled.csv'
        self.nfeatures = 8  # number of col_list in "../config_preprocess.py"
        self.x_len = self.nsteps
        self.y_len = 1
        self.foresight = 0
        self.dev_ratio = 0.1
        self.test_len = 7
        self.seed = None

        # train & test params
        self.train_cell_ids = [11, 16, 18]  # order of cell_id in "../config_preprocess.py"
        self.test_cell_ids = [18]           # order of cell_id in "../config_preprocess.py"
        self.model_dir = None       # Model directory to use in test mode. For example, "model_save/20190405-05"
        self.latest_model = True    # Use lately saved model in test mode. If latest_model=True, model_dir option will be ignored

        # training params
        self.lr = 1e-3
        self.num_epochs = 1000
        self.batch_size = 32
        self.dropout = 0.8
        self.nepoch_no_improv = 5
        self.clip = 5
        self.allow_gpu = True
        self.desc = self._desc()

    def _desc(self):
        desc = ""
        for mem, val in self.__dict__.items():
            desc += mem + ":" + str(val) + ", "
        return desc


if __name__ == "__main__":
    config = Config()
    print(config.desc)
