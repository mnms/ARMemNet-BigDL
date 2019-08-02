# config for preprocessing


DOWNLINK_COL_LIST = [
    'CQI',
    'RSRP',
    'RSRQ',
    'DL_PRB_USAGE_RATE',
]

UPLINK_COL_LIST = [
    'SINR',
    'UE_TX_POWER',
    'PHR',
    # 'UL_PRB_USAGE_RATE',
]

ADDITIONAL_COL_LIST = [
    # 'Bandwidth',
    'UE_CONN_TOT_CNT',
]   


class ConfigData(object):
    def __init__(self):
        # General config
        self.raw_file_path = '../data/raw'
        self.output_path = '../data/aggregated_5min_scaled.csv'
        self.col_list = DOWNLINK_COL_LIST + UPLINK_COL_LIST + ADDITIONAL_COL_LIST
        self.enb_cell_ids = [
            '00000_00',   # Need some ENB_CELL_IDs
        ]
        self.drop_resampled = False # If True, it drop resampled data (file name: aggregated_resampled.csv) before scaling

        # Time & freq config
        self.start_dt='2018-07-02'
        self.end_dt='2018-08-27' # Data will be processed up to (end_dt-1 day). For example, if we set end_dt='2018-08-27', it use data until 2018-08-26.
        self.start_hour=8
        self.end_hour=21
        self.base_freq='10S'        # Base freqeuncy in raw data format.
        self.resample_freq='5min'   # Target frequency

        # Scaling config
        self.scaling_range = (-1., 1.)

