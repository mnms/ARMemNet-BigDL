# import
from zoo import init_nncontext

from pyspark import SparkConf                                                                                                                 
from pyspark.context import SparkContext                                                                                                      
from pyspark.sql import SparkSession, SQLContext

from pyspark.sql.types import StructType, StructField, TimestampType, DateType, FloatType, IntegerType
from pyspark.sql.functions import expr, col, column, array, lit, create_map, monotonically_increasing_id, lead, min, max, row_number, desc, when
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.window import Window

import itertools
import numpy as np
import pandas as pd
import sys

# Config for Preprocessing
class PreprocessConfig():
    def __init__(self):
        self.DATA_PATH      = "/user/nvkvs/data-sample/aggregated_resampled.csv" # data path for resampled dataset
        # self.VALID_CELL_IDS = list(range(20))                       # use 20 cells
        self.VALID_CELL_IDS = [19]                                  # use CELL ID 19

        self.FEAT_MINMAX    = [-1.0, 1.0]                           # feature min/max value for normalization
        self.FEATURE_SIZE   = 8                                     # 8 ; number of features

        self.INPUT_X_SIZE   = 10                                    # 10; Input X Size (5min x 10 = 50min)
        self.INPUT_Y_SIZE   = 1                                     # 1 ; Input Y Size (to predict after 5min)
        self.INPUT_M_SIZE   = self.INPUT_X_SIZE + self.INPUT_Y_SIZE # 11; Should remember X + Y
        self.ITEMS_PER_DAY  = int((21 - 8) * 60 / 5)                # 156 ; Hours from 08 to 21, 5 min resampling interval
        self.DAYS_TO_MEMORY = 7                                     # 7 ; Memory days
        self.COLS_TO_EXCLUDE = ['dt', 'CELL_NUM']                   # columns which are not features

        self.VALID_SET_RATIO = 0.1                                  # validation set ratio
        self.RANDOM_SEED_NUM = 42                                   # random seed for reproducing
        self.TEST_SET_SIZE   = self.ITEMS_PER_DAY * self.DAYS_TO_MEMORY - self.INPUT_X_SIZE # test_set_size

    def __str__(self):
        return "%r" % (self.__dict__)


# load 'resampled' data from given path
def load_resampled_data(data_path):
    data_schema = StructType([
        StructField("dt", TimestampType()),
        StructField("SQI", FloatType()),
        StructField("RSRP", FloatType()),
        StructField("RSRQ", FloatType()),
        StructField("DL_PRB_USAGE_RATE", FloatType()),
        StructField("SINR", FloatType()),
        StructField("UE_TX_POWER", FloatType()),
        StructField("PHR", FloatType()),
        StructField("UE_CONN_TOT_CNT", FloatType()),
        StructField("CELL_NUM", IntegerType())
    ])

    sc = init_nncontext()
    df = SQLContext(sc).read.format('com.databricks.spark.csv').option('header', 'true').schema(data_schema).load(data_path)
    # df = spark.read.csv(data_path, header=header_included, schema=data_schema)

    return df

# nomarlize dataframe by given min/max (feat_minmax)
def norm_df(df_to_norm, feat_minmax=(-1.0, 1.0), cols_to_exclude=('dt', 'CELL_NUM')):
    cols_to_norm = [c for c in df_to_norm.columns if c not in cols_to_exclude]
    aggregated_cols = list(itertools.chain(*([[min(c).alias(c + '_min'), max(c).alias(c + '_max')] for c in cols_to_norm])))
    min_max_row = df_to_norm.select(aggregated_cols).collect()[0]
    feat_min = feat_minmax[0]
    feat_max = feat_minmax[1]
    for col in cols_to_norm:
        real_min = min_max_row[col + '_min']
        real_max = min_max_row[col + '_max']
        df_to_norm = df_to_norm.withColumn(col, (df_to_norm[col] - real_min) * (feat_max - feat_min) / (real_max - real_min) + feat_min)
    return df_to_norm, min_max_row

# un-nomarlization dataframe by given mix/max (real_minmax & feat_minmax)
def unnorm_df(df_to_unnorm, min_max_row, feat_minmax=(-1.0, 1.0), cols_to_exclude=('dt', 'CELL_NUM')):
    cols_to_unnorm = [c for c in df_to_unnorm.columns if c not in cols_to_exclude]
    # Unnomarlize
    feat_min = feat_minmax[0]
    feat_max = feat_minmax[1]
    for col in cols_to_unnorm:
        real_min = min_max_row[col + '_min']
        real_max = min_max_row[col + '_max']
        df_to_unnorm = df_to_unnorm.withColumn(col, (df_to_unnorm[col] - feat_min) / (feat_max - feat_min) * (
                    real_max - real_min) + real_min)
    return df_to_unnorm


# extract valid CELL_IDs
def filter_valid_cells(df, valid_cell_ids):
    df = df.filter('CELL_NUM IN (' + ', '.join(str(cell_id) for cell_id in valid_cell_ids) + ')')
    return df

# assemble all features in One column without 'dt' and 'CELL_NUM'
def assemble_features(df_to_assemble, cols_to_exclude=['dt', 'CELL_NUM']):
    feat_cols = [c for c in df_to_assemble.columns if c not in cols_to_exclude]

    df_feat_assembled = VectorAssembler().setInputCols(feat_cols).setOutputCol("features").transform(
        df_to_assemble).select(['dt', 'CELL_NUM', 'features'])

    return df_feat_assembled

# Generate Dataset for X & Y
def generate_dataset_x_y(df_assembled, CONFIG_PREPROCESS):
    x_window_spec = Window.partitionBy('CELL_NUM').orderBy('dt')

    x_head_skip_size = CONFIG_PREPROCESS.DAYS_TO_MEMORY * CONFIG_PREPROCESS.ITEMS_PER_DAY
    shuffled_with_cell = df_assembled.withColumn('seq', row_number().over(x_window_spec)).cache()

    input_x = shuffled_with_cell.filter('seq > ' + str(x_head_skip_size)).filter(
        shuffled_with_cell['seq'] > x_head_skip_size)

    x_feat_cols = ['features0']
    input_x = input_x.withColumnRenamed('features', x_feat_cols[0])

    for i in range(1, CONFIG_PREPROCESS.INPUT_X_SIZE):
        n_features = lead(col(x_feat_cols[0]), i).over(x_window_spec)
        input_x = input_x.withColumn('features' + str(i), n_features)
        x_feat_cols.append('features{}'.format(i))

    x_tail_skip_size = CONFIG_PREPROCESS.INPUT_X_SIZE + CONFIG_PREPROCESS.INPUT_Y_SIZE - 1
    inbound = when(input_x['seq'] <= (
                max(input_x['seq']).over(x_window_spec.rangeBetween(-sys.maxsize, sys.maxsize)) - x_tail_skip_size),
                   1).otherwise(0)
    input_x = input_x.withColumn('inbound', inbound).filter('inbound == 1')
    input_x = VectorAssembler().setInputCols(x_feat_cols).setOutputCol('features').transform(input_x).select(
        ['dt', 'CELL_NUM', 'features'])

    y_head_skip_size = CONFIG_PREPROCESS.DAYS_TO_MEMORY * CONFIG_PREPROCESS.ITEMS_PER_DAY + CONFIG_PREPROCESS.INPUT_X_SIZE  # rows to skip, for M & X
    input_y = shuffled_with_cell.filter('seq > ' + str(y_head_skip_size)).drop('seq')

    return input_x, input_y

# Generate Dataset for M
def generate_dataset_m(df_assembled, CONFIG_PREPROCESS):
    m_window_spec = Window.partitionBy('CELL_NUM').orderBy('dt')
    m_feat_cols = ['day0_features0']
    m_days_cols = ['day0_features']

    shuffled_with_cell = df_assembled.withColumn('seq', row_number().over(m_window_spec)).cache()
    input_m = shuffled_with_cell.withColumnRenamed('features', m_feat_cols[0])

    # Generate 1 day data (5min * 10 data)
    for i in range(1, CONFIG_PREPROCESS.INPUT_M_SIZE):
        n_features = lead(col(m_feat_cols[0]), i).over(m_window_spec)
        input_m = input_m.withColumn('day{}_features{}'.format(0, i), n_features)
        m_feat_cols.append('day{}_features{}'.format(0, i))

    input_m = input_m.dropna()
    input_m = VectorAssembler().setInputCols(m_feat_cols).setOutputCol(m_days_cols[0]).transform(input_m)

    # for DAYS_TO_MEMORY(7) days memory in same time zone
    for i in range(1, CONFIG_PREPROCESS.DAYS_TO_MEMORY):
        n_features = lead(col('day0_features'), int(CONFIG_PREPROCESS.ITEMS_PER_DAY * i)).over(m_window_spec)
        input_m = input_m.withColumn('day{}_features'.format(i), n_features)
        m_days_cols.append('day{}_features'.format(i))

    m_tail_skip_size = CONFIG_PREPROCESS.ITEMS_PER_DAY  # rows to skip, for 1 Day (X & Y)
    inbound = when(input_m['seq'] <= (
                max(input_m['seq']).over(m_window_spec.rangeBetween(-sys.maxsize, sys.maxsize)) - m_tail_skip_size),
                   1).otherwise(0)
    input_m = input_m.dropna().withColumn('inbound', inbound).filter('inbound == 1')
    input_m = VectorAssembler().setInputCols(m_days_cols).setOutputCol('features').transform(input_m).select(
        ['dt', 'CELL_NUM', 'features'])  # assemble DAYS_TO_MEMORY days columns into one ('features')

    return input_m

# Generate All Dataset, return as NumPy Array
def generate_dataset(df_assembled, CONFIG_PREPROCESS):
    valid_cell_size = len(CONFIG_PREPROCESS.VALID_CELL_IDS)

    df_assembled = df_assembled.cache()
    (input_x, input_y) = generate_dataset_x_y(df_assembled, CONFIG_PREPROCESS)
    input_m = generate_dataset_m(df_assembled, CONFIG_PREPROCESS)

    np_x = np.array(input_x.sort('dt', 'CELL_NUM').select('features').collect()).reshape(-1, valid_cell_size,
                                                                                         CONFIG_PREPROCESS.INPUT_X_SIZE,
                                                                                         CONFIG_PREPROCESS.FEATURE_SIZE)  # [slided, cells, INPUT_X_SIZE, features]
    np_y = np.array(input_y.sort('dt', 'CELL_NUM').select('features').collect()).reshape(-1, valid_cell_size,
                                                                                         CONFIG_PREPROCESS.INPUT_Y_SIZE,
                                                                                         CONFIG_PREPROCESS.FEATURE_SIZE)  # [slided, cells, INPUT_Y_SIZE, features]
    np_m = np.array(input_m.sort('dt', 'CELL_NUM').select('features').collect()).reshape(-1, valid_cell_size,
                                                                                         CONFIG_PREPROCESS.INPUT_M_SIZE * CONFIG_PREPROCESS.DAYS_TO_MEMORY,
                                                                                         CONFIG_PREPROCESS.FEATURE_SIZE)  # [slided, cells, INPUT_M_SIZE * DAYS_TO_MEMORY, features]

    # divide dataset into train / test set
    train_x = np_x[:-CONFIG_PREPROCESS.TEST_SET_SIZE].reshape(-1, CONFIG_PREPROCESS.INPUT_X_SIZE,
                                                              CONFIG_PREPROCESS.FEATURE_SIZE)  # [slided * cells, INPUT_X_SIZE, features]
    # train_y = np_y[:-CONFIG_PREPROCESS.TEST_SET_SIZE].reshape(-1, CONFIG_PREPROCESS.INPUT_Y_SIZE, CONFIG_PREPROCESS.FEATURE_SIZE) # [slided * cells, INPUT_Y_SIZE, features]
    train_y = np_y[:-CONFIG_PREPROCESS.TEST_SET_SIZE].reshape(-1,
                                                              CONFIG_PREPROCESS.FEATURE_SIZE)  # [slided * cells, features]
    train_m = np_m[:-CONFIG_PREPROCESS.TEST_SET_SIZE].reshape(-1,
                                                              CONFIG_PREPROCESS.INPUT_M_SIZE * CONFIG_PREPROCESS.DAYS_TO_MEMORY,
                                                              CONFIG_PREPROCESS.FEATURE_SIZE)  # [slided * cells, INPUT_M_SIZE * DAYS_TO_MEMOR

    test_x = np_x[-CONFIG_PREPROCESS.TEST_SET_SIZE:].reshape(-1, CONFIG_PREPROCESS.INPUT_X_SIZE,
                                                             CONFIG_PREPROCESS.FEATURE_SIZE)  # [slided * cells, INPUT_X_SIZE, features]
    # test_y = np_y[-CONFIG_PREPROCESS.TEST_SET_SIZE:].reshape(-1, CONFIG_PREPROCESS.INPUT_Y_SIZE, CONFIG_PREPROCESS.FEATURE_SIZE) # [slided * cells, INPUT_Y_SIZE, features]
    test_y = np_y[-CONFIG_PREPROCESS.TEST_SET_SIZE:].reshape(-1,
                                                             CONFIG_PREPROCESS.FEATURE_SIZE)  # [slided * cells, INPUT_Y_SIZE, features]
    test_m = np_m[-CONFIG_PREPROCESS.TEST_SET_SIZE:].reshape(-1,
                                                             CONFIG_PREPROCESS.INPUT_M_SIZE * CONFIG_PREPROCESS.DAYS_TO_MEMORY,
                                                             CONFIG_PREPROCESS.FEATURE_SIZE)  # [slided * cells, INPUT_M_SIZE * DAYS_TO_MEMOR

    # valid set factor
    if CONFIG_PREPROCESS.RANDOM_SEED_NUM:
        np.random.seed(CONFIG_PREPROCESS.RANDOM_SEED_NUM)

    valid_len = int(len(train_x) * CONFIG_PREPROCESS.VALID_SET_RATIO)
    valid_idx = np.random.permutation(len(train_x))[:valid_len]
    train_idx = [i for i in range(len(train_x)) if i not in valid_idx]

    # divide train data into train / valid set
    valid_x = train_x[valid_idx]
    valid_y = train_y[valid_idx]
    valid_m = train_m[valid_idx]

    train_x = train_x[train_idx]
    train_y = train_y[train_idx]
    train_m = train_m[train_idx]

    return train_x, train_y, train_m, valid_x, valid_y, valid_m, test_x, test_y, test_m
