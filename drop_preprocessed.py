import os
import shutil
from glob import glob
import numpy as np
import pandas as pd
import pickle
from config_preprocess import ConfigData

# Get a file list.
def data_loader(
    base_path,
    enb_id=None,
    cell_id=None,
    ):
    
    file_list = sorted(
        glob(
            os.path.join(
                base_path,
                "{ENB_ID}_{CELL_ID}_*.csv".format(
                    ENB_ID=enb_id,
                    CELL_ID=cell_id,
                ),
            ),
        ),
    )
    print('Found %s Files.' % len(file_list))

    # Load it.
    full_data = pd.concat(map(pd.read_csv, file_list))
    data_tmidx = pd.to_datetime(
        full_data['EVT_DTM'],
        format='%Y%m%d%H%M%S'
    )
    full_data.set_index(data_tmidx, inplace=True)

    # Drop the Duplicated.
    full_data.drop_duplicates(
        subset='EVT_DTM',
        keep='last',
        inplace=True,
    )
    
    return full_data


def datetime_setter(
    data,
    start_dt=None,
    end_dt=None,
    start_hour=None,
    end_hour=None,
    freq='10S',
    ):
    
    full_datetime = pd.date_range(
            start=str(start_dt),
            end=str(end_dt),
            freq=base_freq,
        )

    selected_datetime = full_datetime[
        (int(start_hour) <= full_datetime.hour) &
        (full_datetime.hour < int(end_hour))
    ]

    tmp_table = pd.DataFrame([],
        index=selected_datetime,
        #columns=col_list,
    )

    joined = tmp_table.join(
        data,
        how='left',
        sort=True,
    )

    if 'UE_CONN_TOT_CNT' in joined.columns:
        joined['UE_CONN_TOT_CNT'] = joined['UE_CONN_TOT_CNT'].astype(
            np.float32
        )
        
    return joined


def scaler(dataframe, col_range_dict=None, feature_range=(-1., 1.), c_idx_scale=False):

    if c_idx_scale:
        if set(dataframe.columns) - set(col_range_dict):
            raise AttributeError("'col_real_range_dict' should be contains all columns in 'dataframe'")
    else:
        result_frame = dataframe.copy()
        feature_max = feature_range[1]
        feature_min = feature_range[0]

        for col in dataframe.columns[:-1]: # last column X
            real_min, real_max = col_range_dict[col]
            scale = (feature_max-feature_min)/(real_max-real_min)
            scaled_col = (result_frame[col]-real_min) * scale + feature_min
            result_frame[col]= scaled_col
        
        return result_frame


def unscaler(scaled, col_to_unscale, col_range_dict='../data/agg_data_5min_range_dict.p', feature_range=(-1., 1.)):

    unscaled = scaled
    with open(col_range_dict, 'rb') as f:
        range_dict = pickle.load(f)

    feature_max = feature_range[1]
    feature_min = feature_range[0]

    for i, col in enumerate(col_to_unscale):
        real_min, real_max = range_dict[col]
        scale = (feature_max-feature_min)/(real_max-real_min)
        unscaled_col = (unscaled[:,i] - feature_min) / scale + real_min
        unscaled[:,i] = unscaled_col
    
    return unscaled



if __name__=="__main__":
    config = ConfigData()
    unique_cells = list(map(lambda enb_cell : enb_cell.split("_"), config.enb_cell_ids))
    raw_file_path = config.raw_file_path

    print("The number of cells for preprocessing: {}".format(len(unique_cells)))
    
    # Read and Resample for each cell
    cell_num = 0 
    total_targetday_data = []   
    for i in range(len(unique_cells)):

        col_list = config.col_list
        enb_id = unique_cells[cell_num][0]
        cell_id = unique_cells[cell_num][1]
        start_dt = config.start_dt
        end_dt = config.end_dt
        start_hour = config.start_hour
        end_hour = config.end_hour
        base_freq = config.base_freq
        resample_freq = config.resample_freq

        # Data Loading
        data = data_loader(
            base_path=raw_file_path,
            enb_id=enb_id,
            cell_id=cell_id,
        )

        # DatetimeIndex Setting
        selected = datetime_setter(
            data,
            start_dt=start_dt,
            end_dt=end_dt,
            start_hour=start_hour,
            end_hour=end_hour,
            freq=base_freq,
        )
        
        # Resampling
        resampled = selected.groupby(
            pd.Grouper(freq='D')
        )[col_list].resample(resample_freq).mean()
        resampled.reset_index(level=0, drop=True, inplace=True)

        daily_filled = resampled.groupby(pd.Grouper(freq='D'))[col_list].apply(
            lambda grp: grp.ffill().bfill()
        )
        daily_filled = daily_filled.ffill().bfill()
        daily_filled = daily_filled.fillna(0.)
        
        daily_filled['CELL_NUM'] = cell_num
        total_targetday_data.append(daily_filled)
        cell_num += 1

    result = pd.concat(total_targetday_data)

    print("Read & Resampling finished")
    
    if config.drop_resampled:
        result.to_csv('../data/{}'.format("aggregated_resampled.csv"), encoding='utf-8')

    print("Starting scaling process")
    # Scaling
    data_np = result[col_list].values
    mins = np.min(data_np, axis=0)
    maxs = np.max(data_np, axis=0)
    
    col_range_dict = {}
    for i, col in enumerate(col_list):
        col_range_dict[col] = [mins[i], maxs[i]]
        
    scaling_range = config.scaling_range
    full_scaled = scaler(result, col_range_dict=col_range_dict, feature_range=scaling_range)
    print("Scaling process finished")

    # Drop scaled
    full_scaled.to_csv('../data/{}'.format(config.output_path), encoding='utf-8')

    col_range_dict_name = "_".join(config.output_path.split("_")[:-1]+["range_dict.p"])
    with open('../data/{}'.format(col_range_dict_name), 'wb') as f:
        pickle.dump(col_range_dict, f)
        