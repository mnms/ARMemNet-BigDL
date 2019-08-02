import random
import pandas as pd
import numpy as np
import logging
import pickle

logger = logging.getLogger()


def batch_loader(iterable, batch_size, shuffle=False):
    length = len(iterable)
    if shuffle:
        random.shuffle(iterable)
    for idx in range(0, length, batch_size):
        yield iterable[idx:min(length, idx + batch_size)]
     

def load_agg_data(
        data_path='../data/aggregated_5min_scaled.csv',
        x_len=10,
        y_len=1,
        ncells=20,
        foresight=0,
        dev_ratio=.1,
        test_len=7,
        seed=None,
):

    data = pd.read_csv(data_path, index_col=0)
    data.index = pd.to_datetime(data.index)

    col_list = list(data.columns)
    col_list.remove("CELL_NUM")
    ndim_x = len(col_list)
    ndim_y = ndim_x

    full_x_ = []
    full_y_ = []
    full_dt_ = []

    for cell_id in range(ncells): # config 
        
        cell_data = data[data['CELL_NUM']==cell_id]

        grouped = cell_data.groupby(pd.Grouper(freq='D'))

        cell_x = np.empty((0, x_len, ndim_x), dtype=np.float32)
        cell_y = np.empty((0, y_len, ndim_y), dtype=np.float32)
        cell_dt = np.empty((0, y_len), dtype=str)

        for day, group in grouped:

            if not group.shape[0]:
                continue
            else:
                group_index = group.index.astype('str')
                source_x = group[col_list].sort_index().values.reshape(-1, ndim_x).astype('float32')
                source_y = group[col_list].sort_index().values.reshape(-1, ndim_y).astype('float32')

                slided_x = np.array([source_x[i:i + x_len] for i in range(0, len(source_x) - x_len - foresight - y_len + 1)])
                y_start_idx = x_len + foresight
                slided_y = np.array([source_y[i:i + y_len] for i in range(y_start_idx, len(source_y) - y_len + 1)])
                slided_dt = np.array([group_index[i:i + y_len] for i in range(y_start_idx, len(source_y) - y_len + 1)])

                cell_x = np.concatenate([cell_x, slided_x],axis=0)
                cell_y = np.concatenate([cell_y, slided_y],axis=0)
                cell_dt = np.concatenate([cell_dt, slided_dt],axis=0)

        full_x_.append(cell_x)
        full_y_.append(cell_y)
        full_dt_.append(cell_dt)

    full_x = np.stack(full_x_, axis=1)
    full_y = np.stack(full_y_, axis=1)
    full_dt = np.stack(full_dt_, axis=1)

    assert len(full_x) == len(full_y)

    # squeeze second dim if y_len = 1

    if y_len == 1:
        full_y = np.squeeze(full_y, axis=2)
        full_dt = np.squeeze(full_dt, axis=2)
    
    full_dt_tmp = full_dt[:, 0]

    end_dt = pd.to_datetime(full_dt_tmp[-1])
    start_dt = end_dt - pd.Timedelta(days=test_len)

    if y_len > 1:
        start_dt = start_dt.astype('str')[0]

    test_ind = -1
    for i in range(len(full_dt_tmp)):
        if y_len == 1:
            d = full_dt_tmp[i]
        else:
            d = full_dt[i,0]

        if d == str(start_dt):
            test_ind = i+1
            break

    assert test_ind != -1

    tr_x = full_x[:test_ind]
    tr_y = full_y[:test_ind]

    if seed :
        np.random.seed(seed)
    dev_len = int(len(tr_x) * dev_ratio)
    dev_ind = np.random.permutation(len(tr_x))[:dev_len]

    tr_ind = np.ones(len(tr_x), np.bool)
    tr_ind[dev_ind] = 0
    train_x, dev_x, test_x = tr_x[tr_ind], tr_x[dev_ind], full_x[test_ind:]
    train_y, dev_y, test_y = tr_y[tr_ind], tr_y[dev_ind], full_y[test_ind:]
    test_dt = full_dt[test_ind:]

    logger.info('X : {}, {}, {}'.format(train_x.shape, dev_x.shape, test_x.shape))
    logger.info('Y : {}, {}, {}'.format(train_y.shape, dev_y.shape, test_y.shape))
    logger.info('Test set start time: {}'.format(test_dt[0, 0]))

    return train_x, dev_x, test_x, train_y, dev_y, test_y, test_dt


def load_agg_selected_data_mem(
        data_path='../data/aggregated_5min_scaled.csv',
        x_len=10,
        y_len=1,
        mem_len=7,
        foresight=0,
        cell_ids=None,
        dev_ratio=.1,
        test_len=7,
        seed=None,
):

    data = pd.read_csv(data_path, index_col=0)
    data.index = pd.to_datetime(data.index)

    col_list = list(data.columns)
    col_list.remove("CELL_NUM")
    ndim_x = len(col_list)
    ndim_y = ndim_x

    full_m_lst=[]
    full_y_lst = []
    full_dt_lst = []
    full_x_lst = []
    full_cell_lst = []
    cell_list = [18]
    for cell_id in cell_list: # config 
        
        cell_data = data[data['CELL_NUM']==cell_id]
        grouped = cell_data.groupby(pd.Grouper(freq='D'))

        m_lst = []
        y_lst = []
        dt_lst = []
        for day, group in grouped:
            if not group.shape[0]:
                continue
            else:
                group_index = group.index.astype('str')
                source_x = group[col_list].sort_index().values.reshape(-1, ndim_x).astype('float32')
                source_y = group[col_list].sort_index().values.reshape(-1, ndim_y).astype('float32')

                slided_x = np.array([source_x[i:i + x_len] for i in range(0, len(source_x) - x_len - foresight - y_len + 1)])
                y_start_idx = x_len + foresight
                slided_y = np.array([source_y[i:i + y_len] for i in range(y_start_idx, len(source_y) - y_len+1)])
                slided_dt = np.array([group_index[i:i + y_len] for i in range(y_start_idx, len(source_y) - y_len + 1)])

                m_lst.append(slided_x)
                y_lst.append(slided_y)
                dt_lst.append(slided_dt)
           
        # [slided, day, x_len, nf]
        m_x = np.stack(m_lst, axis=1)
        m_y = np.stack(y_lst, axis=1)
        m_dt = np.stack(dt_lst, axis=1)
        m = np.concatenate([m_x, m_y], axis=2)
        
        cell = np.ones_like(m_dt)
        cell = cell * cell_id
        full_cell_lst.append(cell)
        full_m_lst.append(m)
        full_x_lst.append(m_x)
        full_y_lst.append(m_y)
        full_dt_lst.append(m_dt)
        
    # [slided, ncells, day, nsteps, nfeatures]
    print("after day window sliding")
    full_m = np.stack(full_m_lst, axis=1) # [slided, ncells, day, nsteps+1, nfeatures]
    full_x = np.stack(full_x_lst, axis=1) # [slided, ncells, day, nsteps, nfeatures]
    full_y = np.stack(full_y_lst, axis=1) # [slided, ncells, day, 1, nfeatures]
    full_dt = np.stack(full_dt_lst, axis=1)  # [slided, ncells, day, 1]
    full_cell = np.stack(full_cell_lst, axis=1)  # [slided, ncells, day, 1]

    for arg in [full_m, full_x, full_y, full_dt, full_cell]:
        print(arg.shape)
    
    # memory sliding for each cell
    x_start_day = mem_len+1
    total_m = []
    total_x = []
    total_y = []
    total_dt = []
    total_cell = []
    for i in range(x_start_day, full_m.shape[2]):
        total_m.append(full_m[:,:,i-mem_len:i,:,:])
        total_x.append(full_x[:, :, i, :, :])
        total_y.append(full_y[:, :, i, :, :])
        total_dt.append(full_dt[:, :, i, :])
        total_cell.append(full_cell[:, :, i, :])

    del full_m, full_x, full_y, full_cell, full_dt, full_m_lst, full_x_lst, full_y_lst, full_cell_lst, full_dt_lst

    # [slided, ncells, nsteps, nfeatures]
    total_x = np.concatenate(total_x, axis=0)
    total_y = np.concatenate(total_y, axis=0)
    total_dt = np.concatenate(total_dt, axis=0)
    total_cell = np.concatenate(total_cell, axis=0)

    # total_m: [slided, ncells, mteps * (nsteps+1), nf]
    total_m = np.concatenate(total_m, axis=0)
    total_m = np.reshape(total_m, [total_m.shape[0], total_m.shape[1], -1, total_m.shape[-1]])
    total_dt_cell0 = total_dt[:, 0, :]
    
    # squeezing
    total_y = np.squeeze(total_y)
    total_y = np.expand_dims(total_y, axis=1)     ## warning : only when using 1 cell !!
    total_dt_cell0 = np.squeeze(total_dt_cell0)
    total_dt = np.squeeze(total_dt)
    total_cell= np.squeeze(total_cell)
    print("after memory sliding")
    for arg in [total_x, total_y, total_dt, total_cell, total_m]:
        print(arg.shape)

    # decide test_ind for test tuples
    end_dt = pd.to_datetime(total_dt_cell0[-1])
    start_dt = end_dt - pd.Timedelta(days=test_len)
    if y_len > 1:
        start_dt = start_dt.astype('str')[0]
    test_ind = -1
    for i in range(len(total_dt_cell0)):
        if y_len == 1:
            d = total_dt_cell0[i]
        else:
            d = total_dt_cell0[i,0]

        if d == str(start_dt):
            test_ind = i+1
            break
    assert test_ind != -1
    print("test ind: {}".format(test_ind))

    tr_x = total_x[:test_ind]
    tr_y = total_y[:test_ind]
    tr_m = total_m[:test_ind]
    tr_c = total_cell[:test_ind]

    def _time_concat(arg):
        '''making shape [slided * ncells, nsteps, nfeatures]'''
        shapes = arg.shape
        right = [shapes[i] for i in range(2, len(shapes))]
        out = np.reshape(arg, [-1]+right)
        return out

    # [slided * ncells, nsteps, nf]
    tr_x = _time_concat(tr_x)
    tr_y = _time_concat(tr_y)
    tr_c = _time_concat(tr_c)
    tr_m = _time_concat(tr_m)

    te_x = _time_concat(total_x[test_ind:])
    te_y = _time_concat(total_y[test_ind:])
    te_m = _time_concat(total_m[test_ind:])
    te_c = _time_concat(total_cell[test_ind:])

    if seed :
        np.random.seed(seed)
    dev_len = int(len(tr_x) * dev_ratio)
    dev_ind = np.random.permutation(len(tr_x))[:dev_len]

    tr_ind = np.ones(len(tr_x), np.bool)
    tr_ind[dev_ind] = 0
    train_x, dev_x = tr_x[tr_ind], tr_x[dev_ind]
    print(tr_y.shape)
    train_y, dev_y = tr_y[tr_ind], tr_y[dev_ind]
    train_m, dev_m = tr_m[tr_ind], tr_m[dev_ind]
    train_c, dev_c = tr_c[tr_ind], tr_c[dev_ind]
    test_dt = total_dt_cell0[test_ind:]

    logger.info('X : {}, {}, {}'.format(train_x.shape, dev_x.shape, te_x.shape))
    logger.info('Y : {}, {}, {}'.format(train_y.shape, dev_y.shape, te_y.shape))
    logger.info('M : {}, {}, {}'.format(train_m.shape, dev_m.shape, te_m.shape))
    logger.info('C : {}, {}, {}'.format(train_c.shape, dev_c.shape, te_c.shape))
    logger.info('Test set start time: {}'.format(test_dt[0]))
    
    return train_x, dev_x, te_x, train_y, dev_y, te_y, train_m, dev_m, te_m, test_dt


def load_agg_data_all(data_path='../data/aggregated_data_5min_scaled.csv', ncells=20, test_len=7):

    data = pd.read_csv(data_path, index_col=0)
    data.index = pd.to_datetime(data.index)

    full_x = []

    for cell_id in range(ncells): # config       
        cell_data = data[data['CELL_NUM']==cell_id]
        
        # Find last test_len days to generate cell vectors
        end_dt = cell_data.index.date[-1]
        from datetime import timedelta
        start_dt = end_dt-timedelta(days=6)
        cell_x = cell_data[start_dt : end_dt+timedelta(days=1)]
        full_x.append(cell_x)

    full_x = np.stack(full_x, axis=0) # [ncells, t, d]
    full_x = np.expand_dims(full_x, axis=0)
    full_x = full_x[:, :, :, :-1]

    print("-----------------------------")              
    print("input shape: {}".format(full_x.shape))
    print("-----------------------------")

    return full_x


if __name__ == "__main__":
    full_x = load_agg_data_all()