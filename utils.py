import logging
import logging.handlers
import os
import datetime

def get_logger(log_path='logs/'):
    """
    :param log_path
    :return: logger instance
    """
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    logger = logging.getLogger()
    
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s %(message)s', date_format)
    i = 0
    today = datetime.datetime.now()
    name = 'log-'+today.strftime('%Y%m%d')+'-'+'%02d'%i+'.log'
    while os.path.exists(log_path+name):
        i += 1
        name = 'log-'+today.strftime('%Y%m%d')+'-'+'%02d'%i+'.log'
    
    fileHandler = logging.FileHandler(os.path.join(log_path+name))
    streamHandler = logging.StreamHandler()
    
    fileHandler.setFormatter(formatter)
    streamHandler.setFormatter(formatter)
    
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)
    
    logger.setLevel(logging.INFO)
    logger.info('Writing logs at {}'.format(os.path.join(log_path+name)))
    return logger, log_path+name

def make_date_dir(path):
    """
    :param path
    :return: os.path.join(path+date_dir)
    """
    if not os.path.exists(path):
        os.mkdir(path)
    i = 0
    today = datetime.datetime.now()
    name = today.strftime('%Y%m%d')+'-'+'%02d' % i
    while os.path.exists(os.path.join(path + name)):
        i += 1
        name = today.strftime('%Y%m%d')+'-'+'%02d' % i
    os.mkdir(os.path.join(path + name))
    return os.path.join(path + name)

def find_latest_dir(path):
    dirs = os.listdir(path)
    dirs_splited = list(map(lambda x:x.split("-"), dirs))
    
    # find latest date
    dirs_date = [int(dir[0]) for dir in dirs_splited]
    dirs_date.sort()
    latest_date = dirs_date[-1]
    
    # find latest num in lastest date
    dirs_num = [int(dir[1]) for dir in dirs_splited if int(dir[0]) == latest_date]
    dirs_num.sort()
    latest_num = dirs_num[-1]
    latest_dir = str(latest_date) + '-' + '%02d' % latest_num

    return os.path.join(path + latest_dir)
    
# if __name__=="__main__":
#     make_date_dir('./log')
#     get_logger('./log')
    