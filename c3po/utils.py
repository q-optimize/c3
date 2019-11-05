import time
import os


def log_setup(data_path):
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    pwd = data_path + time.strftime(
        "%Y_%m_%d_T_%H_%M_%S", time.localtime()
    )
    os.makedirs(pwd)
    recent = data_path + 'recent'
    if os.path.isdir(recent):
        os.remove(recent)
    os.symlink(pwd, recent)
    return pwd + '/'
