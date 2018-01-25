import logging
import os


def add_logger(log_dir, prefix, remove_previous_log=True):
    check_log_dir(log_dir)

    logging.basicConfig(format='%(asctime)s-%(message)s', level=logging.DEBUG)
    logger = logging.getLogger()
    log_file = os.path.join(log_dir, '%s.log' % prefix)

    if remove_previous_log:
        remove_log(log_file)

    hdlr = logging.FileHandler(log_file)
    logger_formatter = logging.Formatter('%(asctime)s-%(message)s')
    hdlr.setFormatter(logger_formatter)
    logger.addHandler(hdlr)
    return logger


def check_ckpt(ckpt_dir, prefix):
    if not os.path.exists(os.path.join(ckpt_dir, prefix)):
        os.makedirs(os.path.join(ckpt_dir, prefix))


def check_log_dir(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


def remove_log(log_file):
    if os.path.exists(log_file):
        os.remove(log_file)