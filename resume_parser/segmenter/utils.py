import datetime
import logging
import os
import subprocess
import sys

def _get_git_revision_hash() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'])\
            .decode('ascii').strip()
    except:
        return 'no_git'


def _get_git_revision_short_hash() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])\
            .decode('ascii').strip()
    except:
        return 'no_git'


def _generate_output_dirname() -> str:
    git_hash = _get_git_revision_short_hash()
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    return now + '_' + git_hash


def generate_output_dirpath(module_path: str, output_rootname: str='output'):
    module_dirpath = os.path.dirname(os.path.abspath(module_path))
    output_dirname = _generate_output_dirname()
    output_dirpath = f'{module_dirpath}/{output_rootname}/{output_dirname}'
    if not os.path.exists(output_dirname):
        os.makedirs(output_dirpath)
    return output_dirpath


def get_logger(filename: str) -> logging.Logger:
    log_format = '%(asctime)s [%(levelname)-5.5s] %(message)s'
    log_formatter = logging.Formatter(log_format)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    return logger
