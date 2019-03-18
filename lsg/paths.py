import os

current_file = os.path.abspath(__file__)

ROOT_PATH = os.path.dirname(os.path.dirname(current_file))

CONFIG_DIR = os.path.join(ROOT_PATH, 'config')
DEFAULT_CONFIG_PATH = os.path.join(CONFIG_DIR, 'default.ini')

DATA_DIR = os.path.join(ROOT_PATH, 'data')
RESPONSES_PATH = os.path.join(DATA_DIR, 'ks_data.csv')
