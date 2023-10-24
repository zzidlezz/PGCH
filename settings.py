import logging
import time
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str, default='flickr', help='Dataset name: WIKI/flickr/coco/nuswide')
parser.add_argument('--bits', type=int, default=128, help='16/32/64/128')
parser.add_argument('--label_class', type=int, default=24, help='10/255/24/80/21')
args = parser.parse_args()

if args.dataname == 'flickr':
    DIR = '../Data/raw_mir.mat'
elif args.dataname == 'WIKI':
    DIR = '../Data/WIKIPEDIA.mat'
elif args.dataname == 'nuswide': # select sample from each dataset
    DIR = '../Data/raw_nus.mat'
elif args.dataname == 'coco':
    DIR = '../Data/coco_cnn.mat'
else:
    print('Dataname Error!')
    DIR = ''


label_class = args.label_class
DATASET_NAME = args.dataname
CODE_LEN = args.bits 
NUM_EPOCH = 40
EVAL_INTERVAL =20
BATCH_SIZE = 128
MOMENTUM = 0.9
WEIGHT_DECAY = 5 * 10 ** -4
NUM_WORKERS = 0
EPOCH_INTERVAL = 4

MODEL_DIR = './checkpoint'
EVAL = False # EVAL = True: just test, EVAL = False: train and eval

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
now = time.strftime("%Y%m%d%H%M%S",time.localtime(time.time())) 
log_name = now + '_log.txt'
log_dir = './log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
txt_log = logging.FileHandler(os.path.join(log_dir, log_name))
txt_log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
txt_log.setFormatter(formatter)
logger.addHandler(txt_log)

stream_log = logging.StreamHandler()
stream_log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_log.setFormatter(formatter)
logger.addHandler(stream_log)
