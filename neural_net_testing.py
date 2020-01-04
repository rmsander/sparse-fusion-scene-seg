import argparse
import os
import torch
import numpy as np
import torch.nn.parallel
import torch.utils.data
from collections import defaultdict
from torch.autograd import Variable
from data_utils.A2D2DataLoader import A2D2DataLoader
import torch.nn.functional as F
import datetime
import logging
from pathlib import Path
from utils import test_semseg
from tqdm import tqdm
from model.pointnet2 import PointNet2SemSeg
from model.pointnet import PointNetSeg, feature_transform_reguliarzer
from torch.utils.data.sampler import SubsetRandomSampler
import pickle

################################################################################
# FLAGS
USE_CLI = False
F_CLASS_DICT_PKL = os.path.join("..", "data", "camera_lidar_semantic",
                                "class_dictionary.pkl")
MINI = True
FULL = False
TRANSFER_LEARNING_GENERAL = False
TRANSFER_LEARNING_TARGET = False
NUM_CLASSES = 55
NUM_CHANNELS = 6  # 3 --> only x'y'z', 6 --> x'y'z' + r'g'b', 9 --> xyz + 
# r'g'b' + x'y'z'

# Training Params
BATCH_SIZE = 12
EPOCH = 10
if TRANSFER_LEARNING_TARGET:
    PRETRAIN = True
else:
    PRETRAIn = False
GPU = '0'
LEARNING_RATE = 0.001
DECAY_RATE = 1e-4
OPTIMIZER = 'Adam'
MULTI_GPU = None
MODEL_NAME = 'pointnet2'

if MINI:
    DATA_PATH = os.path.join(os.getcwd(), "data",
                             "PROCESSED_mini_dataset_norm_tensor.pkl"

    ############################################################################

    # Now load pickle file
    with open(f_class_dict_pkl, "rb") as f:
        class_dict, rgb_dict = pickle.load(f)
    f.close()

    seg_classes = class_dict
    seg_label_to_cat = {}
    for i, cat in enumerate(seg_classes.values()):
        seg_label_to_cat[i] = cat

    print(seg_label_to_cat)

def parse_args():
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batchsize', type=int, default=1,
                        help='input batch size')
    parser.add_argument('--workers', type=int, default=0,
                        help='number of data loading workers')
    parser.add_argument('--epoch', type=int, default=5,
                        help='number of epochs for training')
    parser.add_argument('--pretrain', type=str, default=None,
                        help='whether use pretrain model')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate for training')
    parser.add_argument('--decay_rate', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='type of optimizer')
    parser.add_argument('--multi_gpu', type=str, default=None,
                        help='whether use multi gpu training')
    parser.add_argument('--model_name', type=str, default='pointnet2',
                        help='Name of model')
    parser.add_argument('--data_path', type=str,
                        default='../data/dataset_tensor_normalized_GENERAL.pkl',
                        help='Filepath to pickled dataset')

    return parser.parse_args()


def main(args):
    os.environ[
        "CUDA_VISIBLE_DEVICES"] = args.gpu if args.multi_gpu is None else '0,1,2,3'
    '''CREATE DIR'''
    experiment_dir = Path('./experiment_minus10classes/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/%sSemSeg-' % args.model_name + str(
        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    if USE_CLI:
        args = parse_args()
        logger = logging.getLogger(args.model_name)
    else:
        logger = logging.getLogger(MODEL_NAME)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if USE_CLI:
        file_handler = logging.FileHandler(
            str(log_dir) + '/train_%s_semseg.txt' % args.model_name)
    else:
        file_handler = logging.FileHandler(
            str(log_dir) + '/train_%s_semseg.txt' % MODEL_NAME)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(
        '---------------------------------------------------TRANING'
        '---------------------------------------------------')
    if USE_CLI:
        logger.info('PARAMETER ...')
        logger.info(args)
    print('Load data...')
    # train_data, train_label, test_data, test_label = recognize_all_data(
    # test_area = 5)

    # Now pickle our dataset
    if USE_CLI:
        f_in = args.data_path
    else:
        f_in = DATA_PATH

    # Now pickle file
    with open(f_data, "rb") as f:
        DATA = pickle.load(f)
        f.close()
    random_seed = 42
    indices = [i for i in range(len(list(DATA.keys())))]
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    TEST_SPLIT = 0.2
    test_index = int(np.floor(TEST_SPLIT * len(list(DATA.keys()))))
    print("val index is: {}".format(test_index))
    train_indices, test_indices = indices[test_index:], indices[:test_index]
    if USE_CLI:
        print("LEN TRAIN: {}, LEN TEST: {}, EPOCHS: {}, OPTIMIZER: {}, "
              "DECAY_RATE: {}, LEARNING RATE: {}, \
        DATA PATH: {}".format(len(train_indices), len(test_indices), args.epoch,
                              args.optimizer, args.decay_rate, \
                              args.learning_rate, args.data_path))
    else:
        print("LEN TRAIN: {}, LEN TEST: {}, EPOCHS: {}, OPTIMIZER: {}, "
              "DECAY_RATE: {}, LEARNING RATE: {}, \
        DATA PATH: {}".format(len(train_indices), len(test_indices), EPOCH,
                              OPTIMIZER, DECAY_RATE, \
                              LEARNING_RATE, DATA_PATH))

    if USE_CLI:
        batch_size = args.batch_size
        model_name = args.model_name
        optimizer = args.optimizer
        learning_rate = args.learning_rate
        pretrain = args.pretrain
        multi_gpu = args.multi_gpu
    else:
        batch_size = BATCH_SIZE
        model_name = MODEL_NAME
        optimizer = OPTIMIZER
        learning_rate = LEARNING_RATE
        pretrain = PRETRAIN
        multi_gpu = MULTI_GPU

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    dataset = A2D2DataLoader(DATA)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize,
                                             shuffle=False,
                                             num_workers=int(args.workers),
                                             sampler=train_sampler)
    test_dataset = A2D2DataLoader(DATA)
    testdataloader = torch.utils.data.DataLoader(test_dataset,
                                                 batch_size=batchsize,
                                                 shuffle=False,
                                                 num_workers=int(args.workers),
                                                 sampler=test_sampler)

    num_classes = NUM_CLASSES


if optimizer == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
elif optimizer == 'Adam':
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=decay_rate
    )
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
LEARNING_RATE_CLIP = 1e-5

'''GPU selection and multi-GPU'''
if multi_gpu is not None:
    device_ids = [int(x) for x in multi_gpu.split(',')]
    torch.backends.cudnn.benchmark = True
    model.cuda(device_ids[0])
    model = torch.nn.DataParallel(model, device_ids=device_ids)
else:
    model.cuda()

history = defaultdict(lambda: list())
best_acc = 0
best_meaniou = 0

dp_0 = dataloader.get_item(0)

if __name__ == '__main__':
    args = parse_args()
    main(args)
