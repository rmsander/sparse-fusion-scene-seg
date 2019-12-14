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
from pad_collate_fn import collate_fn
import matplotlib.pyplot as plt

# Native python imports
import copy
import pickle

##############################################################################################
# FLAGS
USE_CLI = False

# Number of classes
COMBINED = True
USE_CONMAT = False

# Get class dictionaries and flags
if COMBINED:
    F_CLASS_DICT_PKL = os.path.join("..", "data", "camera_lidar_semantic", "class_dictionary_COMBINED.pkl")
    F_CLASS_WEIGHTS_PKL = os.path.join("..", "data", "class_weights_COMBINED.pkl")
    NUM_CLASSES = 6

else:    
    F_CLASS_DICT_PKL = os.path.join("..", "data", "camera_lidar_semantic", "class_dictionary.pkl")
    F_CLASS_WEIGHTS_PKL = os.path.join("..", "data", "class_weights.pkl")
    NUM_CLASSES = 55
    
# Dataset size
MINI = False
FULL = True

# Set number of channels for NN
NUM_CHANNELS = 6 # 3 --> only x'y'z', 6 --> x'y'z' + r'g'b', 9 --> xyz + r'g'b' + x'y'z'

# For saving/logging files
EXPERIMENT_HEADER = "10000_points_6_classes_transfer_learning_2_EPOCHS_EACH"

# Class threshold for weighted cross-entropy
THRESHOLD = 1000

# Training Params
BATCH_SIZE = 4  # Don't go above 6 to avoid GPU from crashing

# Iterations (note this is per "stage")
EPOCHS = 2 

# Not currently using pre-trained
PRETRAIN = None  # If used, set later

# GPU Settings
GPU = '0'
MULTI_GPU = None

# Learning
LEARNING_RATE = 0.001
DECAY_RATE = 1e-4
OPTIMIZER = 'Adam'
MODEL_NAME = 'pointnet2'

if MINI:
    dir_above = os.path.abspath("..")
    DATA_PATH = os.path.join(dir_above,"data","dataset_pc_labels_camera_start_0_stop_2000.pkl")

elif FULL:
    dir_above = os.path.abspath("..")
    if COMBINED:
        DATA_PATH = os.path.join(dir_above,"data","dataset_pc_labels_camera_start_0_stop_10000_COMBINED_CLASSES.pkl")
    else:
        DATA_PATH = os.path.join(dir_above,"data","dataset_pc_labels_camera_start_0_stop_10000.pkl")
    
##############################################################################################

# Check GPU
print("Current device: {}".format(torch.cuda.current_device()))
print("Number of devices: {}".format(torch.cuda.device_count()))
print("Device name: {}".format(torch.cuda.get_device_name(0)))
print("Cuda available? {}".format(torch.cuda.is_available()))

def parse_args():
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batchsize', type=int, default=1, help='input batch size')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs for training')
    parser.add_argument('--pretrain', type=str, default=None,help='whether use pretrain model')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for training')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--optimizer', type=str, default='Adam', help='type of optimizer')
    parser.add_argument('--multi_gpu', type=str, default=None, help='whether use multi gpu training')
    parser.add_argument('--model_name', type=str, default='pointnet2', help='Name of model')
    parser.add_argument('--data_path', type=str, default='../data/dataset_tensor_normalized_GENERAL.pkl', help='Filepath to pickled dataset')

    return parser.parse_args()

def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1

def main(args, stage=None, pretrain_model_path=None, e_init=0, e_final=2):
    
    
    # Now load pickle labels mapping file
    class_dict_fname = F_CLASS_DICT_PKL
    print(class_dict_fname)
    with open(class_dict_fname, "rb") as f:
        class_dict = pickle.load(f)
        f.close()

    print("CLASS DICT: {}".format(class_dict))

    # Use to get numeric classes --> semantic classes
    seg_classes = class_dict
    seg_label_to_cat = {}
    for i, cat in enumerate(seg_classes.values()):
        seg_label_to_cat[i] = cat

   
    print('SEG LABEL',seg_label_to_cat)
    
    # First load class weights file
    with open(F_CLASS_WEIGHTS_PKL, "rb") as f:
        class_weights = pickle.load(f)
        f.close()
    print('SEG CLASSES',seg_classes)
    COUNTS = np.array([class_weights[key] for key in list(class_weights.keys())])
    weight_normalizer = np.max(COUNTS)

    weights = []
    for count in COUNTS:
        if count != 0:
            weights.append(weight_normalizer/count)
        else:
            weights.append(0)

    # Threshold weights
    WEIGHTS_NP = np.array(weights)
    WEIGHTS_NP[WEIGHTS_NP > THRESHOLD] = THRESHOLD
    
    print("WEIGHTS ARE: {}".format(WEIGHTS_NP))

    # Convert to pytorch tensor
    weights = torch.from_numpy(WEIGHTS_NP.astype('float32'))
    
    if USE_CLI:
        gpu = args.gpu
        multi_gpu = args.multi_gpu
        batch_size = args.batch_size
        model_name = args.model_name
        optimizer = args.optimizer
        learning_rate = args.learning_rate
        pretrain = args.pretrain
        multi_gpu = args.multi_gpu
        batchsize = args.batchsize
        decay_rate = args.decay_rate
        epochs = args.epochs
    else:
        gpu = GPU
        multi_gpu = MULTI_GPU
        batch_size = BATCH_SIZE
        model_name = MODEL_NAME
        optimizer = OPTIMIZER
        learning_rate = LEARNING_RATE
        pretrain = PRETRAIN
        multi_gpu = MULTI_GPU
        batchsize = BATCH_SIZE
        decay_rate = DECAY_RATE
        epochs = EPOCHS
    
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu if multi_gpu is None else '0,1,2,3'
    '''CREATE DIR'''
    experiment_dir = Path('./experiment/{}'.format(EXPERIMENT_HEADER))
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) +'/%sSemSeg-'%model_name+ str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    if USE_CLI:
        args = parse_args()
        logger = logging.getLogger(model_name)
    else:
        logger = logging.getLogger(MODEL_NAME)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if USE_CLI:
        file_handler = logging.FileHandler(str(log_dir) + '/train_%s_semseg.txt'%args.model_name)
    else:
        file_handler = logging.FileHandler(str(log_dir) + '/train_%s_semseg.txt'%MODEL_NAME)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('---------------------------------------------------TRANING---------------------------------------------------')
    if USE_CLI:
        logger.info('PARAMETER ...')
        logger.info(args)
    print('Load data...')
    #train_data, train_label, test_data, test_label = recognize_all_data(test_area = 5)
    
    # Now pickle our dataset
    if USE_CLI:
      f_in = args.data_path
    else:
      f_in = DATA_PATH
   
    # Now pickle file
    with open(f_in, "rb") as f:
        DATA = pickle.load(f)
        f.close()
    
    # Now, we need to adjust labels according to our stage
    labels_fpath = os.path.join("..", "data", "stage_{}_labels.pkl".format(stage))
    
    # Pickle this stage converter
    with open(labels_fpath, "rb") as f:
        labels = pickle.load(f)
        print(type(labels), len(labels))
        f.close()
    
    # Now let's convert labels appropriately
    keys = list(DATA.keys())
    for key in keys:
        DATA[key]['labels'] = labels[key]
    
    random_seed = 42
    indices = [i for i in range(len(list(DATA.keys())))]
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    TEST_SPLIT = 0.2
    test_index = int(np.floor(TEST_SPLIT*len(list(DATA.keys()))))
    print("val index is: {}".format(test_index))
    train_indices, test_indices = indices[test_index:], indices[:test_index]
    if USE_CLI:
        print("LEN TRAIN: {}, LEN TEST: {}, EPOCHS: {}, OPTIMIZER: {}, DECAY_RATE: {}, LEARNING RATE: {}, \
        DATA PATH: {}".format(len(train_indices), len(test_indices), epochs, args.optimizer, args.decay_rate, \
                              args.learning_rate, args.data_path))
    else:
        print("LEN TRAIN: {}, LEN TEST: {}, EPOCHS: {}, OPTIMIZER: {}, DECAY_RATE: {}, LEARNING RATE: {}, \
        DATA PATH: {}".format(len(train_indices), len(test_indices), e_final-e_init, OPTIMIZER, DECAY_RATE, \
                              LEARNING_RATE, DATA_PATH))
                             
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    print("INTERSECTION OF TRAIN/TEST (should be 0): {}".format(len(set(train_indices).intersection(set(test_indices)))))
    
    # Training dataset
    dataset = A2D2DataLoader(DATA)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize,
                                                 shuffle=False, sampler=train_sampler, collate_fn=collate_fn)
    # Test dataset
    test_dataset = A2D2DataLoader(DATA)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize,
                                                 shuffle=False, sampler=test_sampler, collate_fn=collate_fn)
    
    num_classes = NUM_CLASSES

    blue = lambda x: '\033[94m' + x + '\033[0m'
    model = PointNet2SemSeg(num_classes) if model_name == 'pointnet2' else PointNetSeg(num_classes,feature_transform=True,semseg = True)

    if pretrain_model_path is not None:
        model.load_state_dict(torch.load(pretrain_model_path))
        print('load model %s'%pretrain_model_path)
        logger.info('load model %s'%pretrain_model_path)
    else:
        print('Training from scratch')
        logger.info('Training from scratch')
    #pretrain_var = pretrain
    #init_epoch = int(pretrain_var[-14:-11]) if pretrain is not None else 0
    init_epoch = e_init
            
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
    graph_losses = []
    steps = []
    step = 0
    print("NUMBER OF EPOCHS IS: {}".format(e_final-e_init))
    for epoch in range(e_init, e_final):
        scheduler.step()
        lr = max(optimizer.param_groups[0]['lr'],LEARNING_RATE_CLIP)
        print('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        counter = 0
        # Init confusion matrix
        if USE_CONMAT:
            conf_matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES)
        for points, targets in tqdm(dataloader):
        #for points, target in tqdm(dataloader):
            #points, target = data
            points, targets = Variable(points.float()), Variable(targets.long())
            points = points.transpose(2, 1)
            points, targets = points.cuda(), targets.cuda()
            weights = weights.cuda()
            optimizer.zero_grad()  # REMOVE gradients
            model = model.train()
            if model_name == 'pointnet':
                pred, trans_feat = model(points)
            else:
                pred = model(points[:,:3,:],points[:,3:,:])  # Channels: xyz_norm (first 3) | rgb_norm (second three)
                #pred = model(points)
            if USE_CONMAT:
                conf_matrix = confusion_matrix(pred, targets, conf_matrix)
            pred = pred.contiguous().view(-1, num_classes)
            targets = targets.view(-1, 1)[:, 0]
            loss = F.nll_loss(pred, targets, weight=weights) # Add class weights from dataset
            if model_name == 'pointnet':
                loss += feature_transform_reguliarzer(trans_feat) * 0.001
            graph_losses.append(loss.cpu().data.numpy()) 
            steps.append(step)
            if counter % 100 == 0:
                print("LOSS IS: {}".format(loss.cpu().data.numpy()))
            #print((loss.cpu().data.numpy()))
            history['loss'].append(loss.cpu().data.numpy())
            loss.backward()
            optimizer.step()
            counter += 1
            step += 1
            #if counter > 3:
            #     break
        if USE_CONMAT:
            print("CONFUSION MATRIX: \n {}".format(conf_matrix))
        pointnet2 = model_name == 'pointnet2'
        test_metrics, test_hist_acc, cat_mean_iou = test_semseg(model.eval(), testdataloader, seg_label_to_cat,\
                                                                num_classes = num_classes,pointnet2=pointnet2)
        mean_iou = np.mean(cat_mean_iou)
        print('Epoch %d  %s accuracy: %f  meanIOU: %f' % (
                 epoch, blue('test'), test_metrics['accuracy'],mean_iou))
        logger.info('Epoch %d  %s accuracy: %f  meanIOU: %f' % (
                 epoch, 'test', test_metrics['accuracy'],mean_iou))
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
            print("HERE")
            save_path = '%s/%s_%.3d_%.4f_stage_%s.pth' % (checkpoints_dir,model_name, epoch, best_acc, stage)
            torch.save(model.state_dict(), save_path)
            logger.info(cat_mean_iou)
            logger.info('Save model..')
            print('Save model..')
            print(cat_mean_iou)#
        if mean_iou > best_meaniou:
            best_meaniou = mean_iou
        print('Best accuracy is: %.5f'%best_acc)
        logger.info('Best accuracy is: %.5f'%best_acc)
        print('Best meanIOU is: %.5f'%best_meaniou)
        logger.info('Best meanIOU is: %.5f'%best_meaniou)
        if USE_CONMAT:
            logger.info('Confusion matrix is: \n {}'.format(conf_matrix))
        
        # Plot loss vs. steps
        plt.plot(steps, graph_losses)
        plt.xlabel("Batched Steps (Batch Size = {}".format(batch_size))
        plt.ylabel("Multiclass NLL Loss")
        plt.title("NLL Loss vs. Number of Batched Steps")
        
        # Make directory for loss and other plots
        graphs_dir = os.path.join(experiment_dir, "graphs")
        os.makedirs(graphs_dir, exist_ok=True)
                
        # Save and close figure
        plt.savefig(os.path.join(graphs_dir,"losses.png"))
        plt.clf()
    return save_path  

if __name__ == '__main__':
    # Basic idea - train on the model trained in the previous stage
    args = parse_args()
    pretrained_model_paths = []
    counter = 0
    for STAGE in ["1", "2", "3", "4", "5"]:
        if counter >= 1: 
            pretrain_path = pretrained_model_paths[counter-1]
        else:
            pretrain_path = None
        pretrained_model_paths.append(main(args, stage=STAGE, pretrain_model_path=pretrain_path, e_init=2*counter, \
                                           e_final=2*counter + EPOCHS))
        counter += 1
