import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2 as cv
import json
import pickle
import pptk
from pyntcloud import PyntCloud


# Function from A2D2 tutorial used for importing images
def extract_image_file_name_from_lidar_file_name(file_name_lidar):
    file_name_image = file_name_lidar.split('/')
    file_name_image = file_name_image[-1].split('.')[0]
    file_name_image = file_name_image.split('_')
    file_name_image = file_name_image[0] + '_' + 'camera_' + file_name_image[
        2] + '_' + file_name_image[3] + '.png'

    return file_name_image


# Make a dictionary from hex --> RGB for classes
def hex2rgb(hex_id):
    hex_id = hex_id[1:]
    return tuple(int(hex_id[i:i + 2], 16) for i in (0, 2, 4))


def get_classes_to_ids(
        json_file=os.path.join(os.getcwd(), "data", "camera_lidar_semantic",
                               "class_list.json")):
    with open(json_file) as file:
        hex_dict = json.load(file)
        file.close()

    class_dict = {}
    rgb_dict = {}
    inverse_rgb_dict = {}

    j = 0
    for key in list(hex_dict.keys()):
        class_dict[j] = hex_dict[key]
        rgb_dict[hex2rgb(key)] = j
        inverse_rgb_dict[j] = hex2rgb(key)
        j += 1

    # Pickle these files once we're ready
    dict_save = os.path.join(os.getcwd(), "data", "camera_lidar_semantic",
                             "class_dictionary.pkl")
    with open(dict_save, "wb") as f:
        pickle.dump([class_dict, rgb_dict], f)
        f.close()
    print("FILES PICKLED")
    return class_dict, rgb_dict


# Function for loading and merging data
def merge_data(start_index=None, stop_index=None):
    # Get current working directory and data folder
    CWD = os.getcwd()
    data_folder = os.path.join(CWD, "data", "camera_lidar_semantic")

    # Get classes <--> labels dictionaries
    class_dict, rgb_dict = get_classes_to_ids()

    # Get folders which we will recursively scrape from to aggregate data
    sub_folders = os.listdir(data_folder)
    sub_folders_all_dir = [sub_folder for sub_folder in sub_folders if
                           os.path.isdir(os.path.join(data_folder, sub_folder))]
    sub_folders_filtered = [sub_folder for sub_folder in sub_folders_all_dir if
                            "lidar" in os.listdir(
                                os.path.join(data_folder, sub_folder)) and
                            "label" in os.listdir(
                                os.path.join(data_folder, sub_folder)) and
                            "camera" in os.listdir(
                                os.path.join(data_folder, sub_folder))]

    # Get IDs for iterating through each (image, label, point cloud) triple
    IDs = set()
    for sub_folder in sub_folders_filtered:
        sub_data_dir = os.path.join(data_folder, sub_folder, "camera",
                                    "cam_front_center")
        files = os.listdir(sub_data_dir)
        file_IDs = [file.split("_")[0] + "_" + file.split("_")[-1].split(".")[0]
                    for file in files]
        IDs.update(file_IDs)

    print("Number of IDs in dataset: {}".format(len(list(IDs))))

    Dataset = {}
    i = 0

    if start_index is None:
        start_index = 0
    if stop_index is None:
        stop_index = len(IDs)
    print("START index is {}, STOP index is {}".format(start_index, stop_index))
    # Iterate over point clouds, rgb, and labels
    for ID in list(IDs)[start_index:stop_index]:
        if i % 100 == 0:
            print("Iterated through {} files".format(i))
        # Get ID information
        ID_split = ID.split("_")
        ID_split[-1] = ID_split[-1].split(".")[0]

        # Get directories for each individual set of images, labels, 
        # and point clouds
        ind_data_dir = os.path.join(data_folder,
                                    ID_split[0][:-6] + "_" + ID_split[0][-6:])
        ind_pc_dir = os.path.join(ind_data_dir, "lidar", "cam_front_center")
        ind_label_dir = os.path.join(ind_data_dir, "label", "cam_front_center")
        ind_camera_dir = os.path.join(ind_data_dir, "camera",
                                      "cam_front_center")

        # Get filenames for images, labels, and point clouds
        pc_file_name = ID_split[0] + "_lidar_frontcenter_" + ID_split[
            1] + ".npz"
        label_file_name = ID_split[0] + "_label_frontcenter_" + ID_split[
            1] + ".png"
        camera_file_name = ID_split[0] + "_camera_frontcenter_" + ID_split[
            1] + ".png"

        # Load images, labels, and point clouds
        A_label = cv.imread(os.path.join(ind_label_dir, label_file_name))
        A_camera = cv.imread(os.path.join(ind_camera_dir, camera_file_name))
        pc = np.load(os.path.join(ind_pc_dir, pc_file_name), allow_pickle=True)

        # Give labels to points in PC equal to [row, index] these points map 
        # to in label image space
        keys = list(pc.keys())
        if 'row' in keys and 'col' in keys and 'points' in keys:
            rows, cols = pc['row'], pc['col']
            classes = []
            rgb = []
            N = len(rows)
            for row, col in zip(rows, cols):  # O(n)
                rgb_data = A_camera[int(row), int(col), :][::-1]
                rgb_label = tuple(A_label[int(row), int(col), :])[::-1]
                classes.append(rgb_dict[rgb_label])  # O(1)
                rgb.append(rgb_data)
            classes = np.array(classes)
            rgb = np.array(rgb).reshape((N, 3))
            Dataset[ID] = {'points': pc['points'], 'labels': classes,
                           'rgb': rgb}
        i += 1
        pickle_outfile = os.path.join(os.getcwd(), "data",
                                      "dataset_pc_labels_camera_{"
                                      "}_ids.pkl".format(
                                          i))
    pickle_outfile = os.path.join(os.getcwd(), "data",
                                  "dataset_pc_labels_camera_start_{}_stop_{"
                                  "}.pkl".format(
                                      start_index, stop_index))
    with open(pickle_outfile, "wb") as f:
        pickle.dump(Dataset, f)
        f.close()
    return Dataset


def test_rgb_id_consistency(Dataset):
    dataset_pickle = os.path.join(os.getcwd(), "data",
                                  "dataset_pc_labels_camera.pkl")
    with open(dataset_pickle, "wb") as f:
        pickle.dump(Dataset, f)
        f.close()

    with open(dataset_pickle, "rb") as f:
        D = pickle.load(f)
        f.close()

    keys = list(D.keys())[:len(keys)]
    minidataset = {key: D[key] for key in keys}
    for key in keys:
        A_img = D[key]['rgb']
        A_label = D[key]['rgb_labels']
        print(A_img.shape, A_label.shape)
        plt.imshow(A_img)
        plt.show()
        plt.clf()
        plt.imshow(A_label)
        plt.show()
        plt.clf()


def test_id_consistency(D, index):
    # Define and get IDs
    ID = list(D.keys())[index]
    labels_pc = D[ID]["labels"]

    # Current working directory and data directory
    CWD = os.getcwd()
    data_folder = os.path.join(CWD, "data", "camera_lidar_semantic")

    # Get classes <--> labels dictionaries
    class_dict, rgb_dict = get_classes_to_ids()
    inverse_rgb_dict = {rgb_dict[key]: key for key in list(rgb_dict.keys())}

    # Split ID string for parsing below
    ID_split = ID.split("_")
    ID_split[-1] = ID_split[-1].split(".")[0]
    print("ID is: {}, ID_split is: {}".format(ID, ID_split))

    # Get paths for image, labels, and point clouds
    ind_data_dir = os.path.join(data_folder,
                                ID_split[0][:-6] + "_" + ID_split[0][-6:])
    ind_pc_dir = os.path.join(ind_data_dir, "lidar", "cam_front_center")
    ind_label_dir = os.path.join(ind_data_dir, "label", "cam_front_center")
    ind_camera_dir = os.path.join(ind_data_dir, "camera", "cam_front_center")

    # Get file names for image, labels, and point clouds
    pc_file_name = ID_split[0] + "_lidar_frontcenter_" + ID_split[1] + ".npz"
    label_file_name = ID_split[0] + "_label_frontcenter_" + ID_split[1] + ".png"
    camera_file_name = ID_split[0] + "_camera_frontcenter_" + ID_split[
        1] + ".png"

    # Load data for image, labels, and point clouds
    A_label = cv.imread(os.path.join(ind_label_dir, label_file_name))
    pc = np.load(os.path.join(ind_pc_dir, pc_file_name), allow_pickle=True)
    A_camera = cv.imread(os.path.join(ind_camera_dir, camera_file_name))

    keys = list(pc.keys())
    ID_split = ID.split("_")
    ID_split[-1] = ID_split[-1].split(".")[0]

    # Give labels to points in PC equal to [row, index] these points map to 
    # in label image space 
    label_file_name = ID_split[0] + "_label_frontcenter_" + ID_split[1] + ".png"
    Img = os.path.join(ind_label_dir, label_file_name)
    rows, cols = pc['row'], pc['col']
    classes = []
    test_img = np.zeros(A_label.shape)
    for row, col, label in zip(rows, cols, labels_pc):  # O(n)
        rgb_data = A_camera[int(row), int(col), :][::-1]
        rgb_label = tuple(A_label[int(row), int(col), :])[::-1]
        classes.append(rgb_dict[rgb_label])  # O(1)
        # test_img[int(row),int(col),:] = np.array(inverse_rgb_dict[label])
        test_img[int(row), int(col), :] = rgb_data
    print("TEST if zero (zero indicates correct behavior): {}".format(
        np.sum(np.subtract(labels_pc, np.array(classes)))))

    # Plot to qualitatively confirm results
    plt.imshow(A_camera)
    plt.show()
    plt.clf()
    plt.imshow(A_label)
    plt.show()
    plt.clf()
    plt.imshow(test_img)
    plt.show()
    plt.clf()


# Iterate through all GT semantic images, and find all unique RGB triplets 
# for class values.
def make_semantic_mask_dict(seg_img_dir):
    imgs = os.listdir(seg_img_dir)
    color_triplets = set()
    for img in imgs:
        A_label = cv.imread(img)
        img_triplets = np.unique(img.reshape(-1, img.shape[2]), axis=0)
        color_triplets.update(img_triplets)


# Function for pickling files
def pickle_file(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
        f.close()
    print("Object pickled to file located at {}".format(path))


# Function for loading pickle files
def load_pickle_file(path):
    with open(path, "rb") as f:
        dataset = pickle.load(f)
        f.close()
    print("Object loaded from file located at {}".format(path))
    return dataset


# Finally, let's create a function for merging all of this data, and pickle 
# this file to be saved
def create_and_export_data(start_indices=[i * 2000 for i in range(0, 14)],
                           stop_indices=[i * 2000 for i in range(1, 15)]):
    for start, stop in zip(start_indices, stop_indices):
        Dataset = merge_data(start_index=start, stop_index=stop)
        print("Processed indices from {} to {}".format(start, stop))


# Call the function above, checkpointing on 2 IDs (sanity check), 2000 IDs, 
# 5000 IDs, and 10000 IDs (and full)

# create_and_export_data()
# Should be able to do everything at once
merge_data(start_index=0, stop_index=2000)


def test():
    # Now load dummy dataset
    pickle_outfile = os.path.join(os.getcwd(), "data",
                                  "dataset_pc_labels_camera_start_{}_stop_{"
                                  "}.pkl".format(
                                      0, 3))

    with open(pickle_outfile, "rb") as f:
        A = pickle.load(f)
        f.close()

    keys = list(A.keys())
    for key in keys:
        CWD = os.getcwd()
        data_folder = os.path.join(CWD, "data", "camera_lidar_semantic")
        ID_split = key.split("_")
        ID_split[-1] = ID_split[-1].split(".")[0]

        # Get paths for image, labels, and point clouds
        ind_data_dir = os.path.join(data_folder,
                                    ID_split[0][:-6] + "_" + ID_split[0][-6:])
        RGB = A[key]['rgb']
        ind_camera_dir = os.path.join(ind_data_dir, "camera",
                                      "cam_front_center")
        ind_pc_dir = os.path.join(ind_data_dir, "lidar", "cam_front_center")

        pc_file_name = ID_split[0] + "_lidar_frontcenter_" + \
                       ID_split[1] + ".npz"
        label_file_name = ID_split[0] + "_label_frontcenter_" + \
                          ID_split[1] + ".png"
        camera_file_name = ID_split[0] + "_camera_frontcenter_" + \
                           ID_split[1] + ".png"

        A_camera = cv.imread(os.path.join(ind_camera_dir, camera_file_name))
        pc = np.load(os.path.join(ind_pc_dir, pc_file_name), allow_pickle=True)

        # Give labels to points in PC equal to [row, index] these points map to in label image space
        keys = list(pc.keys())
        if 'row' in keys and 'col' in keys and 'points' in keys:
            rows, cols = pc['row'], pc['col']
            classes = []
            rgb = []
            N = len(rows)
            i = 0
            for row, col in zip(rows, cols):  # O(n)
                print(RGB[i, :])
                print(A_camera[int(row), int(col)])
                print("TEST: ", RGB[i, :] == A_camera[int(row), int(col)][::-1])
                i += 1

    # Get classes <--> labels dictionaries
    class_dict, rgb_dict = get_classes_to_ids()
    inverse_rgb_dict = {rgb_dict[key]: key for key in list(rgb_dict.keys())}

    test_id_consistency(A, 2)
