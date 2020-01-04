import torch
import numpy as np

#FLAGS
NUM_CHANNELS = 6
    
def collate_fn(data, max_length_threshold=20000):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    
    def fill_in_data(xyz_rgb_seq, labels_seq, max_length, debugging=False):
        N = len(xyz_rgb_seq)
        points_needed = max(max_length-N, 0)  # Compare in case we run into issues with thresholding max size
        
        # Randomly sample 
        random_sample_indices = np.random.randint(0, high=N, size=points_needed)
        data_random_samples = xyz_rgb_seq[random_sample_indices,:]
        label_random_samples = labels_seq[random_sample_indices]
        
        cat_features = np.concatenate((xyz_rgb_seq, data_random_samples), axis=0)
        cat_labels = np.concatenate((labels_seq, label_random_samples), axis=0)
        
        if debugging:
            print("GEN_DATA: {}, GEN_LABELS: {}".format(data_random_samples.shape, \
                                                label_random_samples.shape))
            print("IN_DATA: {}, IN_LABELS: {}".format(xyz_rgb_seq.shape, \
                                                labels_seq.shape))
            print("DATA: {}, LABELS: {}".format(cat_features.shape, cat_labels.shape))

        return cat_features, cat_labels
    
    _, labels, lengths = zip(*data)
    max_len = min(max(lengths), max_length_threshold)
    n_ftrs = data[0][0].size(1)
    lengths = torch.tensor(lengths)
    
    # Create output arrays, populate, and convert to tensors
    features_padded = np.zeros((len(data), max_len, n_ftrs)) 
    labels_padded = np.zeros((len(data), max_len))
    
    for i in range(len(data)):
        # Pad with existing points
        j, k = data[i][0].size(0), data[i][0].size(1) # N, D
        cat_features, cat_labels = fill_in_data(data[i][0].numpy(), data[i][1].numpy(), max_len)
        features_padded[i,:,:] = cat_features[:max_len]  # Slice to here in case we have an observation that is longer
        labels_padded[i,:] = cat_labels[:max_len]        # Slice to here in case we have an observation that is longer
    
    # Convert to tensors
    features_padded = torch.tensor(features_padded)
    labels_padded = torch.tensor(labels_padded)
    
    return features_padded.float(), labels_padded.long()