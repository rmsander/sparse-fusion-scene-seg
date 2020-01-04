import torch
import numpy as np

# FLAGS
NUM_CHANNELS = 6

def collate_fn(data, max_length_threshold=20000):
    """
       Function for batched training on point cloud inputs for PointNet
       frameworks.  Tested and validated on PointNet++ framework with
       PyTorch.  Used as a callable collation function when creating
       DataLoaders.

       Arguments:

            1. data (list): is a list of tuples with (example, label,
                length) where 'example' is a tensor of arbitrary shape
                and label/length are scalars.

            2. max_length_threshold (int): Maximum number of points allowed
                per point cloud.  Used to ensure the GPU does not become
                flooded from an exceptionally large point cloud input.

        Returns:

            Padded Point Clouds: A tuple of PyTorch tensors for features and
                labels that are processed through the PyTorch DataLoader
                framework when used in training/testing.
    """

    def fill_in_data(xyz_rgb_seq, labels_seq, max_length, debugging=False):
        """Helper function for the collation function above.  Fills in
        missing data through uniform resampling and concatenates original and
        resampled data together."""

        N = len(xyz_rgb_seq)
        points_needed = max(max_length - N, 0)  # Needed for this point cloud

        # Randomly sample 
        random_sample_indices = np.random.randint(0, high=N, size=points_needed)
        data_random_samples = xyz_rgb_seq[random_sample_indices, :]
        label_random_samples = labels_seq[random_sample_indices]

        # Concatenate original features/labels with resampled features/labels
        merged_features = np.concatenate((xyz_rgb_seq, data_random_samples),
                                          axis=0)
        merged_labels = np.concatenate((labels_seq, label_random_samples),
                                          axis=0)

        return merged_features, merged_labels

    # Original function
    _, labels, lengths = zip(*data)  # For use with A2D2 DataLoader class
    max_len = min(max(lengths), max_length_threshold)
    n_ftrs = data[0][0].size(1)  # Get dimensions

    # Create output arrays, populate, and convert to tensors
    features_padded = np.zeros((len(data), max_len, n_ftrs))  # Output DS
    labels_padded = np.zeros((len(data), max_len))  # Output DS

    for i in range(len(data)):  # Iterate through point clouds in batch
        # Call helper function and resample from point cloud i
        merged_features, merged_labels = fill_in_data(data[i][0].numpy(),
                                                data[i][1].numpy(), max_len)

        # Set padded features
        features_padded[i, :, :] = merged_features[:max_len]

        # Set padded labels
        labels_padded[i, :] = merged_labels[:max_len]

    # Convert to PyTorch tensors
    features_padded = torch.tensor(features_padded)
    labels_padded = torch.tensor(labels_padded)

    return features_padded.float(), labels_padded.long()
