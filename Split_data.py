import random
import torch
from torchvision import transforms

random.seed(42)

def random_split(data):
    random.seed(42)

    # Define the mean and standard deviation for normalization
    stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    # Create the transformation for normalization
    transform = transforms.Compose([
        transforms.Normalize(*stats)
    ])

    # Apply normalization to the data
    normalized_data = [(transform(tensor), label) for tensor, label in data]

    random.shuffle(normalized_data)

    # Calculate the lengths of train, validation, and test sets based on the ratios
    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2

    total_samples = len(normalized_data)
    train_samples = int(train_ratio * total_samples)
    val_samples = int(val_ratio * total_samples)
    test_samples = total_samples - train_samples - val_samples

    # Split the normalized data into train, validation, and test sets
    train_data = normalized_data[:train_samples]
    val_data = normalized_data[train_samples : train_samples + val_samples]
    test_data = normalized_data[train_samples + val_samples:]
    
    return train_data, val_data, test_data