import numpy as np
import math
import torch

def create_helmert_matrix(num_classes=10):
    helmert_matrix = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        helmert_matrix[0][i] = 1.0 / math.sqrt(num_classes)
    for i in range(1, num_classes):
        sqri = 1.0 / math.sqrt(i * (i + 1))
        for j in range(i):
            helmert_matrix[i][j] = sqri
        helmert_matrix[i][i] -= i * sqri
    helmert_matrix = helmert_matrix[1:, :]
    return helmert_matrix

def one_hot_encode(labels, num_classes):
    one_hot = torch.zeros(labels.size(0), num_classes, device=labels.device)
    one_hot[torch.arange(labels.size(0)), labels] = 1.0
    return one_hot

def batch_label_smoothing(labels, num_classes, gamma=0.01):
    batch_size = labels.size(0)
    uniform = torch.full((batch_size, num_classes), 1.0 / num_classes, device=labels.device)
    one_hot = one_hot_encode(labels, num_classes)
    smoothed_labels = (1 - gamma) * one_hot + gamma * uniform
    return smoothed_labels

def ilr_transform_batch(compositions, helmert_matrix):
    geo_mean = compositions.prod(dim=1, keepdim=True) ** (1.0 / compositions.size(1))
    clr = torch.log(compositions) - torch.log(geo_mean)
    helmert_matrix = torch.from_numpy(helmert_matrix).float().to(compositions.device)
    ilr_data = torch.matmul(helmert_matrix, clr.unsqueeze(2)).squeeze(2)
    return ilr_data

def inverse_ilr_transform_batch(ilr_data, helmert_matrix):
    helmert_matrix = torch.from_numpy(helmert_matrix).float().to(ilr_data.device)
    clr = torch.matmul(helmert_matrix.T, ilr_data.T).T
    exp_clr = torch.exp(clr)
    composition = exp_clr / exp_clr.sum(dim=1, keepdim=True)
    return composition

def add_noise_to_dataset(dataset, noise_rate=0.4, noise_type='symmetric'):
    if not 0 <= noise_rate <= 1:
        raise ValueError("Noise rate must be between 0 and 1")
    
    if noise_type not in ['symmetric', 'asymmetric']:
        raise ValueError("Noise type must be either 'symmetric' or 'asymmetric'")
    
    if isinstance(dataset, torch.utils.data.Subset):
        original_dataset = dataset.dataset
        indices = dataset.indices
        noisy_labels = np.array(original_dataset.targets)[indices]
    else:
        noisy_labels = np.array(dataset.targets)
    num_classes = len(np.unique(noisy_labels))

    if noise_type == 'symmetric':
        for i in range(len(noisy_labels)):
            if np.random.rand() < noise_rate:
                noisy_labels[i] = np.random.choice(np.delete(np.arange(num_classes), noisy_labels[i]))
                
    elif noise_type == 'asymmetric':
        # Asymmetric noise: flip labels to similar classes based on CIFAR-10 class relationships
        class_map = {
            0: 2,    # airplane -> bird
            1: 9,    # automobile -> truck
            2: 0,    # bird -> airplane
            3: 5,    # cat -> dog
            4: 7,    # deer -> horse
            5: 3,    # dog -> cat
            6: 4,    # frog -> deer
            7: 4,    # horse -> deer
            8: 1,    # ship -> automobile
            9: 1     # truck -> automobile
        }
        
        for i in range(len(noisy_labels)):
            if np.random.rand() < noise_rate:
                original_label = noisy_labels[i]
                noisy_labels[i] = class_map[original_label]

    if isinstance(dataset, torch.utils.data.Subset):
        for idx, original_idx in enumerate(indices):
            original_dataset.targets[original_idx] = noisy_labels[idx]
    else:
        dataset.targets = noisy_labels.tolist()

    return dataset