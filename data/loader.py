import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_dataset(dataset_name, root='./data'):
    """
    Downloads and returns the training and test datasets.
    """
    if dataset_name.lower() in ['mnist', 'mnist_noniid']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform)
        
    elif dataset_name.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    return train_dataset, test_dataset

def dirichlet_split(dataset, num_clients=4, alpha=0.5, seed=42):
    """
    Splits the dataset into `num_clients` subsets using Dirichlet distribution.
    Returns a list of Subset objects.
    """
    np.random.seed(seed)
    
    # Get targets (labels)
    if hasattr(dataset, 'targets'):
        targets = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        targets = np.array(dataset.labels)
    else:
        # Fallback for datasets where targets might be a list
        targets = np.array([y for _, y in dataset])
        
    num_classes = len(np.unique(targets))
    min_size = 0
    N = len(targets)
    
    # Ensure each client gets at least min_size samples
    while min_size < 10:
        idx_batch = [[] for _ in range(num_clients)]
        
        for k in range(num_classes):
            idx_k = np.where(targets == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            
            # Balance proportions to avoid empty clients
            proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    subsets = [Subset(dataset, idxs) for idxs in idx_batch]
    return subsets

def get_dataloaders(dataset_name, batch_size=32, num_clients=4, alpha=0.5, seed=42):
    """
    Returns:
    - train_loaders: List of DataLoaders for each client (aggregator)
    - test_loader: DataLoader for the global test set
    """
    train_dataset, test_dataset = get_dataset(dataset_name)
    
    # Split training data
    subsets = dirichlet_split(train_dataset, num_clients, alpha, seed)
    
    train_loaders = [
        DataLoader(subset, batch_size=batch_size, shuffle=True)
        for subset in subsets
    ]
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loaders, test_loader
