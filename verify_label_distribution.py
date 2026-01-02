
import torch
import numpy as np
from data.loader import get_dataloaders
from collections import Counter
import sys

def verify_distribution():
    print("Verifying label distribution with alpha=0.5...")
    
    # Use parameters from config checks (or defaults close to it)
    # Simulator uses config['alpha'], which we saw is 0.5
    try:
        train_loaders, test_loader = get_dataloaders(
            dataset_name='MNIST',
            batch_size=32,
            num_clients=10, # Check first 10 to see sufficient heterogeneity, user asked for top 5
            alpha=0.5,
            seed=42
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"\nTotal Clients: {len(train_loaders)}")
    print("Checking label distribution for the first 5 clients:\n")

    for i in range(5):
        loader = train_loaders[i]
        all_labels = []
        for _, params in loader: # loader returns (data, target)
             # params is target
             all_labels.extend(params.tolist())
        
        counts = Counter(all_labels)
        total_samples = len(all_labels)
        
        print(f"Client {i} (Total Samples: {total_samples}):")
        # Sort by label for easier reading
        sorted_counts = dict(sorted(counts.items()))
        print(f"  Label Counts: {sorted_counts}")
        
        # Calculate distribution percentages
        dist = {k: f"{v/total_samples:.2f}" for k, v in sorted_counts.items()}
        print(f"  Distribution: {dist}")
        print("-" * 50)

if __name__ == "__main__":
    verify_distribution()
