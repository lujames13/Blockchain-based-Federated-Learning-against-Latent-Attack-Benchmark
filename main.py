import argparse
import os
from simulator import Simulator

def main():
    parser = argparse.ArgumentParser(description='Latent Attack Benchmark')
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'CIFAR10', 'mnist', 'cifar10'],
                        help='Dataset to use (MNIST or CIFAR10)')
    args = parser.parse_args()
    
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    
    dataset_name = args.dataset.upper()
    if dataset_name == 'CIFAR10':
        dataset_name = 'CIFAR10' # Ensure consistent casing for config lookup
    
    sim = Simulator(dataset_name=dataset_name)
    sim.run()

if __name__ == "__main__":
    main()
