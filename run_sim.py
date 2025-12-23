from simulator import Simulator
import sys

if __name__ == "__main__":
    # Simple argument parsing to support --config if needed, 
    # but Simulator takes config_path in __init__.
    config_path = 'debug_config.yaml'
    dataset = 'MNIST'
    
    # Check for args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='debug_config.yaml')
    parser.add_argument('--dataset', type=str, default='MNIST')
    args = parser.parse_args()
    
    sim = Simulator(config_path=args.config, dataset_name=args.dataset)
    sim.run()
