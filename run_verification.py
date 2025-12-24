from simulator import Simulator
import sys

# Force output to stdout unbuffered
sys.stdout.reconfigure(line_buffering=True)

print("Initializing Simulator with verification config...")
sim = Simulator(config_path='config_verify.yaml', dataset_name='MNIST')
sim.run()
