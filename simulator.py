import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import json
import os
from data.loader import get_dataloaders
from models.mnist_net import MNISTNet
from models.cifar10_net import CIFAR10Net

class Simulator:
    def __init__(self, config_path='config.yaml', dataset_name='MNIST'):
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
        
        # Select config based on dataset
        if dataset_name.lower() == 'mnist':
            self.config = full_config['mnist']
            self.ModelClass = MNISTNet
        elif dataset_name.lower() == 'cifar10':
            self.config = full_config['cifar10']
            self.ModelClass = CIFAR10Net
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
        self.dataset_name = dataset_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize data
        print("Initializing data...")
        self.train_loaders, self.test_loader = get_dataloaders(
            dataset_name, 
            batch_size=self.config['batch_size'],
            num_clients=self.config['num_candidates'],
            alpha=self.config['alpha']
        )
        
        # Initialize models (BlockDFL and Ours start with same weights)
        print("Initializing models...")
        self.model_blockdfl = self.ModelClass().to(self.device)
        self.model_ours = self.ModelClass().to(self.device)
        
        # Ensure identical initialization
        self.model_ours.load_state_dict(self.model_blockdfl.state_dict())
        
        # Results storage
        self.results = {
            "dataset": dataset_name,
            "total_rounds": self.config['total_rounds'],
            "attack_start_round": self.config['attack_start_round'],
            "results": []
        }
        
        # Initialize verifier pool
        self.verifier_pool_size = self.config.get('verifier_pool_size', 100)
        self.initial_attacker_ratio = self.config.get('initial_attacker_ratio', 0.1)
        self.committee_size = self.config.get('committee_size', 7)
        
        num_attackers = int(self.verifier_pool_size * self.initial_attacker_ratio)
        self.verifiers = []
        for i in range(self.verifier_pool_size):
            is_attacker = i < num_attackers
            self.verifiers.append({
                'id': i,
                'is_attacker': is_attacker,
                'stack': 1.0
            })

    def train_candidate(self, base_model, train_loader):
        """
        Trains a candidate update starting from base_model on train_loader.
        Returns the update (difference in weights).
        """
        # Create a copy of the model for training
        candidate_model = copy.deepcopy(base_model)
        candidate_model.train()
        
        optimizer = optim.SGD(
            candidate_model.parameters(), 
            lr=self.config['learning_rate'],
            momentum=self.config.get('momentum', 0.9)
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # Train for local_training_epochs
        for _ in range(self.config['local_training_epochs']):
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = candidate_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        # Calculate update: new_weights - old_weights
        update = {}
        for name, param in candidate_model.named_parameters():
            base_param = base_model.state_dict()[name]
            update[name] = param.data - base_param
            
        return update

    def evaluate_update(self, base_model, update):
        """
        Evaluates the quality of an update by applying it to the base_model
        and testing on the test set.
        Returns accuracy.
        """
        # Create a temporary model to evaluate the update
        eval_model = copy.deepcopy(base_model)
        
        # Apply update
        with torch.no_grad():
            for name, param in eval_model.named_parameters():
                param.data += update[name]
        
        # Evaluate
        eval_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = eval_model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
        return correct / total

    def apply_update(self, model, update):
        """
        Applies the selected update to the model.
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.data += update[name]

    def evaluate_model(self, model):
        """
        Evaluates the current model on the test set.
        """
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        return correct / total

    def run(self):
        print(f"Starting simulation for {self.dataset_name}...")
        print(f"Total Rounds: {self.config['total_rounds']}")
        print(f"Attack Starts at Round: {self.config['attack_start_round']}")
        
        for round_num in range(1, self.config['total_rounds'] + 1):
            # Decay learning rate
            current_lr = self.config['learning_rate'] * (self.config['lr_decay'] ** (round_num - 1))
            
            # Since both models start identical and we want to simulate the divergence point,
            # we can optimize by generating updates from ONE model (e.g., BlockDFL) 
            # as long as they are identical. Once they diverge, we strictly need to generate 
            # updates for each model separately if we were doing a full FL simulation.
            # However, the PRD simplifies this: "Maintain two independent models".
            # So we should generate updates for EACH model separately?
            # PRD Section 2.2 says: "Generate 4 candidate updates... based on different data subsets".
            # And "Evaluate each update quality".
            # If the models are different, the updates generated from them will be different.
            # So we must generate updates for BlockDFL model AND for Ours model separately.
            
            # --- Generate updates for BlockDFL Model ---
            blockdfl_updates = []
            blockdfl_qualities = []
            for i in range(self.config['num_candidates']):
                update = self.train_candidate(self.model_blockdfl, self.train_loaders[i])
                quality = self.evaluate_update(self.model_blockdfl, update)
                blockdfl_updates.append(update)
                blockdfl_qualities.append(quality)
            
            # --- Generate updates for Ours Model ---
            # Note: In the early epochs, these will be identical to BlockDFL updates
            # if we use the same seed/order, but let's compute them explicitly to be safe and correct.
            ours_updates = []
            ours_qualities = []
            for i in range(self.config['num_candidates']):
                update = self.train_candidate(self.model_ours, self.train_loaders[i])
                quality = self.evaluate_update(self.model_ours, update)
                ours_updates.append(update)
                ours_qualities.append(quality)

            # --- Selection Logic ---
            attack_active = round_num >= self.config['attack_start_round']
            
            # --- Committee Selection ---
            # Calculate selection probabilities based on stack
            if len(self.verifiers) < self.committee_size:
                print(f"  [WARNING] Pool size ({len(self.verifiers)}) < Committee size ({self.committee_size}). Selecting all.")
                committee_indices = list(range(len(self.verifiers)))
            else:
                total_stack = sum(v['stack'] for v in self.verifiers)
                probs = [v['stack'] / total_stack for v in self.verifiers]
                
                # Select committee indices
                committee_indices = np.random.choice(
                    len(self.verifiers), 
                    size=self.committee_size, 
                    replace=False, 
                    p=probs
                )
            
            committee = [self.verifiers[i] for i in committee_indices]
            
            num_attackers_in_committee = sum(1 for v in committee if v['is_attacker'])
            num_honest_in_committee = len(committee) - num_attackers_in_committee
            
            # --- Decision Logic ---
            attack_successful = False
            if attack_active and num_attackers_in_committee > num_honest_in_committee:
                attack_successful = True
                
            # BlockDFL Selection
            if attack_successful:
                # Attack: Select WORST update
                blockdfl_idx = np.argmin(blockdfl_qualities)
                blockdfl_selected_update = blockdfl_updates[blockdfl_idx]
                blockdfl_selection_type = "WORST (ATTACK SUCCESS)"
                
                # Penalty: Remove honest committee members from pool
                honest_ids_to_remove = [v['id'] for v in committee if not v['is_attacker']]
                self.verifiers = [v for v in self.verifiers if v['id'] not in honest_ids_to_remove]
                print(f"  [PENALTY] Removed {len(honest_ids_to_remove)} honest verifiers from pool.")
            else:
                # Normal: Select BEST update
                blockdfl_idx = np.argmax(blockdfl_qualities)
                blockdfl_selected_update = blockdfl_updates[blockdfl_idx]
                if attack_active:
                    blockdfl_selection_type = "BEST (ATTACK FAILED)"
                else:
                    blockdfl_selection_type = "BEST"
                
            # Ours Selection
            # Always select BEST update (Audit mechanism)
            ours_idx = np.argmax(ours_qualities)
            ours_selected_update = ours_updates[ours_idx]
            ours_selection_type = "BEST"
            
            # --- Apply Updates ---
            self.apply_update(self.model_blockdfl, blockdfl_selected_update)
            self.apply_update(self.model_ours, ours_selected_update)
            
            # --- Evaluate Global Models ---
            blockdfl_acc = self.evaluate_model(self.model_blockdfl)
            ours_acc = self.evaluate_model(self.model_ours)
            
            # --- Log ---
            print(f"Round {round_num}/{self.config['total_rounds']}")
            if attack_active:
                print(f"  [ATTACK ACTIVE] BlockDFL selected {blockdfl_selection_type} (score={blockdfl_qualities[blockdfl_idx]:.4f})")
                print(f"  Committee: {num_attackers_in_committee} Attackers, {num_honest_in_committee} Honest. Pool Size: {len(self.verifiers)}")
                
                # Calculate current attacker stack share
                current_total_stack = sum(v['stack'] for v in self.verifiers)
                attacker_stack = sum(v['stack'] for v in self.verifiers if v['is_attacker'])
                attacker_share = attacker_stack / current_total_stack if current_total_stack > 0 else 0
                print(f"  Attacker Stack Share: {attacker_share:.4f}")
            else:
                print(f"  [Normal] BlockDFL selected {blockdfl_selection_type} (score={blockdfl_qualities[blockdfl_idx]:.4f})")
            print(f"  Ours selected {ours_selection_type} (score={ours_qualities[ours_idx]:.4f})")
            print(f"  -> BlockDFL Acc: {blockdfl_acc:.4f}")
            print(f"  -> Ours Acc:     {ours_acc:.4f}")
            
            # Save results
            round_data = {
                "round": round_num,
                "blockdfl_accuracy": blockdfl_acc,
                "ours_accuracy": ours_acc,
                "blockdfl_selection": {
                    "type": blockdfl_selection_type,
                    "quality": blockdfl_qualities[blockdfl_idx],
                    "all_qualities": blockdfl_qualities
                },
                "ours_selection": {
                    "type": ours_selection_type,
                    "quality": ours_qualities[ours_idx],
                    "all_qualities": ours_qualities
                }
            }
            self.results["results"].append(round_data)
            
            # Save intermediate results every 10 rounds
            if round_num % 10 == 0:
                self.save_results()

        self.save_results()
        print("Simulation Complete.")

    def save_results(self):
        filename = f"{self.dataset_name.lower()}_results.json"
        filepath = os.path.join('results', filename)
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filepath}")
