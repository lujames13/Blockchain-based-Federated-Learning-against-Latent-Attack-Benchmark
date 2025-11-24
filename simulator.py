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
        
        # Initialize verifiers for BlockDFL
        self.verifiers_blockdfl = []
        for i in range(self.verifier_pool_size):
            is_attacker = i < num_attackers
            self.verifiers_blockdfl.append({
                'id': i,
                'is_attacker': is_attacker,
                'stack': 1.0
            })
            
        # Initialize verifiers for Ours (identical start)
        self.verifiers_ours = copy.deepcopy(self.verifiers_blockdfl)
        
        self.stack_reward = self.config.get('stack_reward', 0.1)

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

            # --- Selection & Stack Update Logic ---
            attack_active = round_num >= self.config['attack_start_round']
            
            # Helper to select committee based on stack
            def select_committee(pool, size):
                if len(pool) < size:
                    return list(range(len(pool)))
                total_stack = sum(v['stack'] for v in pool)
                if total_stack == 0: # Should not happen ideally
                    probs = [1.0/len(pool)] * len(pool)
                else:
                    probs = [v['stack'] / total_stack for v in pool]
                
                indices = np.random.choice(
                    len(pool), 
                    size=size, 
                    replace=False, 
                    p=probs
                )
                return indices

            # --- BlockDFL Logic ---
            committee_indices_blockdfl = select_committee(self.verifiers_blockdfl, self.committee_size)
            committee_blockdfl = [self.verifiers_blockdfl[i] for i in committee_indices_blockdfl]
            
            num_attackers_blockdfl = sum(1 for v in committee_blockdfl if v['is_attacker'])
            num_honest_blockdfl = len(committee_blockdfl) - num_attackers_blockdfl
            
            # Determine BlockDFL Outcome
            blockdfl_attack_successful = False
            if attack_active and num_attackers_blockdfl > num_honest_blockdfl:
                blockdfl_attack_successful = True
                
            # Calculate Total Pot
            total_pot = self.committee_size * self.stack_reward
            
            if blockdfl_attack_successful:
                # Attack Success: Select WORST
                blockdfl_idx = np.argmin(blockdfl_qualities)
                blockdfl_selection_type = "WORST (ATTACK SUCCESS)"
                
                # Stack Update: Attackers split the ENTIRE pot
                # Honest get nothing
                reward_per_attacker = total_pot / num_attackers_blockdfl
                
                for v in committee_blockdfl:
                    if v['is_attacker']:
                        v['stack'] += reward_per_attacker
            else:
                # Attack Failed (or not active): Select BEST
                # Even if attack is active, if attackers are minority, they cooperate to get reward
                blockdfl_idx = np.argmax(blockdfl_qualities)
                blockdfl_selection_type = "BEST"
                if attack_active:
                    blockdfl_selection_type += " (ATTACK FAILED)"
                
                # Stack Update: Everyone splits the pot (Normal reward)
                # reward_per_member = total_pot / len(committee_blockdfl) = stack_reward
                for v in committee_blockdfl:
                    v['stack'] += self.stack_reward

            blockdfl_selected_update = blockdfl_updates[blockdfl_idx]

            # --- Ours Logic ---
            committee_indices_ours = select_committee(self.verifiers_ours, self.committee_size)
            committee_ours = [self.verifiers_ours[i] for i in committee_indices_ours]
            
            # Ours always selects BEST (Audit)
            ours_idx = np.argmax(ours_qualities)
            ours_selected_update = ours_updates[ours_idx]
            ours_selection_type = "BEST"
            
            num_attackers_ours = sum(1 for v in committee_ours if v['is_attacker'])
            num_honest_ours = len(committee_ours) - num_attackers_ours
            
            # Stack Update for Ours
            # User requirement: Attackers are slashed ONLY if they successfully attack (majority).
            # If they don't have majority, they cooperate and get reward.
            
            ours_attack_successful = attack_active and (num_attackers_ours > num_honest_ours)
            
            if ours_attack_successful:
                # Attackers in committee are slashed to 0
                for v in committee_ours:
                    if v['is_attacker']:
                        v['stack'] = 0.0
                    else:
                        v['stack'] += self.stack_reward
            else:
                # Normal operation / Attack failed / Not active: Everyone gets reward
                for v in committee_ours:
                    v['stack'] += self.stack_reward

            
            # --- Apply Updates ---
            self.apply_update(self.model_blockdfl, blockdfl_selected_update)
            self.apply_update(self.model_ours, ours_selected_update)
            
            # --- Evaluate Global Models ---
            blockdfl_acc = self.evaluate_model(self.model_blockdfl)
            ours_acc = self.evaluate_model(self.model_ours)
            
            # --- Calculate Stack Stats ---
            def get_avg_stack(pool):
                honest_stacks = [v['stack'] for v in pool if not v['is_attacker']]
                attacker_stacks = [v['stack'] for v in pool if v['is_attacker']]
                avg_honest = sum(honest_stacks) / len(honest_stacks) if honest_stacks else 0
                avg_attacker = sum(attacker_stacks) / len(attacker_stacks) if attacker_stacks else 0
                return avg_honest, avg_attacker

            bdfl_avg_honest, bdfl_avg_attacker = get_avg_stack(self.verifiers_blockdfl)
            ours_avg_honest, ours_avg_attacker = get_avg_stack(self.verifiers_ours)

            # --- Log ---
            print(f"Round {round_num}/{self.config['total_rounds']}")
            print(f"  BlockDFL: {blockdfl_selection_type} (Attacker Stack: {bdfl_avg_attacker:.2f}, Honest Stack: {bdfl_avg_honest:.2f})")
            print(f"  Ours:     {ours_selection_type} (Attacker Stack: {ours_avg_attacker:.2f}, Honest Stack: {ours_avg_honest:.2f})")
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
                },
                "stack_stats": {
                    "blockdfl": {
                        "avg_honest": bdfl_avg_honest,
                        "avg_attacker": bdfl_avg_attacker
                    },
                    "ours": {
                        "avg_honest": ours_avg_honest,
                        "avg_attacker": ours_avg_attacker
                    }
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
