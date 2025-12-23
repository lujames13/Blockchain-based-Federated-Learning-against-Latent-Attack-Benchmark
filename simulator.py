import sys
print("DEBUG: Simulator starting...", file=sys.stderr)
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
        elif dataset_name.lower() == 'mnist_noniid':
            self.config = full_config['mnist_noniid']
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
        self.num_candidates = self.config.get('num_candidates', 4)
        self.providers_per_aggregator = 5  # Fixed number of providers per aggregator
        
        # Check if pool is large enough
        required_nodes = self.committee_size + self.num_candidates * (1 + self.providers_per_aggregator)
        if self.verifier_pool_size < required_nodes:
            print(f"WARNING: Verifier pool size ({self.verifier_pool_size}) is smaller than required ({required_nodes}). Resizing pool to {required_nodes}.")
            self.verifier_pool_size = required_nodes
            
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

    def train_malicious_candidate(self, base_model, train_loader):
        """
        Trains a malicious candidate update with label flipping.
        Target = 9 - Target (assuming 10 classes).
        """
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
                
                # Label Flipping Attack
                target = 9 - target
                
                optimizer.zero_grad()
                output = candidate_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        # Calculate update
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
            
            # Helper to assign roles
            def assign_roles(pool):
                committee_size = self.committee_size
                num_aggregators = self.num_candidates
                providers_per_agg = self.providers_per_aggregator
                
                # Weight by stack
                total_stack = sum(v['stack'] for v in pool)
                if total_stack == 0:
                    probs = [1.0/len(pool)] * len(pool)
                else:
                    probs = [v['stack'] / total_stack for v in pool]
                
                # Sample all needed roles at once to ensure disjoint sets
                total_needed = committee_size + num_aggregators * (1 + providers_per_agg)
                if len(pool) < total_needed:
                     # Fallback if pool shrunk or not enough
                     # This should not happen given __init__ check, but for safety:
                     indices = np.random.choice(len(pool), size=len(pool), replace=False, p=probs)
                else:
                    indices = np.random.choice(len(pool), size=total_needed, replace=False, p=probs)
                
                # Slice indices
                committee_idxs = indices[:committee_size]
                remaining = indices[committee_size:]
                
                aggregators_with_providers = []
                cursor = 0
                for _ in range(num_aggregators):
                    agg_idx = remaining[cursor]
                    cursor += 1
                    prov_idxs = remaining[cursor : cursor + providers_per_agg]
                    cursor += providers_per_agg
                    
                    aggregators_with_providers.append({
                        'aggregator': pool[agg_idx],
                        'providers': [pool[i] for i in prov_idxs]
                    })
                
                committee = [pool[i] for i in committee_idxs]
                
                return committee, aggregators_with_providers

            # Generate Updates and Roles for BlockDFL
            committee_bdfl, groups_bdfl = assign_roles(self.verifiers_blockdfl)
            
            # Generate Updates and Roles for Ours
            committee_ours, groups_ours = assign_roles(self.verifiers_ours)
            
            # --- Logic to Generate Updates ---
            # We generate updates based on the ASSIGNED roles. 
            # Note: In a real simulation, providers send updates to aggregator, who aggregates.
            # Here we simulate the *result* of that process. 
            # If Aggregator is Malicious, he creates a Malicious Update (Poisoned).
            # If Aggregator is Honest, he creates a Valid Update (from Training).
            # For simplicity, we use the `train_candidate` or `train_malicious_candidate` directly for the Aggregator's "Output".
            
            def process_candidates(model, groups, is_attack_active):
                updates = []
                qualities = []
                corruption_scores = [] # count of malicious nodes in the group (Agg + Providers)
                is_malicious_update = []
                
                for group in groups:
                    agg = group['aggregator']
                    provs = group['providers']
                    
                    # Compute Corruption Score
                    num_mal_prov = sum(1 for p in provs if p['is_attacker'])
                    score = (1 if agg['is_attacker'] else 0) + num_mal_prov
                    corruption_scores.append(score)
                    
                    # Generate Update
                    # If Aggregator is Malicious AND Attack Active -> Generate Malicious Update
                    # Warning: Aggregator might be honest but have malicious providers.
                    # Standard assumption: Malicious Aggregator = Malicious Update.
                    # Honest Aggregator with Malicious Providers = Slightly degraded but mostly Honest update (robust aggregation)?
                    # For this high-level sim, we assume:
                    # Malicious Aggregator -> Poisoned Update (Label Flip)
                    # Honest Aggregator -> Legitimate Update (Normal Training)
                    
                    if is_attack_active and agg['is_attacker']:
                        # Malicious Aggregator injects attack
                        upd = self.train_malicious_candidate(model, self.train_loaders[0]) # Use loader 0 as dummy source
                        is_malicious_update.append(True)
                    else:
                        # Honest Aggregator (even if some providers are malicious, we assume he filters/aggregates correctly for now, or just simply honest training)
                        # To vary the updates, utilize different loaders if possible, or just one.
                        # We have 4 loaders. Assign randomly.
                        upd = self.train_candidate(model, self.train_loaders[np.random.randint(len(self.train_loaders))])
                        is_malicious_update.append(False)
                    
                    # Evaluate
                    qual = self.evaluate_update(model, upd)
                    updates.append(upd)
                    qualities.append(qual)
                    
                return updates, qualities, corruption_scores, is_malicious_update

            attack_active = round_num >= self.config['attack_start_round']
            
            # --- BlockDFL Process ---
            bdfl_updates, bdfl_qualities, bdfl_scores, bdfl_is_mal = process_candidates(
                self.model_blockdfl, groups_bdfl, attack_active
            )
            
            num_attackers_bdfl = sum(1 for v in committee_bdfl if v['is_attacker'])
            committee_captured_bdfl = attack_active and (num_attackers_bdfl > (2 / 3) * self.committee_size)
            
            if committee_captured_bdfl:
                # Malicious Committee selects the update with MAX Corruption Score (Nepotism)
                # To maximize stake accumulation for the malicious clan
                bdfl_idx = np.argmax(bdfl_scores)
                bdfl_selection_type = "NEPOTISM (CAPTURED)"
            else:
                # Honest Committee selects BEST Quality
                bdfl_idx = np.argmax(bdfl_qualities)
                bdfl_selection_type = "BEST"
                
            # BlockDFL Rewards: Everyone in Committee + Winning Group gets Reward
            # No Slashing in BlockDFL
            winning_group_bdfl = groups_bdfl[bdfl_idx]
            
            # Reward Committee
            for v in committee_bdfl:
                v['stack'] += self.stack_reward
            
            # Reward Winning Aggregator & Providers
            winning_group_bdfl['aggregator']['stack'] += self.stack_reward
            for p in winning_group_bdfl['providers']:
                p['stack'] += self.stack_reward
                
            blockdfl_selected_update = bdfl_updates[bdfl_idx]
            
            # --- Ours Process ---
            ours_updates, ours_qualities, ours_scores, ours_is_mal = process_candidates(
                self.model_ours, groups_ours, attack_active
            )
            
            num_attackers_ours = sum(1 for v in committee_ours if v['is_attacker'])
            committee_captured_ours = attack_active and (num_attackers_ours > (2 / 3) * self.committee_size)
            
            if committee_captured_ours:
                # Malicious Committee selects MAX Corruption Score (Nepotism)
                ours_idx = np.argmax(ours_scores)
                ours_selection_type = "NEPOTISM (CAPTURED)"
                
                # Reward Loop: Committee + Winning Group (bypass Slash)
                # "Accumulate Stake"
                winning_group_ours = groups_ours[ours_idx]
                
                for v in committee_ours:
                    v['stack'] += self.stack_reward
                
                winning_group_ours['aggregator']['stack'] += self.stack_reward
                for p in winning_group_ours['providers']:
                    p['stack'] += self.stack_reward
                    
            else:
                # Honest Committee selects BEST Quality
                ours_idx = np.argmax(ours_qualities)
                ours_selection_type = "BEST"
                
                winning_group_ours = groups_ours[ours_idx]
                
                # Check for Slashing Opportunity (Honest Defense)
                # If the *Selected* update was Malicious (unlikely if Honest selects Best, but possible if Malicious was good quality or luck)
                # OR, checking if any CANDIDATE was malicious? 
                # BlockDFL doesn't slash. Ours does.
                # Standard Logic: If an honest committee selects a Valid update, they reward it.
                # If they detect a Malicious update during verification, they slash it.
                # Here, we assume they selected the Best. If it's Honest, Reward.
                # If it's Malicious (e.g. good Attack), they might be fooled -> Reward.
                # BUT, if they *rejected* Malicious updates, do they Slash the losers?
                # User says "Ours has slash mechanism to handle malicious attacks".
                # To simulate "handling", we should slash the Malicious Aggregators who *tried* to attack but failed.
                
                # Apply Reward to Winner
                for v in committee_ours:
                    v['stack'] += self.stack_reward
                winning_group_ours['aggregator']['stack'] += self.stack_reward
                for p in winning_group_ours['providers']:
                    p['stack'] += self.stack_reward
                
                # Apply Slashing to Malicious Candidates (Defense)
                # Iterate all candidates: if they were Malicious (and Attack Active, implying they tried to poison), Slash them.
                num_slashed = 0
                if attack_active:
                     for i, grp in enumerate(groups_ours):
                         agg = grp['aggregator']
                         if agg['is_attacker']:
                             # Detected Malicious Aggregator -> Slash 
                             agg['stack'] = 0.0
                             num_slashed += 1
                
                if num_slashed > 0:
                    ours_selection_type += f" (SLASHED {num_slashed})"
                

            ours_selected_update = ours_updates[ours_idx]
            
            # --- Apply Updates ---
            self.apply_update(self.model_blockdfl, blockdfl_selected_update)
            self.apply_update(self.model_ours, ours_selected_update)
            
            # --- Evaluate Global Models ---
            blockdfl_acc = self.evaluate_model(self.model_blockdfl)
            ours_acc = self.evaluate_model(self.model_ours)
            
            # --- Log ---
            def get_avg_stack(pool):
                honest_stacks = [v['stack'] for v in pool if not v['is_attacker']]
                attacker_stacks = [v['stack'] for v in pool if v['is_attacker']]
                avg_honest = sum(honest_stacks) / len(honest_stacks) if honest_stacks else 0
                avg_attacker = sum(attacker_stacks) / len(attacker_stacks) if attacker_stacks else 0
                return avg_honest, avg_attacker

            bdfl_avg_honest, bdfl_avg_attacker = get_avg_stack(self.verifiers_blockdfl)
            ours_avg_honest, ours_avg_attacker = get_avg_stack(self.verifiers_ours)

            print(f"Round {round_num}/{self.config['total_rounds']}")
            print(f"  BlockDFL: {bdfl_selection_type} | Attacker Stack: {bdfl_avg_attacker:.2f} | Honest Stack: {bdfl_avg_honest:.2f}")
            print(f"  Ours:     {ours_selection_type} | Attacker Stack: {ours_avg_attacker:.2f} | Honest Stack: {ours_avg_honest:.2f}")
            print(f"  -> BlockDFL Acc: {blockdfl_acc:.4f}")
            print(f"  -> Ours Acc:     {ours_acc:.4f}")
            
            # Save results
            round_data = {
                "round": round_num,
                "blockdfl_accuracy": blockdfl_acc,
                "ours_accuracy": ours_acc,
                "blockdfl_selection": bdfl_selection_type,
                "ours_selection": ours_selection_type,
                "stack_stats": {
                    "blockdfl": {"avg_honest": bdfl_avg_honest, "avg_attacker": bdfl_avg_attacker},
                    "ours": {"avg_honest": ours_avg_honest, "avg_attacker": ours_avg_attacker}
                }
            }
            self.results["results"].append(round_data)
            
            if round_num % 10 == 0:
                self.save_results()
                
        self.save_results()
        print("Simulation Complete.")

    def save_results(self):
        filename = f"{self.dataset_name.lower()}_results.json"
        filepath = os.path.join('results', filename)
        os.makedirs('results', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filepath}")

