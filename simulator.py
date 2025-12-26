import sys
print("DEBUG: Simulator starting...", file=sys.stderr)
import copy
import random
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
        # Set fixed random seed for reproducibility
        seed = 1042
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
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
        # Strict config reading - no default value
        if 'initial_attacker_ratio' not in self.config:
            raise ValueError(f"initial_attacker_ratio must be defined in config for {dataset_name}")
        self.initial_attacker_ratio = self.config['initial_attacker_ratio']
        self.committee_size = self.config.get('committee_size', 7)
        self.num_candidates = self.config.get('num_candidates', 4)
        self.providers_per_aggregator = 5  # Fixed number of providers per aggregator
        
        # Check if pool is large enough
        required_nodes = self.committee_size + self.num_candidates * (1 + self.providers_per_aggregator)
        if self.verifier_pool_size < required_nodes:
            print(f"WARNING: Verifier pool size ({self.verifier_pool_size}) is smaller than required ({required_nodes}). Resizing pool to {required_nodes}.")
            self.verifier_pool_size = required_nodes
            
        num_attackers = int(self.verifier_pool_size * self.initial_attacker_ratio)
        
        
        # Initialize participants for BlockDFL (formerly verifiers_blockdfl)
        self.participants_blockdfl = []
        for i in range(self.verifier_pool_size): # kept variable name for config compatibility
            is_attacker = i < num_attackers
            self.participants_blockdfl.append({
                'id': i,
                'is_attacker': is_attacker,
                'stack': 1.0
            })
            
        # Initialize participants for Ours (identical start)
        self.participants_ours = copy.deepcopy(self.participants_blockdfl)
        
        # Load detailed rewards
        self.rewards = self.config.get('rewards', {
            'verifier': 0.1,
            'aggregator': 0.2,
            'provider': 0.05
        })
        self.slash_penalty = self.config.get('slash_penalty', 'full')

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
                providers_per_agg = self.providers_per_aggregator # Should be (len(pool) - comm - agg) / agg
                
                # 1. Select Verifiers (Committee)
                # Weight by stack
                total_stack = sum(v['stack'] for v in pool)
                if total_stack == 0:
                    probs_v = [1.0/len(pool)] * len(pool)
                else:
                    probs_v = [v['stack'] / total_stack for v in pool]
                
                # Sample Committee indices without replacement
                committee_idxs = np.random.choice(len(pool), size=committee_size, replace=False, p=probs_v)
                committee = [pool[i] for i in committee_idxs]
                
                # Remaining pool for Aggregators
                remaining_idxs = [i for i in range(len(pool)) if i not in committee_idxs]
                remaining_pool_obj = [pool[i] for i in remaining_idxs]
                
                # 2. Select Aggregators from Remaining
                total_stack_rem = sum(p['stack'] for p in remaining_pool_obj)
                if total_stack_rem == 0:
                    probs_a = [1.0/len(remaining_pool_obj)] * len(remaining_pool_obj)
                else:
                    probs_a = [p['stack'] / total_stack_rem for p in remaining_pool_obj]
                
                # Sample Aggregator indices (relative to remaining_pool_obj)
                # We need M aggregators
                agg_rel_idxs = np.random.choice(len(remaining_pool_obj), size=num_aggregators, replace=False, p=probs_a)
                
                aggregators = [remaining_pool_obj[i] for i in agg_rel_idxs]
                
                # Remaining for Providers
                # Remove aggregators from remaining_pool_obj
                final_remaining_obj = [p for i, p in enumerate(remaining_pool_obj) if i not in agg_rel_idxs]
                
                # 3. Assign Providers to Aggregators
                # Shuffle final_remaining to distribute randomly
                np.random.shuffle(final_remaining_obj)
                
                # Split evenly
                # Note: We might have leftovers if not perfectly divisible, but we'll handle by splitting current chunks
                # The user said: num=participents/aggregators. We assume it's roughly equal.
                chunk_size = len(final_remaining_obj) // num_aggregators
                
                aggregators_with_providers = []
                for i in range(num_aggregators):
                    start = i * chunk_size
                    # For the last one, take all remaining to avoid orphans
                    if i == num_aggregators - 1:
                        end = len(final_remaining_obj)
                    else:
                        end = (i + 1) * chunk_size
                        
                    my_providers = final_remaining_obj[start:end]
                    
                    aggregators_with_providers.append({
                        'aggregator': aggregators[i],
                        'providers': my_providers
                    })
                
                return committee, aggregators_with_providers

            # Generate Updates and Roles for BlockDFL
            committee_bdfl, groups_bdfl = assign_roles(self.participants_blockdfl)
            
            # Generate Updates and Roles for Ours
            committee_ours, groups_ours = assign_roles(self.participants_ours)
            
            # --- Logic to Generate Updates ---
            # We generate updates based on the ASSIGNED roles. 
            # Note: In a real simulation, providers send updates to aggregator, who aggregates.
            # Here we simulate the *result* of that process. 
            # If Aggregator is Malicious, he creates a Malicious Update (Poisoned).
            # If Aggregator is Honest, he creates a Valid Update (from Training).
            # For simplicity, we use the `train_candidate` or `train_malicious_candidate` directly for the Aggregator's "Output".
            
            def process_candidates(model, groups, is_attack_active, is_committee_captured):
                """
                Generate candidate updates based on attack state.
                
                is_attack_active: True if round >= attack_start_round
                is_committee_captured: True if malicious nodes > 2/3 of committee
                
                Latent Phase (not captured): All aggregators use honest training
                Attack Phase (captured): Malicious aggregators use label flipping
                """
                updates = []
                qualities = []
                corruption_scores = [] # count of malicious nodes in the group (Agg + Providers)
                is_malicious_update = []
                
                for group in groups:
                    agg = group['aggregator']
                    provs = group['providers']
                    
                    # Compute Corruption Score
                    num_mal_prov = sum(1 for p in provs if p['is_attacker'])
                    score = (100 if agg['is_attacker'] else 0) + num_mal_prov
                    corruption_scores.append(score)
                    
                    # Generate Update
                    # LATENT PHASE: Even if attack is "active", malicious nodes act honestly until committee captured
                    # ATTACK PHASE: Once committee captured, malicious aggregators launch label flipping attack
                    
                    if is_attack_active and is_committee_captured and agg['is_attacker']:
                        # ATTACK PHASE: Malicious Aggregator launches Label Flipping attack
                        upd = self.train_malicious_candidate(model, self.train_loaders[0])
                        is_malicious_update.append(True)
                    else:
                        # LATENT PHASE or Honest Aggregator: Normal training
                        # Malicious nodes remain stealthy, accumulating stake
                        upd = self.train_candidate(model, self.train_loaders[np.random.randint(len(self.train_loaders))])
                        is_malicious_update.append(False)
                    
                    # Evaluate
                    qual = self.evaluate_update(model, upd)
                    updates.append(upd)
                    qualities.append(qual)
                    
                return updates, qualities, corruption_scores, is_malicious_update

            attack_active = round_num >= self.config['attack_start_round']
            
            # --- Calculate Committee Capture Status BEFORE generating updates ---
            num_attackers_bdfl = sum(1 for v in committee_bdfl if v['is_attacker'])
            committee_captured_bdfl = attack_active and (num_attackers_bdfl > (2 / 3) * self.committee_size)
            
            num_attackers_ours = sum(1 for v in committee_ours if v['is_attacker'])
            committee_captured_ours = attack_active and (num_attackers_ours > (2 / 3) * self.committee_size)
            
            # --- BlockDFL Process ---
            bdfl_updates, bdfl_qualities, bdfl_scores, bdfl_is_mal = process_candidates(
                self.model_blockdfl, groups_bdfl, attack_active, committee_captured_bdfl
            )
            
            # Determine attack status for BlockDFL
            if committee_captured_bdfl:
                # ATTACK PHASE: Committee captured, selecting based on corruption
                bdfl_idx = np.argmax(bdfl_scores)
                winning_agg_bdfl = groups_bdfl[bdfl_idx]['aggregator']
                if winning_agg_bdfl['is_attacker']:
                    bdfl_attack_status = "ATK:CAPTURED_MAL_FLIP"
                else:
                    bdfl_attack_status = "ATK:CAPTURED_HON_NEPO"
            else:
                # LATENT PHASE: Honest selection
                bdfl_idx = np.argmax(bdfl_qualities)
                winning_agg_bdfl = groups_bdfl[bdfl_idx]['aggregator']
                if winning_agg_bdfl['is_attacker']:
                    bdfl_attack_status = "LATENT:MAL_STEALTH"
                else:
                    bdfl_attack_status = "LATENT:HON_STEALTH"
                
            # BlockDFL Rewards: Everyone in Committee + Winning Group gets Reward
            # No Slashing in BlockDFL
            winning_group_bdfl = groups_bdfl[bdfl_idx]
            
            # Reward Committee (Conditional based on capture status)
            if committee_captured_bdfl:
                # Attack Success: Only malicious verifiers get rewards
                for v in committee_bdfl:
                    if v['is_attacker']:
                        v['stack'] += self.rewards['verifier']
                    # Honest verifiers get nothing when committee is captured
            else:
                # Normal operation: All committee members get rewards
                for v in committee_bdfl:
                    v['stack'] += self.rewards['verifier']
            
            # Reward Winning Aggregator & Providers
            winning_group_bdfl['aggregator']['stack'] += self.rewards['aggregator']
            for p in winning_group_bdfl['providers']:
                p['stack'] += self.rewards['provider']
                
            blockdfl_selected_update = bdfl_updates[bdfl_idx]
            
            # --- Ours Process ---
            ours_updates, ours_qualities, ours_scores, ours_is_mal = process_candidates(
                self.model_ours, groups_ours, attack_active, committee_captured_ours
            )
            
            # Determine attack status for Ours
            if committee_captured_ours:
                # ATTACK PHASE: Committee captured
                ours_idx = np.argmax(ours_scores)
                winning_agg_ours = groups_ours[ours_idx]['aggregator']
                if winning_agg_ours['is_attacker']:
                    ours_attack_status = "ATK:CAPTURED_MAL_FLIP"
                else:
                    ours_attack_status = "ATK:CAPTURED_HON_NEPO"
                
                # SLASHING MECHANISM (Ours only)
                num_slashed = 0
                for v in committee_ours:
                    if v['is_attacker']:
                        if self.slash_penalty == 'full':
                            v['stack'] = 0.0
                        else:
                            v['stack'] = max(0, v['stack'] - float(self.slash_penalty))
                        num_slashed += 1
                
                if num_slashed > 0:
                    ours_attack_status += f" (SLASHED {num_slashed})"

                # Winning Group gets Rewards
                winning_group_ours = groups_ours[ours_idx]
                winning_group_ours['aggregator']['stack'] += self.rewards['aggregator']
                for p in winning_group_ours['providers']:
                    p['stack'] += self.rewards['provider']
                    
            else:
                # LATENT PHASE: Honest selection
                ours_idx = np.argmax(ours_qualities)
                winning_agg_ours = groups_ours[ours_idx]['aggregator']
                if winning_agg_ours['is_attacker']:
                    ours_attack_status = "LATENT:MAL_STEALTH"
                else:
                    ours_attack_status = "LATENT:HON_STEALTH"
                
                # Standard Rewards
                for v in committee_ours:
                    v['stack'] += self.rewards['verifier']
                
                winning_group_ours = groups_ours[ours_idx]
                winning_group_ours['aggregator']['stack'] += self.rewards['aggregator']
                for p in winning_group_ours['providers']:
                    p['stack'] += self.rewards['provider']

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

            bdfl_avg_honest, bdfl_avg_attacker = get_avg_stack(self.participants_blockdfl)
            ours_avg_honest, ours_avg_attacker = get_avg_stack(self.participants_ours)

            print(f"Round {round_num}/{self.config['total_rounds']}")
            print(f"  BlockDFL: [{bdfl_attack_status}] | Attacker Stack: {bdfl_avg_attacker:.2f} | Honest Stack: {bdfl_avg_honest:.2f}")
            print(f"  Ours:     [{ours_attack_status}] | Attacker Stack: {ours_avg_attacker:.2f} | Honest Stack: {ours_avg_honest:.2f}")
            print(f"  -> BlockDFL Acc: {blockdfl_acc:.4f}")
            print(f"  -> Ours Acc:     {ours_acc:.4f}")
            
            # Save results
            round_data = {
                "round": round_num,
                "blockdfl_accuracy": blockdfl_acc,
                "ours_accuracy": ours_acc,
                "blockdfl_attack_status_code": bdfl_attack_status,
                "ours_attack_status_code": ours_attack_status,
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

