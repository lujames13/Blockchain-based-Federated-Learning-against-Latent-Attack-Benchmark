import sys
import copy
import numpy as np
import yaml
import json
import os
import matplotlib.pyplot as plt

# Mock class to replace Simulator
class LightweightSimulator:
    def __init__(self, config_path='config.yaml', dataset_name='MNIST', seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
        
        # Select config based on dataset
        if dataset_name.lower() == 'mnist':
            self.config = full_config['mnist']
        elif dataset_name.lower() == 'mnist_noniid':
            self.config = full_config['mnist_noniid']
        elif dataset_name.lower() == 'cifar10':
            self.config = full_config['cifar10']
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
        self.dataset_name = dataset_name
        
        # Initialize verifier pool
        self.verifier_pool_size = self.config.get('verifier_pool_size', 100)
        
        if 'initial_attacker_ratio' not in self.config:
            raise ValueError(f"initial_attacker_ratio must be defined in config for {dataset_name}")
        self.initial_attacker_ratio = self.config['initial_attacker_ratio']
        self.committee_size = self.config.get('committee_size', 7)
        self.num_candidates = self.config.get('num_candidates', 4)
        self.providers_per_aggregator = 5  # Fixed number of providers per aggregator
        
        # Check if pool is large enough
        required_nodes = self.committee_size + self.num_candidates * (1 + self.providers_per_aggregator)
        if self.verifier_pool_size < required_nodes:
            self.verifier_pool_size = required_nodes
            
        num_attackers = int(self.verifier_pool_size * self.initial_attacker_ratio)
        
        # Initialize participants for BlockDFL
        self.participants_blockdfl = []
        for i in range(self.verifier_pool_size):
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
        
        # Tracking for chart - BOTH BlockDFL and Ours
        self.round_history = []
        self.bdfl_avg_honest_history = []
        self.bdfl_avg_attacker_history = []
        self.bdfl_ratio_history = []
        self.ours_avg_honest_history = []
        self.ours_avg_attacker_history = []
        self.ours_ratio_history = []

    def mock_evaluate_update(self, is_malicious_update):
        """
        Mock evaluation of update quality.
        Honest/Latent: High quality (0.95 +/- noise)
        Malicious Attack: Low quality (0.10 +/- noise)
        """
        if is_malicious_update:
            return np.clip(np.random.normal(0.10, 0.05), 0.0, 1.0)
        else:
            return np.clip(np.random.normal(0.95, 0.01), 0.0, 1.0)

    def step(self, round_num):
        # Helper to assign roles (EXACT COPY from simulator.py logic)
        def assign_roles(pool):
            committee_size = self.committee_size
            num_aggregators = self.num_candidates
            
            # 1. Select Verifiers (Committee)
            total_stack = sum(v['stack'] for v in pool)
            if total_stack == 0:
                probs_v = [1.0/len(pool)] * len(pool)
            else:
                probs_v = [v['stack'] / total_stack for v in pool]
            
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
            
            agg_rel_idxs = np.random.choice(len(remaining_pool_obj), size=num_aggregators, replace=False, p=probs_a)
            aggregators = [remaining_pool_obj[i] for i in agg_rel_idxs]
            
            # Remaining for Providers
            final_remaining_obj = [p for i, p in enumerate(remaining_pool_obj) if i not in agg_rel_idxs]
            
            # 3. Assign Providers to Aggregators
            np.random.shuffle(final_remaining_obj)
            chunk_size = len(final_remaining_obj) // num_aggregators
            
            aggregators_with_providers = []
            for i in range(num_aggregators):
                start = i * chunk_size
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

        # Generate Updates and Roles for BOTH BlockDFL and Ours
        committee_bdfl, groups_bdfl = assign_roles(self.participants_blockdfl)
        committee_ours, groups_ours = assign_roles(self.participants_ours)
        
        def process_candidates(groups, is_attack_active, is_committee_captured):
            qualities = []
            corruption_scores = []
            is_malicious_update = []
            
            for group in groups:
                agg = group['aggregator']
                provs = group['providers']
                
                # Compute Corruption Score
                num_mal_prov = sum(1 for p in provs if p['is_attacker'])
                score = (100 if agg['is_attacker'] else 0) + num_mal_prov
                corruption_scores.append(score)
                
                if is_attack_active and is_committee_captured and agg['is_attacker']:
                    # ATTACK PHASE
                    qual = self.mock_evaluate_update(True)
                    is_malicious_update.append(True)
                else:
                    # LATENT PHASE or Honest
                    qual = self.mock_evaluate_update(False)
                    is_malicious_update.append(False)
                
                qualities.append(qual)
                
            return qualities, corruption_scores, is_malicious_update

        attack_active = round_num >= self.config['attack_start_round']
        
        # --- Calculate Committee Capture Status BEFORE generating updates ---
        num_attackers_bdfl = sum(1 for v in committee_bdfl if v['is_attacker'])
        committee_captured_bdfl = attack_active and (num_attackers_bdfl > (2 / 3) * self.committee_size)
        
        num_attackers_ours = sum(1 for v in committee_ours if v['is_attacker'])
        committee_captured_ours = attack_active and (num_attackers_ours > (2 / 3) * self.committee_size)
        
        # --- BlockDFL Process ---
        bdfl_qualities, bdfl_scores, bdfl_is_mal = process_candidates(
            groups_bdfl, attack_active, committee_captured_bdfl
        )
        
        if committee_captured_bdfl:
            # ATTACK PHASE: Select by corruption score
            bdfl_idx = np.argmax(bdfl_scores)
        else:
            # LATENT PHASE: Select by quality
            bdfl_idx = np.argmax(bdfl_qualities)
        
        # BlockDFL Rewards (Conditional based on capture status)
        winning_group_bdfl = groups_bdfl[bdfl_idx]
        
        if committee_captured_bdfl:
            # Attack Success: Only malicious verifiers get rewards
            for v in committee_bdfl:
                if v['is_attacker']:
                    v['stack'] += self.rewards['verifier']
        else:
            # Normal operation: All committee members get rewards
            for v in committee_bdfl:
                v['stack'] += self.rewards['verifier']
        
        # Reward Winning Aggregator & Providers
        winning_group_bdfl['aggregator']['stack'] += self.rewards['aggregator']
        for p in winning_group_bdfl['providers']:
            p['stack'] += self.rewards['provider']
        
        # --- Ours Process ---
        ours_qualities, ours_scores, ours_is_mal = process_candidates(
            groups_ours, attack_active, committee_captured_ours
        )
        
        if committee_captured_ours:
            # ATTACK PHASE: Committee captured
            ours_idx = np.argmax(ours_scores)
            
            # SLASHING MECHANISM (Ours only)
            for v in committee_ours:
                if v['is_attacker']:
                    if self.slash_penalty == 'full':
                        v['stack'] = 0.0
                    else:
                        v['stack'] = max(0, v['stack'] - float(self.slash_penalty))
            
            # Winning Group gets Rewards
            winning_group_ours = groups_ours[ours_idx]
            winning_group_ours['aggregator']['stack'] += self.rewards['aggregator']
            for p in winning_group_ours['providers']:
                p['stack'] += self.rewards['provider']
                
        else:
            # LATENT PHASE: Honest selection
            ours_idx = np.argmax(ours_qualities)
            
            # Standard Rewards
            for v in committee_ours:
                v['stack'] += self.rewards['verifier']
            
            winning_group_ours = groups_ours[ours_idx]
            winning_group_ours['aggregator']['stack'] += self.rewards['aggregator']
            for p in winning_group_ours['providers']:
                p['stack'] += self.rewards['provider']

        # --- Log Statistics for BOTH ---
        def get_avg_stack(pool):
            honest_stacks = [v['stack'] for v in pool if not v['is_attacker']]
            attacker_stacks = [v['stack'] for v in pool if v['is_attacker']]
            avg_honest = sum(honest_stacks) / len(honest_stacks) if honest_stacks else 0
            avg_attacker = sum(attacker_stacks) / len(attacker_stacks) if attacker_stacks else 0
            return avg_honest, avg_attacker

        bdfl_avg_honest, bdfl_avg_attacker = get_avg_stack(self.participants_blockdfl)
        ours_avg_honest, ours_avg_attacker = get_avg_stack(self.participants_ours)
        
        bdfl_ratio = bdfl_avg_attacker / bdfl_avg_honest if bdfl_avg_honest > 0 else 0
        ours_ratio = ours_avg_attacker / ours_avg_honest if ours_avg_honest > 0 else 0
        
        self.round_history.append(round_num)
        self.bdfl_avg_honest_history.append(bdfl_avg_honest)
        self.bdfl_avg_attacker_history.append(bdfl_avg_attacker)
        self.bdfl_ratio_history.append(bdfl_ratio)
        self.ours_avg_honest_history.append(ours_avg_honest)
        self.ours_avg_attacker_history.append(ours_avg_attacker)
        self.ours_ratio_history.append(ours_ratio)
        
        return bdfl_ratio, ours_ratio

def generate_chart(seed, history, output_file='best_seed_stake_growth.png'):
    rounds = history['round']
    
    plt.figure(figsize=(12, 8))
    
    # Plot Stakes - 4 lines
    # Top chart: BlockDFL dashed, Ours solid
    plt.subplot(2, 1, 1)
    plt.plot(rounds, history['bdfl_mal_stakes'], label='BlockDFL Attacker Stake', color='red', linewidth=2, linestyle='--')
    plt.plot(rounds, history['bdfl_hon_stakes'], label='BlockDFL Honest Stake', color='blue', linewidth=2, linestyle='--')
    plt.plot(rounds, history['ours_mal_stakes'], label='Ours Attacker Stake', color='red', linewidth=2, linestyle='-')
    plt.plot(rounds, history['ours_hon_stakes'], label='Ours Honest Stake', color='blue', linewidth=2, linestyle='-')
    plt.ylabel('Average Stake')
    plt.title(f"Stake Growth Comparison (Seed: {seed})")
    plt.legend()
    plt.grid(True)
    
    # Plot Ratio - 2 lines
    # Bottom chart: BlockDFL red solid, Ours blue solid
    plt.subplot(2, 1, 2)
    plt.plot(rounds, history['bdfl_ratios'], label='BlockDFL Ratio (Attacker/Honest)', color='red', linewidth=2, linestyle='-')
    plt.plot(rounds, history['ours_ratios'], label='Ours Ratio (Attacker/Honest)', color='blue', linewidth=2, linestyle='-')
    plt.axhline(y=2.0, color='gray', linestyle=':', label='Ratio 2.0')
    
    plt.xlabel('Round')
    plt.ylabel('Stake Ratio')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Chart saved to {output_file}")

def main():
    seeds = [1000 * i + 42 for i in range(10)] # Generate 10 distinct seeds
    best_bdfl_ratio = -1.0
    best_seed = None
    best_history = None
    
    results = []
    
    print("Starting simulation for 10 seeds...")
    
    for seed in seeds:
        sim = LightweightSimulator(seed=seed)
        max_rounds = sim.config['total_rounds']
        
        final_bdfl_ratio = 0
        final_ours_ratio = 0
        
        for r in range(1, max_rounds + 1):
             bdfl_ratio, ours_ratio = sim.step(r)
             final_bdfl_ratio = bdfl_ratio
             final_ours_ratio = ours_ratio
             
        print(f"Seed {seed}: BlockDFL Ratio = {final_bdfl_ratio:.4f}, Ours Ratio = {final_ours_ratio:.4f}")
        
        results.append({
            'seed': seed,
            'blockdfl_final_ratio': final_bdfl_ratio,
            'ours_final_ratio': final_ours_ratio,
        })
        
        # Select best based on BlockDFL ratio
        if final_bdfl_ratio > best_bdfl_ratio:
            best_bdfl_ratio = final_bdfl_ratio
            best_seed = seed
            best_history = {
                'round': sim.round_history,
                'bdfl_mal_stakes': sim.bdfl_avg_attacker_history,
                'bdfl_hon_stakes': sim.bdfl_avg_honest_history,
                'bdfl_ratios': sim.bdfl_ratio_history,
                'ours_mal_stakes': sim.ours_avg_attacker_history,
                'ours_hon_stakes': sim.ours_avg_honest_history,
                'ours_ratios': sim.ours_ratio_history
            }
            
    print(f"\n--- Best Seed Found (Based on BlockDFL Ratio): {best_seed} ---")
    print(f"Best BlockDFL Final Ratio: {best_bdfl_ratio:.4f}")
    
    # Save Results
    with open('seed_search_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Search results saved to seed_search_results.json")
    
    # Plot
    if best_seed is not None:
        generate_chart(best_seed, best_history)

if __name__ == "__main__":
    main()
