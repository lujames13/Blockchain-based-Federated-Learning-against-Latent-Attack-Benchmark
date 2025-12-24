import numpy as np
import copy

class FastStakeSimulation:
    def __init__(self, 
                 total_nodes=100, 
                 attacker_ratio=0.3, 
                 committee_size=7, 
                 num_aggregators=4, 
                 providers_per_aggregator=5,
                 rewards={'verifier': 0.1, 'aggregator': 0.2, 'provider': 0.05},
                 attack_start_round=10):
        
        self.total_nodes = total_nodes
        self.attacker_ratio = attacker_ratio
        self.committee_size = committee_size
        self.num_aggregators = num_aggregators
        self.providers_per_aggregator = providers_per_aggregator
        self.rewards = rewards
        self.attack_start_round = attack_start_round
        
        self.num_attackers = int(total_nodes * attacker_ratio)
        
        # Initialize participants
        # We only track 'is_attacker' and 'stake' for speed
        # IDs are indices 0 to N-1
        self.participants = []
        for i in range(total_nodes):
            self.participants.append({
                'id': i,
                'is_attacker': i < self.num_attackers,
                'stake': 1.0
            })
            
    def get_avg_stakes(self):
        attacker_stacks = [p['stake'] for p in self.participants if p['is_attacker']]
        honest_stacks = [p['stake'] for p in self.participants if not p['is_attacker']]
        
        avg_attacker = sum(attacker_stacks) / len(attacker_stacks) if attacker_stacks else 0.0
        avg_honest = sum(honest_stacks) / len(honest_stacks) if honest_stacks else 0.0
        return avg_honest, avg_attacker

    def assign_roles(self):
        """
        Mimics simulator.py logic but with strict provider limits.
        """
        pool = self.participants
        
        # 1. Select Committee (Weighted by Stake)
        total_stack = sum(p['stake'] for p in pool)
        probs_v = [p['stake'] / total_stack for p in pool] if total_stack > 0 else [1.0/len(pool)] * len(pool)
        
        committee_idxs = np.random.choice(len(pool), size=self.committee_size, replace=False, p=probs_v)
        committee = [pool[i] for i in committee_idxs]
        
        # Remaining for Aggregators
        remaining_idxs = [i for i in range(len(pool)) if i not in committee_idxs]
        remaining_pool = [pool[i] for i in remaining_idxs]
        
        # 2. Select Aggregators (Weighted by Stake from remaining)
        total_stack_rem = sum(p['stake'] for p in remaining_pool)
        probs_a = [p['stake'] / total_stack_rem for p in remaining_pool] if total_stack_rem > 0 else [1.0/len(remaining_pool)] * len(remaining_pool)
        
        # Sample relative indices
        agg_rel_idxs = np.random.choice(len(remaining_pool), size=self.num_aggregators, replace=False, p=probs_a)
        aggregators = [remaining_pool[i] for i in agg_rel_idxs]
        
        # Remaining for Providers
        # Filter out selected aggregators from remaining_pool
        final_remaining = [p for i, p in enumerate(remaining_pool) if i not in agg_rel_idxs]
        
        # 3. Select Providers (Randomly from remaining)
        np.random.shuffle(final_remaining)
        
        # Strict Limit: Assign exactly providers_per_aggregator to each aggregator
        # Excess nodes are idle
        groups = []
        current_provider_idx = 0
        
        for agg in aggregators:
            start = current_provider_idx
            end = start + self.providers_per_aggregator
            
            if end > len(final_remaining):
                 # Not enough providers left to fill this group fully
                 # (Start of implementation note: User said P=5 is constant constraint. 
                 # We skip if we can't fill? Or fill partially? 
                 # The constraint implies "Every group has 5". 
                 # If we run out, we just break or have smaller group? 
                 # For safety in "Optimization", let's assume valid config ensures we have enough.
                 # If config is invalid, we might crash or fill partially. 
                 # Let's fill partially but validation should prevent this case.)
                 my_providers = final_remaining[start:]
            else:
                 my_providers = final_remaining[start:end]
            
            groups.append({
                'aggregator': agg,
                'providers': my_providers
            })
            current_provider_idx = end
            
        return committee, groups

    def step(self, round_num):
        """
        Executes one round of simulation.
        Returns: is_captured (bool)
        """
        committee, groups = self.assign_roles()
        
        # Check Capture Status
        num_mal_committee = sum(1 for p in committee if p['is_attacker'])
        is_captured = (num_mal_committee > (2/3) * self.committee_size)
        
        attack_active = round_num >= self.attack_start_round
        
        winning_group = None
        
        # Determine Winning Group
        if attack_active and is_captured:
            # ATTACK PHASE: Malicious Committee chooses Malicious-favored update
            # Strategy: Highest Corruption Score (Agg is Mal + Mal Providers)
            best_score = -1
            best_candidates = []
            
            for group in groups:
                agg = group['aggregator']
                provs = group['providers']
                score = (1 if agg['is_attacker'] else 0) + sum(1 for p in provs if p['is_attacker'])
                
                if score > best_score:
                    best_score = score
                    best_candidates = [group]
                elif score == best_score:
                    best_candidates.append(group)
            
            # Tie-break randomly (or by first found) 
            winning_group = best_candidates[np.random.randint(len(best_candidates))]
            
        else:
            # LATENT/HONEST PHASE: Random winner (assuming equal quality)
            # In latent phase, malicious nodes act honestly.
            winning_group = groups[np.random.randint(len(groups))]
            
        
        # Distributed Rewards
        # 1. Committee Reward
        if attack_active and is_captured:
            # ATTACK PHASE: Attackers control the committee and block rewards for honest members
            for p in committee:
                if p['is_attacker']:
                    p['stake'] += self.rewards['verifier']
        else:
            # LATENT/HONEST PHASE: Standard reward distribution to all members
            for p in committee:
                p['stake'] += self.rewards['verifier']
            
        # 2. Aggregator Reward
        winning_group['aggregator']['stake'] += self.rewards['aggregator']
        
        # 3. Provider Reward
        for p in winning_group['providers']:
            p['stake'] += self.rewards['provider']
            
        return is_captured

    def run_simulation(self, max_rounds=1000):
        for r in range(1, max_rounds + 1):
            is_captured = self.step(r)
            if is_captured and r >= self.attack_start_round:
                # Capture condition met: >2/3 malicious in committee AFTER attack starts
                # We return the round number of FIRST capture
                return r, self.get_avg_stakes()
        
        # If never captured
        return max_rounds, self.get_avg_stakes()
