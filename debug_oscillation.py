import numpy as np
import yaml
from fast_stake_engine import FastStakeSimulation

def run_diagnostic():
    # Load Optimal Config
    with open('optimal_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract flat config or nested
    if 'mnist' in config:
        params = config['mnist']
        rewards = params['rewards']
        num_agg = params['num_candidates']
        prov_per = params['extra_params']['providers_per_aggregator']
    else:
        rewards = config['rewards']
        num_agg = config['num_candidates']
        prov_per = config['providers_per_aggregator']
        
    print("Debug Config:", rewards, num_agg, prov_per)
    
    engine = FastStakeSimulation(
        total_nodes=100, 
        attacker_ratio=0.3,
        committee_size=7,
        num_aggregators=num_agg,
        providers_per_aggregator=prov_per,
        rewards=rewards,
        attack_start_round=10
    )
    
    print(f"{'Round':<6} | {'Ratio':<6} | {'Captured?':<9} | {'Committee':<10} | {'Winner':<8}")
    print("-" * 60)
    
    for r in range(1, 501):
        # We need to peek inside 'step' or modify engine to return extra info
        # Or just look at state before/after?
        # Let's inspect BEFORE step
        
        # 1. Assign Roles
        committee, groups = engine.assign_roles()
        num_mal_committee = sum(1 for p in committee if p['is_attacker'])
        is_captured = (num_mal_committee > (2/3) * engine.committee_size)
        
        # 2. Determine Winner (Re-implement simple logic to log it)
        attack_active = r >= 10
        winner_is_mal = False
        
        if attack_active and is_captured:
            # Attack Mode: Picks malicious
            # We assume at least one mal group exists if captured? Not guaranteed but likely
            # In engine logic, it finds best score.
            pass # We let the engine do the update, we just want to know IF it was captured
        else:
            # Latent Mode: Random
            pass
            
        # Run Step
        # Note: calling engine.step() recalculates roles, so our peek above is separate from actual step
        # This adds noise. Ideally we modify engine, but for now let's just observe Ratio trend vs Capture Probability
        
        # To strictly correlate, we'll rely on the engine's internal step.
        # But we can't see "is_captured" from outside using current API.
        # I will hot-patch the engine instance to log useful info.
        
        # Actually, let's just update the loop to use the logic manually so we see everything
        
        # MANUAL STEP with LOGGING
        # committee, groups = engine.assign_roles() # Already called above
        
        # Check Capture
        # num_mal_committee... defined above
        
        winning_group = None
        if attack_active and is_captured:
            # Attack Logic
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
            winning_group = best_candidates[np.random.randint(len(best_candidates))]
            winner_type = "MAL-ATK" if winning_group['aggregator']['is_attacker'] else "HON-ATK?" 
            # Note: Honest agg could win if no mal agg exists? (unlikely if captured, but possible)
        else:
            # Latent Logic
            winning_group = groups[np.random.randint(len(groups))]
            winner_type = "MAL-RND" if winning_group['aggregator']['is_attacker'] else "HON-RND"
            
        # Apply Rewards
        for p in committee: p['stake'] += rewards['verifier']
        winning_group['aggregator']['stake'] += rewards['aggregator']
        for p in winning_group['providers']: p['stake'] += rewards['provider']
        
        # Stats
        h, m = engine.get_avg_stakes()
        ratio = m/h if h>0 else 0
        
        if r % 20 == 0 or r > 380: # Log more around the drop area
             print(f"{r:<6} | {ratio:<6.3f} | {str(is_captured):<9} | {num_mal_committee:<10} | {winner_type:<8}")

if __name__ == "__main__":
    run_diagnostic()
