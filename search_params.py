import multiprocessing
import itertools
import yaml
import matplotlib.pyplot as plt
import numpy as np
from fast_stake_engine import FastStakeSimulation

# Search Space
REWARDS_VERIFIER = [0.1, 0.5, 1.0, 2.0]
REWARDS_AGGREGATOR = [0.01, 0.1, 0.2, 1.0]
REWARDS_PROVIDER = [0.01, 0.05, 0.1, 1.0]
NUM_AGGREGATORS = [4, 10, 20]
PROVIDERS_PER_AGG = [5, 10, 20]

# Constants
SIMULATIONS_PER_CONFIG = 10
MAX_ROUNDS = 800
TOTAL_NODES = 100
COMMITTEE_SIZE = 7
TARGET_RATIO = 2.0

def run_simulation_batch(args):
    """
    Runs a batch of simulations for a single configuration.
    args: (verifier_r, agg_r, prov_r, num_agg, prov_per_agg)
    """
    v_r, a_r, p_r, n_agg, p_per_agg = args
    
    # Validation Check
    required_nodes = COMMITTEE_SIZE + n_agg + (n_agg * p_per_agg)
    if required_nodes > TOTAL_NODES:
        return None # Invalid config
        
    rewards = {'verifier': v_r, 'aggregator': a_r, 'provider': p_r}
    
    dominance_rounds = [] 
    final_stake_ratios = [] 
    capture_rounds = []
    seeds = []
    
    # We'll use a local random generator to ensure reproducibility per process if needed,
    # but for simplicity we'll just record the global seed start for each sub-sim.
    base_seed = hash((v_r, a_r, p_r, n_agg, p_per_agg)) % (2**32)
    
    for i in range(SIMULATIONS_PER_CONFIG):
        sim_seed = (base_seed + i) % (2**32)
        np.random.seed(sim_seed)
        seeds.append(sim_seed)
        
        engine = FastStakeSimulation(
            total_nodes=TOTAL_NODES,
            committee_size=COMMITTEE_SIZE,
            num_aggregators=n_agg,
            providers_per_aggregator=p_per_agg,
            rewards=rewards,
            attack_start_round=10
        )
        
        dom_round = MAX_ROUNDS + 1 
        captured_at = MAX_ROUNDS + 1
        
        for r in range(1, MAX_ROUNDS + 1):
            is_captured = engine.step(r)
            avg_hon, avg_mal = engine.get_avg_stakes()
            
            if is_captured and r >= 10 and captured_at > MAX_ROUNDS:
                captured_at = r
            
            if avg_hon > 0:
                ratio = avg_mal / avg_hon
                if ratio > TARGET_RATIO and dom_round > MAX_ROUNDS:
                    dom_round = r
            
        final_ratio = avg_mal / avg_hon if avg_hon > 0 else 0
        dominance_rounds.append(dom_round)
        capture_rounds.append(captured_at)
        final_stake_ratios.append(final_ratio)
        
    avg_dominance = sum(dominance_rounds) / len(dominance_rounds)
    avg_capture = sum(capture_rounds) / len(capture_rounds)
    avg_ratio = sum(final_stake_ratios) / len(final_stake_ratios)
    
    # Find the best seed in this batch
    # Best seed is the one that reached dominance earliest
    best_idx = np.argmin(dominance_rounds)
    # If tie, use max final ratio
    if dominance_rounds[best_idx] > MAX_ROUNDS:
        best_idx = np.argmax(final_stake_ratios)
    
    return {
        'params': {
            'rewards': rewards,
            'num_candidates': n_agg,
            'providers_per_aggregator': p_per_agg
        },
        'avg_dominance_round': avg_dominance,
        'avg_capture_round': avg_capture,
        'avg_stake_ratio': avg_ratio,
        'best_seed': int(seeds[best_idx])
    }

def main():
    print(f"Generating parameter combinations (Target Ratio > {TARGET_RATIO})...")
    combinations = list(itertools.product(
        REWARDS_VERIFIER, 
        REWARDS_AGGREGATOR, 
        REWARDS_PROVIDER,
        NUM_AGGREGATORS,
        PROVIDERS_PER_AGG
    ))
    
    print(f"Total combinations to check: {len(combinations)}")
    
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    results = pool.map(run_simulation_batch, combinations)
    pool.close()
    pool.join()
    
    valid_results = [r for r in results if r is not None]
    print(f"Valid configurations tested: {len(valid_results)}")
    
    if not valid_results:
        print("No valid configurations found!")
        return

    sorted_results = sorted(valid_results, key=lambda x: (x['avg_dominance_round'], -x['avg_stake_ratio']))
    
    best_result = sorted_results[0]
    best_params = best_result['params']
    best_seed = best_result['best_seed']
    
    if best_result['avg_dominance_round'] > MAX_ROUNDS:
        print("WARNING: Target Ratio > 2.0 was NEVER reached within max rounds by any config.")
    
    print("\n--- Best Configuration Found ---")
    print(f"Avg Dominance Round (> {TARGET_RATIO}x): {best_result['avg_dominance_round']:.2f}")
    print(f"Best Seed: {best_seed}")
    print(f"Parameters: {best_params}")
    
    # Save config
    output_config = {
        'mnist': { 
             'rewards': best_params['rewards'],
             'num_candidates': best_params['num_candidates'],
             'extra_params': {
                 'providers_per_aggregator': best_params['providers_per_aggregator'],
                 'best_seed': best_seed
             }
        },
        'rewards': best_params['rewards'],
        'num_candidates': best_params['num_candidates'],
        'providers_per_aggregator': best_params['providers_per_aggregator'],
        'best_seed': best_seed
    }
    
    with open('optimal_config.yaml', 'w') as f:
        yaml.dump(output_config, f)
    print("Saved to optimal_config.yaml")
    
    # Generate Chart with the specific best seed
    generate_chart(best_params, best_seed)

def generate_chart(params, seed):
    print(f"Plotting with seed: {seed}")
    np.random.seed(seed)
    engine = FastStakeSimulation(
        total_nodes=TOTAL_NODES,
        committee_size=COMMITTEE_SIZE,
        num_aggregators=params['num_candidates'],
        providers_per_aggregator=params['providers_per_aggregator'],
        rewards=params['rewards'],
        attack_start_round=10
    )
    
    rounds = []
    ratios = []
    mal_stakes = []
    hon_stakes = []
    local_dom_round = MAX_ROUNDS + 1
    
    for r in range(1, 801): # Record up to 800 rounds
        engine.step(r)
        h, m = engine.get_avg_stakes()
        rounds.append(r)
        mal_stakes.append(m)
        hon_stakes.append(h)
        ratio = m / h if h > 0 else 0
        ratios.append(ratio)
        if ratio > TARGET_RATIO and local_dom_round > MAX_ROUNDS:
            local_dom_round = r
             
    plt.figure(figsize=(10, 6))
    
    # Plot Stakes
    plt.subplot(2, 1, 1)
    plt.plot(rounds, mal_stakes, label='Avg Attacker Stake', color='red', linewidth=2)
    plt.plot(rounds, hon_stakes, label='Avg Honest Stake', color='blue', linewidth=2)
    plt.ylabel('Average Stake')
    plt.title(f"Stake Growth (Seed: {seed}, Target: Ratio > {TARGET_RATIO})")
    plt.legend()
    plt.grid(True)
    
    # Plot Ratio
    plt.subplot(2, 1, 2)
    plt.plot(rounds, ratios, label='Stake Ratio (Attacker/Honest)', color='purple', linewidth=2)
    plt.axhline(y=TARGET_RATIO, color='green', linestyle='--', label=f'Target Ratio {TARGET_RATIO}')
    if local_dom_round <= 800:
        plt.axvline(x=local_dom_round, color='orange', linestyle=':', label=f'Dominance @ {local_dom_round}')
    
    plt.xlabel('Round')
    plt.ylabel('Stake Ratio')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('stake_growth_chart.png')
    print("Chart saved to stake_growth_chart.png")

if __name__ == "__main__":
    main()
