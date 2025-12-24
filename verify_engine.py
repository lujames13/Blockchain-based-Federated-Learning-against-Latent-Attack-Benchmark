from fast_stake_engine import FastStakeSimulation

def test_engine():
    print("Testing FastStakeSimulation...")
    
    # Test strict constraints
    engine = FastStakeSimulation(
        total_nodes=100, 
        attacker_ratio=0.3, 
        committee_size=7, 
        num_aggregators=4, 
        providers_per_aggregator=5
    )
    
    print("Initial Stakes (Avg Honest, Avg Attacker):", engine.get_avg_stakes())
    
    # Run 1 step and check roles
    committee, groups = engine.assign_roles()
    
    print(f"Committee Size: {len(committee)} (Expected: 7)")
    print(f"Num Groups: {len(groups)} (Expected: 4)")
    
    for i, g in enumerate(groups):
        print(f"  Group {i}: {len(g['providers'])} Providers (Expected: 5)")
        # Verify no overlap
        # (visual check mostly, logic ensures it)
        
    # Run simulation
    capture_round, final_stakes = engine.run_simulation(max_rounds=50)
    print(f"Simulation Result: Capture at Round {capture_round}")
    print(f"Final Stakes: {final_stakes}")
    
    # Check if stakes changed
    if final_stakes[0] == 1.0 and final_stakes[1] == 1.0:
        print("ERROR: Stakes did not change!")
    else:
        print("SUCCESS: Stakes updated.")

if __name__ == "__main__":
    test_engine()
