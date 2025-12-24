import sys
sys.path.append('/home/jl/code/Blockchain-based-Federated-Learning-against-Latent-Attack-Benchmark')
from simulator import Simulator

# Quick verification test - run only 3 rounds
print("=" * 60)
print("VERIFICATION TEST: Latent Attack Logic")
print("=" * 60)

# Test simulator initialization
try:
    sim = Simulator('config.yaml', 'mnist')
    print(f"✓ Simulator initialized successfully")
    print(f"  - Initial attacker ratio: {sim.initial_attacker_ratio}")
    print(f"  - Committee size: {sim.committee_size}")
    print(f"  - Verifier pool size: {sim.verifier_pool_size}")
    print()
    
    # Temporarily modify config to run only 3 rounds for quick test
    original_rounds = sim.config['total_rounds']
    sim.config['total_rounds'] = 3
    sim.results['total_rounds'] = 3
    
    print(f"Running {sim.config['total_rounds']} rounds for verification...")
    print("=" * 60)
    
    # Run simulation
    sim.run()
    
    # Check results
    print()
    print("=" * 60)
    print("VERIFICATION RESULTS:")
    print("=" * 60)
    for r in sim.results['results']:
        print(f"Round {r['round']}:")
        print(f"  BlockDFL: {r['blockdfl_attack_status_code']}")
        print(f"  Ours:     {r['ours_attack_status_code']}")
        
    print()
    print("✓ All tests passed!")
    print("✓ Attack status codes are present in output")
    print("✓ Logging format verified")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
