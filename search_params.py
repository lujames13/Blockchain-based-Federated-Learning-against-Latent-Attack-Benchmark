"""
Parameter search with constraint: initial_attacker_ratio <= 0.3
尋找在攻擊者比例 ≤ 30% 限制下能達成委員會佔領的參數
"""

from quick_stake_sim import QuickStakeSimulator, SimConfig
import itertools

def search_with_constraint():
    """搜尋參數，限制攻擊者比例 <= 0.3"""
    
    # 參數範圍（攻擊者比例限制在 30% 以下）
    attacker_ratios = [0.25, 0.28, 0.30]
    committee_sizes = [3, 5, 7]
    stack_rewards = [0.2, 0.3, 0.4, 0.5]
    attack_starts = [5, 10, 15, 20]
    pool_sizes = [10, 12, 15, 20]
    total_rounds_options = [300, 500, 800, 1000]
    
    print("=" * 120)
    print("Searching with CONSTRAINT: initial_attacker_ratio <= 0.3")
    print("=" * 120)
    
    all_results = []
    total_tests = 0
    
    for total_rounds, pool_size, attacker_ratio, committee_size, stack_reward, attack_start in \
        itertools.product(total_rounds_options, pool_sizes, attacker_ratios, 
                         committee_sizes, stack_rewards, attack_starts):
        
        # 跳過不合理的組合
        if committee_size > pool_size * 0.5:
            continue
        if attack_start > total_rounds * 0.2:  # 攻擊開始不應太晚
            continue
        
        total_tests += 1
        
        config = SimConfig(
            total_rounds=total_rounds,
            attack_start_round=attack_start,
            verifier_pool_size=pool_size,
            initial_attacker_ratio=attacker_ratio,
            committee_size=committee_size,
            stack_reward=stack_reward
        )
        
        sim = QuickStakeSimulator(config)
        sim.run(verbose=False)
        summary = sim.get_summary()
        
        all_results.append({
            'config': config,
            'summary': summary,
            'dominated': summary['committee_dominated']
        })
        
        # 即時顯示找到的好配置
        if summary['last_50_rounds_attacker_committee_ratio'] > 0.7:  # 70% 以上就顯示
            status = "✓✓✓ DOMINATED" if summary['committee_dominated'] else "✓✓ STRONG"
            print(f"{status}: rounds={total_rounds}, pool={pool_size}, "
                  f"attackers={attacker_ratio:.0%}, committee={committee_size}, "
                  f"reward={stack_reward}, start={attack_start}")
            print(f"        → Committee: {summary['last_50_rounds_attacker_committee_ratio']:.1%}, "
                  f"Stack Ratio: {summary['stack_ratio']:.2f}, "
                  f"Success: {summary['attack_success_rate']:.1%}\n")
    
    print("\n" + "=" * 120)
    print(f"Search Complete: Tested {total_tests} configurations")
    print("=" * 120)
    
    # 分析結果
    dominated = [r for r in all_results if r['dominated']]
    strong = [r for r in all_results if r['summary']['last_50_rounds_attacker_committee_ratio'] > 0.7]
    
    if dominated:
        print(f"\n✓ Found {len(dominated)} configurations achieving DOMINATION (>80%)")
        
        # 按委員會佔領率排序
        dominated.sort(
            key=lambda x: x['summary']['last_50_rounds_attacker_committee_ratio'],
            reverse=True
        )
        
        print("\n" + "=" * 120)
        print("TOP 10 DOMINATING CONFIGURATIONS:")
        print("=" * 120)
        
        for i, result in enumerate(dominated[:10], 1):
            config = result['config']
            summary = result['summary']
            print(f"\n{i}. Rounds={config.total_rounds}, Pool={config.verifier_pool_size}, "
                  f"Attackers={config.initial_attacker_ratio:.0%}, "
                  f"Committee={config.committee_size}, "
                  f"Reward={config.stack_reward}, Start={config.attack_start_round}")
            print(f"   Committee Ratio: {summary['last_50_rounds_attacker_committee_ratio']:.1%} | "
                  f"Stack Ratio: {summary['stack_ratio']:.2f} | "
                  f"Attack Success: {summary['attack_success_rate']:.1%}")
        
        # 找出最短輪數達成的配置
        dominated_by_rounds = sorted(dominated, key=lambda x: x['config'].total_rounds)
        best_short = dominated_by_rounds[0]
        
        print("\n" + "=" * 120)
        print("RECOMMENDED: Shortest Training to Achieve Domination")
        print("=" * 120)
        print(f"""
mnist:
  dataset: MNIST
  total_rounds: {best_short['config'].total_rounds}
  attack_start_round: {best_short['config'].attack_start_round}
  verifier_pool_size: {best_short['config'].verifier_pool_size}
  initial_attacker_ratio: {best_short['config'].initial_attacker_ratio}
  committee_size: {best_short['config'].committee_size}
  stack_reward: {best_short['config'].stack_reward}
  
# Expected Results:
# - Last 50 rounds committee ratio: {best_short['summary']['last_50_rounds_attacker_committee_ratio']:.1%}
# - Final stack ratio (attacker/honest): {best_short['summary']['stack_ratio']:.2f}
# - Attack success rate: {best_short['summary']['attack_success_rate']:.1%}
""")
        
        return best_short['config']
        
    elif strong:
        print(f"\n⚠ No configuration achieved >80% domination")
        print(f"But found {len(strong)} configurations with >70% committee ratio\n")
        
        strong.sort(
            key=lambda x: x['summary']['last_50_rounds_attacker_committee_ratio'],
            reverse=True
        )
        
        print("TOP 5 STRONG CONFIGURATIONS (>70%):")
        print("-" * 120)
        
        for i, result in enumerate(strong[:5], 1):
            config = result['config']
            summary = result['summary']
            print(f"{i}. Rounds={config.total_rounds}, Pool={config.verifier_pool_size}, "
                  f"Attackers={config.initial_attacker_ratio:.0%}, "
                  f"Committee={config.committee_size}, "
                  f"Reward={config.stack_reward}, Start={config.attack_start_round}")
            print(f"   → {summary['last_50_rounds_attacker_committee_ratio']:.1%} committee, "
                  f"{summary['stack_ratio']:.2f}x stack, "
                  f"{summary['attack_success_rate']:.1%} success\n")
        
        best = strong[0]
        print("\nRECOMMENDED CONFIG (Best available):")
        print(f"""
mnist:
  total_rounds: {best['config'].total_rounds}
  attack_start_round: {best['config'].attack_start_round}
  verifier_pool_size: {best['config'].verifier_pool_size}
  initial_attacker_ratio: {best['config'].initial_attacker_ratio}
  committee_size: {best['config'].committee_size}
  stack_reward: {best['config'].stack_reward}
""")
        return best['config']
    else:
        print("\n✗ No configuration achieved >70% committee ratio")
        print("\nRecommendations:")
        print("  1. Extend total_rounds to 1500-2000")
        print("  2. Use smaller committee_size (3)")
        print("  3. Use higher stack_reward (0.5-0.6)")
        print("  4. Start attack very early (round 5)")
        return None

if __name__ == "__main__":
    print("Starting parameter search...")
    print("This may take 1-2 minutes...\n")
    
    best_config = search_with_constraint()
    
    if best_config:
        print("\n" + "=" * 120)
        print("Running detailed simulation with RECOMMENDED config...")
        print("=" * 120)
        
        sim = QuickStakeSimulator(best_config)
        sim.run(verbose=True)
        summary = sim.get_summary()
        
        print("\n" + "=" * 120)
        print("FINAL VERIFICATION:")
        print("=" * 120)
        print(f"✓ Last 50 rounds committee ratio: {summary['last_50_rounds_attacker_committee_ratio']:.1%}")
        print(f"✓ Final stack ratio: {summary['stack_ratio']:.2f}")
        print(f"✓ Attack success rate: {summary['attack_success_rate']:.1%}")
        print(f"✓ Committee dominated: {'YES' if summary['committee_dominated'] else 'NO'}")
        print("=" * 120)
        
        sim.plot(save_path='results/constrained_best_config.png')
        print("\n✓ Plot saved to results/constrained_best_config.png")
