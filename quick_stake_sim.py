"""
Quick Stake Simulation - Fast parameter testing for committee capture
不需要真正訓練模型，只模擬 stake 累積和委員會選擇
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict
import json

@dataclass
class SimConfig:
    total_rounds: int = 300
    attack_start_round: int = 20
    verifier_pool_size: int = 15
    initial_attacker_ratio: float = 0.3
    committee_size: int = 5
    stack_reward: float = 0.2
    
    def __str__(self):
        return (f"pool={self.verifier_pool_size}, "
                f"attackers={self.initial_attacker_ratio:.0%}, "
                f"committee={self.committee_size}, "
                f"reward={self.stack_reward}, "
                f"attack_start={self.attack_start_round}")

class QuickStakeSimulator:
    def __init__(self, config: SimConfig):
        self.config = config
        
        # Initialize verifier pool
        num_attackers = int(config.verifier_pool_size * config.initial_attacker_ratio)
        self.verifiers = []
        for i in range(config.verifier_pool_size):
            is_attacker = i < num_attackers
            self.verifiers.append({
                'id': i,
                'is_attacker': is_attacker,
                'stack': 1.0
            })
        
        # Results tracking
        self.history = {
            'rounds': [],
            'avg_attacker_stack': [],
            'avg_honest_stack': [],
            'attack_success_count': 0,
            'total_attack_rounds': 0,
            'attacker_committee_ratio': []  # 攻擊者在委員會中的比例
        }
    
    def select_committee(self, size: int) -> List[Dict]:
        """基於 stake 加權選擇委員會"""
        total_stack = sum(v['stack'] for v in self.verifiers)
        if total_stack == 0:
            probs = [1.0/len(self.verifiers)] * len(self.verifiers)
        else:
            probs = [v['stack'] / total_stack for v in self.verifiers]
        
        indices = np.random.choice(
            len(self.verifiers),
            size=min(size, len(self.verifiers)),
            replace=False,
            p=probs
        )
        return [self.verifiers[i] for i in indices]
    
    def run(self, verbose: bool = False):
        """運行模擬"""
        for round_num in range(1, self.config.total_rounds + 1):
            attack_active = round_num >= self.config.attack_start_round
            
            # 選擇委員會
            committee = self.select_committee(self.config.committee_size)
            num_attackers = sum(1 for v in committee if v['is_attacker'])
            num_honest = len(committee) - num_attackers
            
            # 記錄攻擊者在委員會中的比例
            attacker_ratio = num_attackers / len(committee) if len(committee) > 0 else 0
            self.history['attacker_committee_ratio'].append(attacker_ratio)
            
            # 計算總獎勵池
            total_pot = self.config.committee_size * self.config.stack_reward
            
            # 判斷攻擊是否成功
            attack_successful = attack_active and num_attackers > num_honest
            
            if attack_successful:
                # 攻擊成功：攻擊者瓜分獎勵池
                reward_per_attacker = total_pot / num_attackers
                for v in committee:
                    if v['is_attacker']:
                        v['stack'] += reward_per_attacker
                
                self.history['attack_success_count'] += 1
                status = "ATTACK SUCCESS"
            else:
                # 攻擊失敗或未攻擊：所有人平分
                for v in committee:
                    v['stack'] += self.config.stack_reward
                
                status = "ATTACK FAILED" if attack_active else "NORMAL"
            
            if attack_active:
                self.history['total_attack_rounds'] += 1
            
            # 記錄統計數據
            honest_stacks = [v['stack'] for v in self.verifiers if not v['is_attacker']]
            attacker_stacks = [v['stack'] for v in self.verifiers if v['is_attacker']]
            
            avg_honest = sum(honest_stacks) / len(honest_stacks) if honest_stacks else 0
            avg_attacker = sum(attacker_stacks) / len(attacker_stacks) if attacker_stacks else 0
            
            self.history['rounds'].append(round_num)
            self.history['avg_honest_stack'].append(avg_honest)
            self.history['avg_attacker_stack'].append(avg_attacker)
            
            # 每 50 輪或最後一輪輸出
            if verbose and (round_num % 50 == 0 or round_num == self.config.total_rounds):
                print(f"Round {round_num:3d}: {status:15s} | "
                      f"Committee: {num_attackers}/{len(committee)} attackers | "
                      f"Avg Stack - Attacker: {avg_attacker:.2f}, Honest: {avg_honest:.2f}")
    
    def get_summary(self) -> Dict:
        """獲取模擬摘要"""
        final_avg_attacker = self.history['avg_attacker_stack'][-1]
        final_avg_honest = self.history['avg_honest_stack'][-1]
        
        # 計算最後 50 輪攻擊者在委員會中的平均比例
        last_50_ratio = np.mean(self.history['attacker_committee_ratio'][-50:])
        
        return {
            'config': str(self.config),
            'final_avg_attacker_stack': final_avg_attacker,
            'final_avg_honest_stack': final_avg_honest,
            'stack_ratio': final_avg_attacker / final_avg_honest if final_avg_honest > 0 else 0,
            'attack_success_rate': (self.history['attack_success_count'] / 
                                   self.history['total_attack_rounds'] 
                                   if self.history['total_attack_rounds'] > 0 else 0),
            'last_50_rounds_attacker_committee_ratio': last_50_ratio,
            'committee_dominated': last_50_ratio > 0.8  # 80% 以上算是佔領
        }
    
    def plot(self, save_path: str = None):
        """繪製 stake 變化圖"""
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history['rounds'], self.history['avg_attacker_stack'], 
                 'r-', label='Attacker Avg Stack', linewidth=2)
        plt.plot(self.history['rounds'], self.history['avg_honest_stack'], 
                 'b-', label='Honest Avg Stack', linewidth=2)
        plt.axvline(x=self.config.attack_start_round, color='gray', 
                   linestyle='--', alpha=0.5, label='Attack Start')
        plt.xlabel('Round')
        plt.ylabel('Average Stack')
        plt.title('Stack Evolution Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['rounds'], self.history['attacker_committee_ratio'], 
                 'g-', linewidth=2)
        plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Majority Line')
        plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Domination Line')
        plt.axvline(x=self.config.attack_start_round, color='gray', 
                   linestyle='--', alpha=0.5, label='Attack Start')
        plt.xlabel('Round')
        plt.ylabel('Attacker Ratio in Committee')
        plt.title('Committee Composition')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

def test_multiple_configs():
    """測試多種配置"""
    configs = [
        # Pool=50, Committee=3, Reward=0.2, Start=0
        SimConfig(verifier_pool_size=50, initial_attacker_ratio=0.3, 
                 committee_size=3, stack_reward=0.2, attack_start_round=0, total_rounds=800),
        
        # Pool=50, Committee=3, Reward=0.1, Start=0
        SimConfig(verifier_pool_size=50, initial_attacker_ratio=0.3, 
                 committee_size=3, stack_reward=0.1, attack_start_round=0, total_rounds=800),

        # Pool=20, Committee=3, Reward=0.2, Start=0
        SimConfig(verifier_pool_size=20, initial_attacker_ratio=0.3, 
                 committee_size=3, stack_reward=0.2, attack_start_round=0, total_rounds=800),
                 
        # Pool=10, Committee=3, Reward=0.2, Start=0 (High variance might get lucky?)
        SimConfig(verifier_pool_size=10, initial_attacker_ratio=0.3, 
                 committee_size=3, stack_reward=0.2, attack_start_round=0, total_rounds=800),
    ]
    
    print("=" * 100)
    print("Testing Multiple Configurations")
    print("=" * 100)
    
    results = []
    for i, config in enumerate(configs, 1):
        print(f"\n[Config {i}] {config}")
        sim = QuickStakeSimulator(config)
        sim.run(verbose=False)
        summary = sim.get_summary()
        results.append(summary)
        
        print(f"  Final Stack Ratio (Attacker/Honest): {summary['stack_ratio']:.2f}")
        print(f"  Attack Success Rate: {summary['attack_success_rate']:.1%}")
        print(f"  Last 50 Rounds Committee Ratio: {summary['last_50_rounds_attacker_committee_ratio']:.1%}")
        print(f"  Committee Dominated (>80%): {'✓ YES' if summary['committee_dominated'] else '✗ NO'}")
    
    print("\n" + "=" * 100)
    print("Summary: Best Configurations for Committee Domination")
    print("=" * 100)
    
    dominated = [r for r in results if r['committee_dominated']]
    if dominated:
        for result in dominated:
            print(f"✓ {result['config']}")
    else:
        print("None of the tested configurations achieved committee domination.")
        print("Try increasing initial_attacker_ratio or decreasing committee_size.")

if __name__ == "__main__":
    # 快速測試多種配置
    test_multiple_configs()
    
    # 詳細查看最佳配置
    print("\n" + "=" * 100)
    print("Detailed Simulation of Best Config")
    print("=" * 100)
    
    best_config = SimConfig(
        verifier_pool_size=15,
        initial_attacker_ratio=0.45,
        committee_size=5,
        stack_reward=0.2,
        attack_start_round=20
    )
    
    print(f"\nRunning: {best_config}\n")
    sim = QuickStakeSimulator(best_config)
    sim.run(verbose=True)
    
    summary = sim.get_summary()
    print("\n" + "=" * 100)
    print("Final Results:")
    print(f"  Stack Ratio: {summary['stack_ratio']:.2f}")
    print(f"  Attack Success Rate: {summary['attack_success_rate']:.1%}")
    print(f"  Last 50 Rounds Committee Ratio: {summary['last_50_rounds_attacker_committee_ratio']:.1%}")
    print(f"  Committee Dominated: {'✓ YES' if summary['committee_dominated'] else '✗ NO'}")
    print("=" * 100)
    
    # 生成圖表
    sim.plot(save_path='results/quick_stake_simulation.png')
