import json
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np

def plot_convergence(results_file, output_file):
    with open(results_file, 'r') as f:
        data = json.load(f)
        
    dataset = data['dataset']
    attack_start = data['attack_start_round']
    results = data['results']
    
    rounds = [r['round'] for r in results]
    blockdfl_acc = [r['blockdfl_accuracy'] for r in results]
    ours_acc = [r['ours_accuracy'] for r in results]
    
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(rounds, blockdfl_acc, 'r--', label='BlockDFL', linewidth=2)
    plt.plot(rounds, ours_acc, 'b-', label='Ours', linewidth=2)
    
    plt.axvline(x=attack_start, color='gray', linestyle=':', alpha=0.8)
    plt.text(attack_start + 2, 0.5, 'Attack Starts', rotation=90, color='gray')
    
    plt.title(f'{dataset}: Convergence Under Latent Collusion Attack')
    plt.xlabel('Number of Communication Rounds')
    plt.ylabel('Test Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Saved convergence plot to {output_file}")
    plt.close()

def plot_comparison(results_file, output_file):
    with open(results_file, 'r') as f:
        data = json.load(f)
        
    results = data['results']
    attack_start = data['attack_start_round']
    total_rounds = data['total_rounds']
    
    # Get accuracy at attack start (approx) and end
    # Find round closest to attack start (before attack)
    pre_attack_round = attack_start
    # Find last round
    final_round = total_rounds
    
    pre_attack_res = next((r for r in results if r['round'] == pre_attack_round), results[-1])
    final_res = results[-1]
    
    labels = ['Before Attack', 'After Attack']
    blockdfl_vals = [pre_attack_res['blockdfl_accuracy'], final_res['blockdfl_accuracy']]
    ours_vals = [pre_attack_res['ours_accuracy'], final_res['ours_accuracy']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.figure(figsize=(8, 6), dpi=300)
    plt.bar(x - width/2, blockdfl_vals, width, label='BlockDFL', color='red', alpha=0.7)
    plt.bar(x + width/2, ours_vals, width, label='Ours', color='blue', alpha=0.7)
    
    plt.ylabel('Test Accuracy')
    plt.title('Accuracy Comparison: Before vs After Attack')
    plt.xticks(x, labels)
    plt.legend()
    plt.ylim(0, 1.05)
    
    # Add value labels
    for i, v in enumerate(blockdfl_vals):
        plt.text(i - width/2, v + 0.01, f'{v:.2%}', ha='center')
    for i, v in enumerate(ours_vals):
        plt.text(i + width/2, v + 0.01, f'{v:.2%}', ha='center')
        
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Saved comparison plot to {output_file}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate plots from results')
    parser.add_argument('--results', type=str, required=True, help='Path to results JSON file')
    args = parser.parse_args()
    
    if not os.path.exists(args.results):
        print(f"Error: File {args.results} not found.")
        return
        
    base_name = os.path.splitext(os.path.basename(args.results))[0]
    dir_name = os.path.dirname(args.results)
    
    convergence_file = os.path.join(dir_name, f"{base_name}_convergence.png")
    comparison_file = os.path.join(dir_name, f"{base_name}_comparison.png")
    
    plot_convergence(args.results, convergence_file)
    plot_comparison(args.results, comparison_file)

if __name__ == "__main__":
    main()
