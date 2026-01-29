import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_quality_analysis(results_file):
    with open(results_file, 'r') as f:
        data = json.load(f)
        
    results = data['results']
    rounds = [r['round'] for r in results]
    
    # Extract selected qualities
    bdfl_quality = [r['blockdfl_selection']['quality'] for r in results]
    caca_quality = [r['caca_selection']['quality'] for r in results]
    
    # Extract min/max available qualities to see the range
    min_qualities = [min(r['blockdfl_selection']['all_qualities']) for r in results]
    max_qualities = [max(r['blockdfl_selection']['all_qualities']) for r in results]
    
    plt.figure(figsize=(12, 6), dpi=300)
    
    # Plot Range
    plt.fill_between(rounds, min_qualities, max_qualities, color='gray', alpha=0.2, label='Available Quality Range')
    
    # Plot Selected
    plt.plot(rounds, bdfl_quality, 'r-', label='BlockDFL Selected (Worst)', linewidth=1.5, alpha=0.8)
    plt.plot(rounds, caca_quality, 'b-', label='CACA Selected (Best)', linewidth=1.5, alpha=0.8)
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    plt.title('Selected Update Quality (Test Accuracy Change)')
    plt.xlabel('Round')
    plt.ylabel('Quality (Accuracy Delta)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = results_file.replace('.json', '_quality_analysis.png')
    plt.savefig(output_file)
    print(f"Saved quality analysis plot to {output_file}")

if __name__ == "__main__":
    plot_quality_analysis('results/mnist_noniid_results.json')
