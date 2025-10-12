import matplotlib.pyplot as plt
import numpy as np
from math import pi
import json
import argparse
import sys

def load_results_from_json(json_path):
    """Load results from JSON file"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {json_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in: {json_path}")
        sys.exit(1)

def create_radar_chart_from_json(json_path, save_path="radar.png"):
    """
    Create radar chart directly from JSON file
    """
    # Load results
    results = load_results_from_json(json_path)
    
    # Extract data
    mcd = results["mcd_dtw_sl"]
    secs = results["secs_cosine"]
    f0_rmse = results["prosody"]["f0_rmse_hz"]
    f0_corr = results["prosody"]["f0_corr"]
    duration_diff = results["prosody"]["duration_diff_sec"]
    
    # Define metrics (clean labels)
    categories = ['MCD', 'SECS', 'F0-RMSE', 'F0-CORR', 'DUR-DIFF']
    
    # Normalize values for radar chart
    mcd_norm = max(0, 1 - (mcd / 30))
    secs_norm = secs
    f0_rmse_norm = max(0, 1 - (f0_rmse / 50))
    f0_corr_norm = max(0, (f0_corr + 1) / 2)
    duration_norm = max(0, 1 - (duration_diff / 5))
    
    values = [mcd_norm, secs_norm, f0_rmse_norm, f0_corr_norm, duration_norm]
    
    # Calculate angles for radar chart
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    values += values[:1]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # Plot radar
    ax.plot(angles, values, 'o-', linewidth=2, color='#2E86AB', markersize=6)
    ax.fill(angles, values, alpha=0.3, color='#2E86AB')
    
    # Add categories
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    
    # Adjust Y-axis
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['', '', '', '', ''], fontsize=0)  # Empty labels
    ax.grid(True, alpha=0.3)
    
    # No title
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()  # Close plot to avoid display
    
    print(f"PNG created: {save_path}")
    return fig

def create_comparison_radar(json_paths, system_names, save_path="comparison_radar.png"):
    """
    Create comparison radar chart for multiple voice cloning systems
    """
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    categories = ['MCD', 'SECS', 'F0-RMSE', 'F0-CORR', 'DUR-DIFF']
    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#1D8A99']
    
    for i, json_path in enumerate(json_paths):
        try:
            results = load_results_from_json(json_path)
            
            # Normalize values
            mcd_norm = max(0, 1 - (results["mcd_dtw_sl"] / 30))
            secs_norm = results["secs_cosine"]
            f0_rmse_norm = max(0, 1 - (results["prosody"]["f0_rmse_hz"] / 50))
            f0_corr_norm = max(0, (results["prosody"]["f0_corr"] + 1) / 2)
            duration_norm = max(0, 1 - (results["prosody"]["duration_diff_sec"] / 5))
            
            values = [mcd_norm, secs_norm, f0_rmse_norm, f0_corr_norm, duration_norm]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, color=colors[i % len(colors)], markersize=6)
            ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
            
        except FileNotFoundError:
            print(f"⚠️ File not found: {json_path}")
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticklabels(['', '', '', '', ''])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"PNG created: {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Create radar chart from voice cloning results')
    parser.add_argument('--json', type=str, required=True, 
                       help='Path to JSON results file')
    parser.add_argument('--output', type=str, default='radar.png',
                       help='Output filename')
    parser.add_argument('--compare', nargs='+', 
                       help='Compare multiple systems')
    
    args = parser.parse_args()
    
    if args.compare:
        json_paths = []
        system_names = []
        for item in args.compare:
            if ':' in item:
                path, name = item.split(':', 1)
                json_paths.append(path)
                system_names.append(name)
            else:
                json_paths.append(item)
                name = item.replace('results_', '').replace('.json', '').title()
                system_names.append(name)
        
        output_name = args.output if args.output != 'radar.png' else 'comparison_radar.png'
        create_comparison_radar(json_paths, system_names, output_name)
    else:
        create_radar_chart_from_json(args.json, args.output)

if __name__ == "__main__":
    main()
