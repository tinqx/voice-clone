"""
tts_evaluation.py
"""

import argparse
import os
import time
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torchaudio

from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voices

def setup_argument_parser():
    parser = argparse.ArgumentParser(
        description='Generate speech samples with Tortoise TTS'
    )
    
    parser.add_argument('--text', type=str, required=True,
                       help='Text to synthesize. For emphasis, use parentheses: Hello (world)!')
    
    # Voice parameters
    parser.add_argument('--voice', type=str, default='random',
                       help='Voice to use. Options: random, myvoices, etc.')
    
    # Quality parameters
    parser.add_argument('--preset', type=str, default='standard',
                       choices=['ultra_fast', 'fast', 'standard', 'high_quality'],
                       help='Quality preset: ultra_fast (lowest), fast, standard, high_quality (highest)')
    
    parser.add_argument('--candidates', type=int, default=3,
                       help='Number of output candidates to generate per voice')
    
    # Technical parameters
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible results')
    
    parser.add_argument('--half_precision', action='store_true',
                       help='Use float16 precision for faster inference')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='thesis_results',
                       help='Directory to store output files')
    
    parser.add_argument('--experiment_name', type=str, default='',
                       help='Name for this experiment run (used in filenames)')
    
    return parser

def create_experiment_metadata(args, output_dir):
    """Create metadata file documenting the experiment parameters."""
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'text': args.text,
        'voice': args.voice,
        'preset': args.preset,
        'candidates': args.candidates,
        'seed': args.seed,
        'half_precision': args.half_precision,
        'output_dir': output_dir
    }

    #save metadata as JSON file
    metadata_path = os.path.join(output_dir, 'experiment_metadata.json')
    with open(metadata_path, 'w') as f:
        import json
        json.dump(metadata, f, indent=2)
    
    return metadata_path

def main():
    """Main function for TTS generation."""
    # Parse arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Create output directory
    experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if args.experiment_name:
        experiment_id = f"{args.experiment_name}_{experiment_id}"
    
    output_dir = os.path.join(args.output_dir, experiment_id)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting TTS generation")
    print(f"Text: {args.text}")
    print(f"Voice: {args.voice}")
    print(f"Preset: {args.preset}")
    print(f"Candidates: {args.candidates}")
    print(f"Seed: {args.seed}")
    print(f"Output: {output_dir}")
    
    # Save experiment metadata
    metadata_path = create_experiment_metadata(args, output_dir)
    print(f"Metadata saved: {metadata_path}")
    
    # Load TTS model
    start_time = time.time()
    
    tts = TextToSpeech(
        half=args.half_precision,
        kv_cache=True 
    )
    
    init_time = time.time() - start_time
    print(f"Model loaded in {init_time:.2f} seconds")
    
    # Load voice samples
    if '&' in args.voice:
        voice_sel = args.voice.split('&')
    else:
        voice_sel = [args.voice]
    
    voice_samples, conditioning_latents = load_voices(voice_sel)
    
    # Generate speech
    gen_start_time = time.time()
    
    gen, dbg_state = tts.tts_with_preset(
        args.text,
        k=args.candidates,
        voice_samples=voice_samples,
        conditioning_latents=conditioning_latents,
        preset=args.preset,
        use_deterministic_seed=args.seed,
        return_deterministic_state=True
    )
    
    gen_time = time.time() - gen_start_time
    print(f"Audio generated in {gen_time:.2f} seconds")
    
    # Save audio files
    if isinstance(gen, list):
        for j, audio in enumerate(gen):
            filename = f"{args.voice}_candidate_{j+1}.wav"
            filepath = os.path.join(output_dir, filename)
            torchaudio.save(filepath, audio.squeeze(0).cpu(), 24000)
            print(f"Saved: {filename}")
    else:
        filename = f"{args.voice}.wav"
        filepath = os.path.join(output_dir, filename)
        torchaudio.save(filepath, gen.squeeze(0).cpu(), 24000)
        print(f"Saved: {filename}")
    
    # Save debug state for reproducibility
    debug_path = os.path.join(output_dir, 'debug_state.pth')
    torch.save(dbg_state, debug_path)
    print(f"Debug state saved: {debug_path}")
    
    total_time = time.time() - start_time
    print(f"Experiment completed in {total_time:.2f} seconds")
    print(f"Results available in: {output_dir}")

if __name__ == '__main__':
    main()
