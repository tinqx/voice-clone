"""
Tortoise-TTS Preset Benchmarking Tool

This script performs a comprehensive benchmark of Tortoise-TTS voice generation
across different quality presets. It measures generation time and GPU memory
usage
"""

import time
import torch
import os
import argparse
import json
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio


def load_voice_samples(voice_samples_dir):
    """
    Loads all WAV files from a directory for cloning
    """
    samples = []
    if not os.path.exists(voice_samples_dir):
        raise FileNotFoundError(f"Voice directory not found: {voice_samples_dir}")
    
    for file_name in sorted(os.listdir(voice_samples_dir)):
        if file_name.lower().endswith('.wav'):
            file_path = os.path.join(voice_samples_dir, file_name)
            try:
                audio = load_audio(file_path, 22050)
                samples.append(audio)
            except Exception as e:
                print(f"Failed to load {file_name}: {e}")
    
    if not samples:
        raise ValueError("No valid WAV files found")
    
    return samples


def benchmark_preset(tts_instance, preset_name, text, voice_samples, repetitions=3):
    """
    Executes benchmark for a single preset config.
    """
    start_time = time.time()
    
    # Initialize GPU memory tracking
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Execute multiple generations
    for iteration in range(repetitions):
        audio = tts_instance.tts_with_preset(
            text, 
            voice_samples=voice_samples, 
            preset=preset_name
        )
    
    total_time = time.time() - start_time
    average_time = total_time / repetitions
    
    # Capture peak GPU memory usage
    if torch.cuda.is_available():
        peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB
    else:
        peak_gpu_memory = 0.0
    
    return average_time, peak_gpu_memory


def main():
    parser = argparse.ArgumentParser(
        description='TTS Preset Benchmark',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--voice', type=str, required=True,
                       help='Dir containing WAV voice samples for cloning')
    
    parser.add_argument('--text', type=str, required=True,
                       help='Text content to synthesize during benchmarking')
    
    parser.add_argument('--repetitions', type=int, default=3,
                       help='Number of generations per preset for statistical reliability')
    
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output file for JSON results')
    
    args = parser.parse_args()

    # Initialize Tortoise-TTS engine
    try:
        tts_engine = TextToSpeech()
    except Exception as e:
        print(f"Failed to initialize TTS engine: {e}")
        return

    # Load voice samples
    try:
        voice_samples = load_voice_samples(args.voice)
    except Exception as e:
        print(f"Error loading voice samples: {e}")
        return

    # Define presets to benchmark
    quality_presets = ["ultra_fast", "fast", "standard", "high_quality"]
    results = {}

    # Execute benchmark for each preset
    for preset in quality_presets:
        try:
            avg_time, gpu_memory = benchmark_preset(
                tts_engine, preset, args.text, voice_samples, args.repetitions
            )
            
            results[preset] = {
                "avg_time_sec": round(avg_time, 2),
                "gpu_memory_gb": round(gpu_memory, 2),
                "repetitions": args.repetitions
            }
            
        except Exception as e:
            results[preset] = {"error": str(e)}

    # Save results to JSON file
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {args.output}")
    except Exception as e:
        print(f"Failed to save results: {e}")
        return

    # Display summary
    print("\nBenchmark:")
    for preset, data in results.items():
        if "error" not in data:
            print(f"  {preset}: {data['avg_time_sec']:.1f}s, {data['gpu_memory_gb']:.1f}GB")
        else:
            print(f"  {preset}: ERROR - {data['error']}")


if __name__ == "__main__":
    main()
