"""
eval_voiceclone.py - Objective evaluation for Voice Cloning
Computes MCD-DTW-SL, SECS and F0 metrics for voice similarity assessment
"""

import argparse
import math
import numpy as np
import librosa
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import torch
from speechbrain.inference import EncoderClassifier
import soundfile as sf
from pathlib import Path
import json

class Config:
    sr = 22050
    n_mfcc = 13
    use_c0 = False
    dtw_radius = 10
    sl_weight = 1.0
    secs_model = "speechbrain/spkrec-ecapa-voxceleb"
    device = "cuda" if torch.cuda.is_available() else "cpu"

def load_audio(path, sr):
    y, _ = librosa.load(path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    return y

def mfcc_features(y, sr, n_mfcc, include_c0):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # MFCC normalization for numerical stability
    mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-8)
    
    if not include_c0:
        mfcc = mfcc[1:, :]
    return mfcc.T

def mcd_dtw_sl(ref_wav, synth_wav, cfg):
    y_ref = load_audio(ref_wav, cfg.sr)
    y_syn = load_audio(synth_wav, cfg.sr)

    mfcc_ref = mfcc_features(y_ref, cfg.sr, cfg.n_mfcc, cfg.use_c0)
    mfcc_syn = mfcc_features(y_syn, cfg.sr, cfg.n_mfcc, cfg.use_c0)

    distance, path = fastdtw(mfcc_ref, mfcc_syn, radius=cfg.dtw_radius, dist=euclidean)
    avg_dist = distance / max(1, len(path))
    mcd = (10.0 / math.log(10.0)) * math.sqrt(2) * avg_dist

    T_ref, T_syn = mfcc_ref.shape[0], mfcc_syn.shape[0]
    length_penalty = cfg.sl_weight * abs(T_syn - T_ref) / max(1, T_ref)
    
    return float(mcd + length_penalty)

def secs_similarity(ref_wav, synth_wav, cfg):
    classifier = EncoderClassifier.from_hparams(source=cfg.secs_model)
    y_ref = load_audio(ref_wav, cfg.sr)
    y_syn = load_audio(synth_wav, cfg.sr)
    
    y_ref_tensor = torch.tensor(np.array([y_ref]), dtype=torch.float32).to(cfg.device)
    y_syn_tensor = torch.tensor(np.array([y_syn]), dtype=torch.float32).to(cfg.device)

    emb_ref = classifier.encode_batch(y_ref_tensor)
    emb_syn = classifier.encode_batch(y_syn_tensor)
    v1 = emb_ref.squeeze().detach().cpu().numpy()
    v2 = emb_syn.squeeze().detach().cpu().numpy()

    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))

def f0_metrics_simple(ref_wav, synth_wav, cfg):
    y_ref = load_audio(ref_wav, cfg.sr)
    y_syn = load_audio(synth_wav, cfg.sr)
    
    f0_ref, voiced_flag_ref, _ = librosa.pyin(y_ref, fmin=50, fmax=400, sr=cfg.sr, frame_length=1024, hop_length=256)
    f0_syn, voiced_flag_syn, _ = librosa.pyin(y_syn, fmin=50, fmax=400, sr=cfg.sr, frame_length=1024, hop_length=256)
    
    min_length = min(len(f0_ref), len(f0_syn))
    f0_ref = f0_ref[:min_length]
    f0_syn = f0_syn[:min_length]
    
    mask = (~np.isnan(f0_ref)) & (~np.isnan(f0_syn))
    if np.sum(mask) < 3:
        return {"f0_rmse_hz": float("nan"), "f0_corr": float("nan"), "duration_diff_sec": 0.0}
    
    f0_ref_voiced = f0_ref[mask]
    f0_syn_voiced = f0_syn[mask]
    
    rmse = float(np.sqrt(np.mean((f0_ref_voiced - f0_syn_voiced) ** 2)))
    corr = float(np.corrcoef(f0_ref_voiced, f0_syn_voiced)[0, 1])
    
    dur_diff = float(abs(len(y_syn) - len(y_ref)) / cfg.sr)
    
    return {"f0_rmse_hz": rmse, "f0_corr": corr, "duration_diff_sec": dur_diff}

def main():
    parser = argparse.ArgumentParser(description="Objective evaluation for Voice Cloning")
    parser.add_argument("--ref", required=True, help="Path to reference/original voice")
    parser.add_argument("--synth", required=True, help="Path to synthetic voice")
    parser.add_argument("--output", default="results.json", help="Output JSON file path")
    args = parser.parse_args()

    cfg = Config()
    results = {}

    # Calculate metrics
    results["mcd_dtw_sl"] = mcd_dtw_sl(args.ref, args.synth, cfg)
    results["secs_cosine"] = secs_similarity(args.ref, args.synth, cfg)
    results["prosody"] = f0_metrics_simple(args.ref, args.synth, cfg)

    # Save results
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    
    # show final results
    print("Evaluation completed:")
    print(f"MCD-DTW-SL: {results['mcd_dtw_sl']:.3f}")
    print(f"SECS: {results['secs_cosine']:.3f}")
    print(f"F0-RMSE: {results['prosody']['f0_rmse_hz']:.3f} Hz")
    print(f"F0-Correlation: {results['prosody']['f0_corr']:.3f}")
    print(f"Duration-Diff: {results['prosody']['duration_diff_sec']:.3f} s")
    print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()
