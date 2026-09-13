"""
eval_voiceclone.py - Objective evaluation for Voice Cloning
Computes MCD-DTW-SL, SECS and F0 metrics for voice similarity assessment
"""

import argparse
import numpy as np
import librosa
import torch
from pathlib import Path
import json
from pymcd.mcd import Calculate_MCD
from speechbrain.inference import EncoderClassifier

class Config:
    sr = 22050
    secs_sr = 16000
    secs_model = "speechbrain/spkrec-ecapa-voxceleb"

    f0_min = 50
    f0_max = 400
    frame_length = 1024
    hop_length = 256

    device = "cuda" if torch.cuda.is_available() else "cpu"


def load_audio(path, sr):
    y, _ = librosa.load(path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)

    return y

# MCD

def mcd_dtw_sl(ref_wav, synth_wav):

    mcd_toolbox = Calculate_MCD(
        MCD_mode="dtw_sl"
    )

    value = mcd_toolbox.calculate_mcd(
        str(ref_wav),
        str(synth_wav)
    )

    return float(value)

# SECS

def secs_similarity(ref_wav, synth_wav, cfg, classifier):

    y_ref = load_audio(ref_wav, cfg.secs_sr)
    y_syn = load_audio(synth_wav, cfg.secs_sr)
    
    y_ref_tensor = torch.tensor(y_ref, dtype=torch.float32).unsqueeze(0).to(cfg.device)
    y_syn_tensor = torch.tensor(y_syn, dtype=torch.float32).unsqueeze(0).to(cfg.device)


    with torch.inference_mode():
        emb_ref = classifier.encode_batch(y_ref_tensor)
        emb_syn = classifier.encode_batch(y_syn_tensor)

    v1 = emb_ref.squeeze().detach().cpu().numpy()
    v2 = emb_syn.squeeze().detach().cpu().numpy()

    similarity = np.dot(v1,v2)/(np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)

    return float(similarity)

# F0 METRICS

def f0_metrics(ref_wav, synth_wav, cfg):

    y_ref = load_audio(ref_wav, cfg.sr)
    y_syn = load_audio(synth_wav, cfg.sr)
    
    f0_ref, _, _ = librosa.pyin(y_ref, fmin=cfg.f0_min, fmax=cfg.f0_max, sr=cfg.sr, frame_length=cfg.frame_length, hop_length=cfg.hop_length)
    f0_syn, _, _ = librosa.pyin(y_syn, fmin=cfg.f0_min, fmax=cfg.f0_max, sr=cfg.sr, frame_length=cfg.frame_length, hop_length=cfg.hop_length)


    mfcc_ref = librosa.feature.mfcc(y=y_ref, sr=cfg.sr, n_mfcc=13, n_fft=cfg.frame_length, hop_length=cfg.hop_length)
    mfcc_syn = librosa.feature.mfcc(y=y_syn, sr=cfg.sr, n_mfcc=13, n_fft=cfg.frame_length, hop_length=cfg.hop_length)
    
    _, warp_path = librosa.sequence.dtw(X=mfcc_ref, Y=mfcc_syn, metric="euclidean")

    warp_path = warp_path[::-1]

    aligned_ref = []
    aligned_syn = []

    for ref_idx, syn_idx in warp_path:

        if (
            ref_idx >= len(f0_ref)
            or syn_idx >= len(f0_syn)
        ):
            continue

        ref_value = f0_ref[ref_idx]
        syn_value = f0_syn[syn_idx]

        if (
            not np.isnan(ref_value)
            and not np.isnan(syn_value)
        ):
            aligned_ref.append(ref_value)
            aligned_syn.append(syn_value)

    aligned_ref = np.asarray(aligned_ref)
    aligned_syn = np.asarray(aligned_syn)

    duration_diff = float(
        abs(len(y_syn) - len(y_ref))
        / cfg.sr
    )

    if len(aligned_ref) < 3:
        return {
            "f0_rmse_hz": float("nan"),
            "f0_corr": float("nan"),
            "duration_diff_sec": duration_diff
        }

    rmse = float(
        np.sqrt(
            np.mean(
                (aligned_ref - aligned_syn) ** 2
            )
        )
    )

    if (
        np.std(aligned_ref) < 1e-8
        or np.std(aligned_syn) < 1e-8
    ):
        corr = float("nan")

    else:
        corr = float(
            np.corrcoef(
                aligned_ref,
                aligned_syn
            )[0, 1]
        )

    return {
        "f0_rmse_hz": rmse,
        "f0_corr": corr,
        "duration_diff_sec": duration_diff
    }

def main():

    parser = argparse.ArgumentParser( description="Objective evaluation for Voice Cloning")

    parser.add_argument("--ref", required=True, help="Path to original/reference voice")

    parser.add_argument("--synth", required=True, nargs="+", help="One or more synthetic voice files")

    parser.add_argument("--output", default="results.json", help="Output JSON file")

    args = parser.parse_args()

    cfg = Config()

    classifier = EncoderClassifier.from_hparams(
        source=cfg.secs_model,
        run_opts={"device": cfg.device}
    )

    results = {
        "reference": str(args.ref),
        "candidates": {}
   }

    # Evaluate every synthetic candidate
    
    for index, synth_path in enumerate(
        args.synth,
        start=1
    ):

        print()
        print(
            f"Evaluating candidate {index}/{len(args.synth)}:"
        )
        print(synth_path)

        candidate_result = {}

        candidate_result["file"] = str(synth_path)

        candidate_result["mcd_dtw_sl"] = mcd_dtw_sl(args.ref, synth_path)

        candidate_result["secs_cosine"] = secs_similarity(args.ref, synth_path, cfg, classifier)

        candidate_result["prosody"] = f0_metrics(args.ref, synth_path,cfg)

        candidate_name = f"candidate_{index}"

        results["candidates"][candidate_name] = (
            candidate_result
        )

        # show final results
        print(
            f"MCD-DTW-SL: "
            f"{candidate_result['mcd_dtw_sl']:.3f}"
        )

        print(
            f"SECS: "
            f"{candidate_result['secs_cosine']:.3f}"
        )

        print(
            f"F0-RMSE: "
            f"{candidate_result['prosody']['f0_rmse_hz']:.3f} Hz"
        )

        print(
            f"F0-Correlation: "
            f"{candidate_result['prosody']['f0_corr']:.3f}"
        )

        print(
            f"Duration-Diff: "
            f"{candidate_result['prosody']['duration_diff_sec']:.3f} s"
        )

    # Calculate mean values across all candidates

    mcd_values = []
    secs_values = []
    f0_rmse_values = []
    f0_corr_values = []
    duration_values = []

    for candidate in results["candidates"].values():

        mcd_values.append(
            candidate["mcd_dtw_sl"]
        )

        secs_values.append(
            candidate["secs_cosine"]
        )

        f0_rmse = candidate["prosody"]["f0_rmse_hz"]
        f0_corr = candidate["prosody"]["f0_corr"]

        if not np.isnan(f0_rmse):
            f0_rmse_values.append(f0_rmse)

        if not np.isnan(f0_corr):
            f0_corr_values.append(f0_corr)

        duration_values.append(
            candidate["prosody"]["duration_diff_sec"]
        )

    results["summary"] = {
        "number_of_candidates": len(args.synth),
        "mean_mcd_dtw_sl": float(
            np.mean(mcd_values)
        ),

        "mean_secs_cosine": float(np.mean(secs_values)
        ),

        "mean_f0_rmse_hz": (
            float(np.mean(f0_rmse_values))
            if f0_rmse_values
            else float("nan")
        ),

        "mean_f0_corr": (
            float(np.mean(f0_corr_values))
            if f0_corr_values
            else float("nan")
        ),

        "mean_duration_diff_sec": float(
            np.mean(duration_values)
        )
    }

    # save results

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path.write_text(
        json.dumps(
            results, indent=2, allow_nan=True
        ),
        encoding="utf-8"
    )

    # Summary

    print()
    print("Average of all Candidates")

    print(
        f"Mean MCD-DTW-SL: "
        f"{results['summary']['mean_mcd_dtw_sl']:.3f}"
    )

    print(
        f"Mean SECS: "
        f"{results['summary']['mean_secs_cosine']:.3f}"
    )

    print(
        f"Mean F0-RMSE: "
        f"{results['summary']['mean_f0_rmse_hz']:.3f} Hz"
    )

    print(
        f"Mean F0-Correlation: "
        f"{results['summary']['mean_f0_corr']:.3f}"
    )

    print(
        f"Mean Duration-Diff: "
        f"{results['summary']['mean_duration_diff_sec']:.3f} s"
    )

    print()
    print(
        f"Results saved to: {output_path}"
    )


if __name__ == "__main__":
    main()
