"""Unpack the DCASE 2025 Task-5 AudioQA train audio from the HF parquet mirror
(`gijs/dcase2025-audio-qa`, ungated) into the official folder layout, and write a
copy of a dcase_subset.json-shaped QA list whose `audio_path`s point at it.

The official repo (PeacefulData/2025_DCASE_AudioQA_Official) is gated and pulls its
audio from several external sources; the mirror packs the same 8221 train items, with
audio converted to mono 16-bit 48 kHz WAV.

Safe to interrupt and rerun: WAVs that already exist are skipped, and each file is
written to a temp name first, then renamed.

Usage:
    python prepare_dcase_audio.py \\
        --parquet-dir <root>/datasets/dcase2025-audio-qa-parquet/data \\
        --audio-root  <root>/datasets/2025_DCASE_AudioQA_Official \\
        --qa-json     dcase_7481.json \\
        --output-json <root>/datasets/dcase_7481_local.json
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
from tqdm import tqdm


def write_audio(audio: dict, dest: Path) -> None:
    tmp = dest.with_name(dest.name + ".part")
    data = audio.get("bytes")
    if data is None:
        raise ValueError(f"No audio bytes for {dest.name}")
    if data[:4] == b"RIFF":
        tmp.write_bytes(data)
    else:
        # Not a WAV container: decode and re-encode so downstream tools get a WAV.
        y, sr = sf.read(io.BytesIO(data))
        sf.write(str(tmp), y, sr, format="WAV")
    os.replace(tmp, dest)


def unpack(parquet_dir: Path, audio_root: Path, split: str) -> int:
    out_dir = audio_root / "local_audio_path" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(parquet_dir.glob(f"{split}-*.parquet"))
    if not files:
        raise SystemExit(f"No {split}-*.parquet files in {parquet_dir}")
    written = 0
    for path in tqdm(files, desc="parquet shards"):
        pf = pq.ParquetFile(str(path))
        for rg in range(pf.num_row_groups):
            names = pf.read_row_group(rg, columns=["audio_url"]).column("audio_url").to_pylist()
            todo = [i for i, n in enumerate(names) if not (out_dir / Path(n).name).exists()]
            if not todo:
                continue
            audios = pf.read_row_group(rg, columns=["audio"]).column("audio")
            for i in todo:
                dest = out_dir / Path(names[i]).name
                if dest.exists():  # duplicate audio_url within the shard
                    continue
                write_audio(audios[i].as_py(), dest)
                written += 1
    return written


def localize_qa(qa_json: Path, audio_root: Path, output_json: Path) -> None:
    items = json.loads(qa_json.read_text(encoding="utf-8"))
    kept, missing = [], []
    for item in items:
        rel = item.get("audio_url") or item.get("audio_path")
        # e.g. "../../local_audio_path/train/audio_01071.wav" -> train/audio_01071.wav
        parts = Path(rel).parts
        split = parts[-2]
        local = audio_root / "local_audio_path" / split / parts[-1]
        if local.exists():
            item = dict(item, audio_path=str(local))
            kept.append(item)
        else:
            missing.append(item.get("id"))
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(kept, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(kept)} items to {output_json} ({len(missing)} skipped: audio not found)")
    if missing:
        print("  first missing ids:", missing[:10])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--qa-json", type=Path, default=Path(__file__).resolve().parent / "dcase_7481.json")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    n = unpack(args.parquet_dir, args.audio_root, args.split)
    print(f"Unpacked {n} new audio file(s) into {args.audio_root / 'local_audio_path' / args.split}")
    localize_qa(args.qa_json, args.audio_root, args.output_json)


if __name__ == "__main__":
    main()
