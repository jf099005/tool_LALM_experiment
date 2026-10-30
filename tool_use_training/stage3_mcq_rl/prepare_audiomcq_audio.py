"""Put Clotho and AudioCaps audio on disk under the names AudioMCQ expects.

- Clotho v2.1: HF mirror `gijs/clotho` (parquet, all of dev/val/eval with original
  file names) -> <clotho-out>/<file_name>.wav
- AudioCaps v2.0: HF `Zhaowc/AudioCaps` (one tar.gz split into parts, members named
  audiocaps_raw_audio/<youtube_id>_<start>.wav) -> <audiocaps-out>/<youtube_id>_<start>.wav.
  Only members referenced by AudioMCQ_AudioCaps.jsonl are extracted.

Safe to interrupt and rerun: existing files are skipped, each file is written to a temp
name and renamed. (The tarball is read as one stream, so a rerun re-reads it from the
start but writes nothing it already has.)

Usage:
    python prepare_audiomcq_audio.py \\
        --audiomcq-dir <root>/datasets/AudioMCQ/datasets \\
        --clotho-parquet-dir <root>/datasets/Clotho/parquet/data --clotho-out <root>/datasets/Clotho/audio \\
        --audiocaps-tar-dir <root>/datasets/AudioCaps/raw --audiocaps-out <root>/datasets/AudioCaps/audio
"""
from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
from pathlib import Path
from typing import BinaryIO, List, Set

from tqdm import tqdm


def _atomic_write(dest: Path, data: bytes) -> None:
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, dest)


def wanted_stems(jsonl: Path) -> Set[str]:
    with jsonl.open(encoding="utf-8") as f:
        return {Path(json.loads(line)["audio_path"]).stem for line in f}


def prepare_clotho(parquet_dir: Path, out_dir: Path) -> None:
    import pyarrow.parquet as pq
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(parquet_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No parquet files in {parquet_dir}")
    written = 0
    for path in tqdm(files, desc="clotho shards"):
        pf = pq.ParquetFile(str(path))
        for rg in range(pf.num_row_groups):
            names = pf.read_row_group(rg, columns=["file_name"]).column("file_name").to_pylist()
            todo = [i for i, n in enumerate(names) if not (out_dir / f"{Path(n).stem}.wav").exists()]
            if not todo:
                continue
            audios = pf.read_row_group(rg, columns=["audio"]).column("audio")
            for i in todo:
                dest = out_dir / f"{Path(names[i]).stem}.wav"
                data = audios[i].as_py()["bytes"]
                if data[:4] != b"RIFF":
                    y, sr = sf.read(io.BytesIO(data))
                    buf = io.BytesIO()
                    sf.write(buf, y, sr, format="WAV")
                    data = buf.getvalue()
                _atomic_write(dest, data)
                written += 1
    print(f"Clotho: {written} new wav(s), {len(list(out_dir.glob('*.wav')))} total in {out_dir}")


class _ConcatReader(io.RawIOBase):
    """Read several files back-to-back as one stream (the tar.gz.part.NN pieces)."""

    def __init__(self, paths: List[Path]):
        self._paths = list(paths)
        self._fh: BinaryIO | None = None

    def readable(self) -> bool:
        return True

    def readinto(self, buf) -> int:
        while True:
            if self._fh is None:
                if not self._paths:
                    return 0
                self._fh = open(self._paths.pop(0), "rb")
            n = self._fh.readinto(buf)
            if n:
                return n
            self._fh.close()
            self._fh = None


def prepare_audiocaps(tar_dir: Path, out_dir: Path, wanted: Set[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    have = {p.stem for p in out_dir.glob("*.wav")}
    todo = wanted - have
    print(f"AudioCaps: {len(have & wanted)}/{len(wanted)} already present, {len(todo)} to extract")
    if not todo:
        return
    parts = sorted(tar_dir.glob("audiocaps_raw_audio.tar.gz.part.*"))
    if not parts:
        raise SystemExit(f"No audiocaps_raw_audio.tar.gz.part.* in {tar_dir}")
    stream = io.BufferedReader(_ConcatReader(parts), buffer_size=16 << 20)
    written = 0
    with tarfile.open(fileobj=stream, mode="r|gz") as tar, tqdm(total=len(todo), desc="audiocaps") as bar:
        for member in tar:
            name = Path(member.name).name
            if not member.isfile() or name.startswith("._") or not name.endswith(".wav"):
                continue
            stem = name[:-4]
            if stem not in todo:
                continue
            _atomic_write(out_dir / name, tar.extractfile(member).read())
            todo.discard(stem)
            written += 1
            bar.update(1)
            if not todo:
                break
    print(f"AudioCaps: {written} new wav(s); {len(todo)} referenced clip(s) not in the tarball")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audiomcq-dir", type=Path, required=True)
    ap.add_argument("--clotho-parquet-dir", type=Path)
    ap.add_argument("--clotho-out", type=Path)
    ap.add_argument("--audiocaps-tar-dir", type=Path)
    ap.add_argument("--audiocaps-out", type=Path)
    args = ap.parse_args()

    if args.clotho_parquet_dir and args.clotho_out:
        prepare_clotho(args.clotho_parquet_dir, args.clotho_out)
    if args.audiocaps_tar_dir and args.audiocaps_out:
        prepare_audiocaps(args.audiocaps_tar_dir, args.audiocaps_out,
                          wanted_stems(args.audiomcq_dir / "AudioMCQ_AudioCaps.jsonl"))


if __name__ == "__main__":
    main()
