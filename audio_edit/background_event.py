"""Insert a labeled, real-world background sound event (dog bark, wind, siren, ...)
into a clip at a given timestamp.

Design follows the standard approach used to synthesize labeled sound-event
datasets in the DCASE / sound-event-detection literature:

- Foreground events are drawn from a large labeled corpus (here: a local
  AudioSet subset, `audioset_metadata.json` + the matching WAV files) and
  overlaid onto a background/host recording at an explicit onset timestamp,
  exactly as Scaper (Salamon, MacConnell, Cartwright & Bello, "Scaper: A
  Library for Soundscape Synthesis and Augmentation", WASPAA 2017) generates
  its synthetic soundscapes, and as the DESED corpus (Turpault et al., DCASE
  2019 Task 4) was built for weakly-/strongly-labeled sound event detection.
- Event loudness relative to the background is controlled by an explicit
  target SNR in dB rather than a raw gain, following the same
  Scaper/DESED convention (and the MUSAN/Kaldi `augment_data_dir.py`
  noise-augmentation recipe, Snyder et al. 2015, for the wider "mix a labeled
  clip in at a target SNR" idea).
- A short raised-cosine fade (Scaper's default `ramp_duration` = 10 ms) is
  applied to the event's onset/offset to avoid an audible click/discontinuity
  where it is spliced in.
- The result carries the same "strong label" bookkeeping DESED/AudioSet
  annotations use: source label, onset, and offset (both in seconds and
  HH:MM:SS.mmm), so the edit is reproducible and machine-checkable.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from audio_edit._audio_utils import (  # noqa: E402
    AudioEditError,
    default_output_path,
    fade_in_out,
    fit_to_length,
    format_timestamp,
    load_audio,
    parse_timestamp,
    save_audio,
    scale_to_snr,
)
from tools.abstract_tool import Tool, ToolValidationError  # noqa: E402

DEFAULT_METADATA_PATH = _REPO_ROOT / "audioset_metadata.json"
DEFAULT_AUDIO_DIR = Path(os.environ.get("AUDIOSET_AUDIO_DIR", "/work/u1501463/audioset_20k/20k/train"))
DEFAULT_LABEL = "Dog"
# DESED/Scaper synthetic soundscapes typically place foreground events a few
# dB above the background so they are clearly audible without being jarring;
# tools/synthetic_registry.py (this repo's dataset synthesizer) samples the same range.
DEFAULT_SNR_DB = 5.0
DEFAULT_FADE_MS = 10.0


@functools.lru_cache(maxsize=8)
def _load_metadata(metadata_path: str) -> List[Dict[str, Any]]:
    path = Path(metadata_path)
    if not path.exists():
        raise AudioEditError(f"AudioSet metadata file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_label_matches(label: str, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Case-insensitive lookup: exact label match first, else whole-word substring.

    Word-boundary matching avoids false positives like "Rain" matching inside
    "Train" that plain substring search would produce.
    """
    exact = [e for e in entries if any(l.lower() == label.lower() for l in e.get("label", []))]
    if exact:
        return exact

    pattern = re.compile(rf"\b{re.escape(label)}\b", re.IGNORECASE)
    return [e for e in entries if any(pattern.search(l) for l in e.get("label", []))]


def available_labels(metadata_path: Path = DEFAULT_METADATA_PATH) -> List[str]:
    entries = _load_metadata(str(metadata_path))
    labels = {l for e in entries for l in e.get("label", [])}
    return sorted(labels)


def insert_background_event(
    audio_path: str,
    output_path: Optional[str] = None,
    *,
    label: str = DEFAULT_LABEL,
    timestamp: Optional[str] = None,
    duration: Optional[float] = None,
    snr_db: float = DEFAULT_SNR_DB,
    fade_ms: float = DEFAULT_FADE_MS,
    audio_dir: Optional[str] = None,
    metadata_path: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Overlay a labeled background sound event onto `audio_path` at `timestamp`.

    `label` is matched case-insensitively against the AudioSet ontology labels
    in `metadata_path` (e.g. "Dog", "Wind", "Siren"). If `timestamp` is None a
    valid onset is chosen at random. If `duration` is None the event's own
    (looped/trimmed-to-fit) length is used.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise AudioEditError(f"Audio file not found: {audio_path}")

    metadata_path = Path(metadata_path) if metadata_path else DEFAULT_METADATA_PATH
    audio_dir = Path(audio_dir) if audio_dir else DEFAULT_AUDIO_DIR

    entries = _load_metadata(str(metadata_path))
    matches = find_label_matches(label, entries)
    if not matches:
        raise AudioEditError(
            f"No AudioSet entries found for label '{label}'. "
            f"Try one of: {', '.join(available_labels(metadata_path)[:20])}, ..."
        )

    rng = random.Random(seed)
    entry = rng.choice(matches)
    event_path = audio_dir / entry["file name"]
    if not event_path.exists():
        raise AudioEditError(
            f"Matched AudioSet clip '{event_path}' does not exist. "
            f"Pass audio_dir= to point at the local copy of the AudioSet WAVs."
        )

    host, sr = load_audio(audio_path, sr=None, mono=True)
    event, _ = load_audio(event_path, sr=sr, mono=True)

    host_duration = host.shape[-1] / sr
    if timestamp is not None:
        onset_sec = parse_timestamp(timestamp)
    else:
        max_onset = max(0.0, host_duration - (duration or event.shape[-1] / sr))
        onset_sec = rng.uniform(0.0, max_onset)

    if onset_sec < 0 or onset_sec >= host_duration:
        raise AudioEditError(
            f"timestamp {onset_sec:.3f}s is out of bounds for a {host_duration:.3f}s clip."
        )

    event_len = int(sr * duration) if duration else event.shape[-1]
    onset_sample = int(onset_sec * sr)
    max_len = host.shape[-1] - onset_sample
    event_len = min(event_len, max_len)
    if event_len <= 0:
        raise AudioEditError("timestamp leaves no room to insert the event before the clip ends.")

    event = fit_to_length(event, event_len)
    event = fade_in_out(event, sr, fade_ms)

    local_host = host[onset_sample:onset_sample + event_len]
    event = scale_to_snr(local_host, event, snr_db)

    mixed = host.copy()
    mixed[onset_sample:onset_sample + event_len] += event

    out_path = Path(output_path) if output_path else default_output_path(audio_path, "insert_event")
    save_audio(out_path, mixed, sr)

    offset_sec = onset_sec + event_len / sr
    matched_label = next((l for l in entry["label"] if l.lower() == label.lower()), entry["label"][0])

    return {
        "audio_path": str(audio_path),
        "event_label": matched_label,
        "requested_label": label,
        "source_event_id": entry.get("id"),
        "source_event_path": str(event_path),
        "onset": round(onset_sec, 3),
        "offset": round(offset_sec, 3),
        "onset_timestamp": format_timestamp(onset_sec),
        "offset_timestamp": format_timestamp(offset_sec),
        "snr_db": snr_db,
        "fade_ms": fade_ms,
        "status": "success",
        "output_path": str(out_path),
        "message": "Background event insertion completed successfully.",
    }


class InsertBackgroundEventTool(Tool):
    @classmethod
    def name(cls) -> str:
        return "insert_background_event"

    @classmethod
    def description(cls) -> str:
        return (
            "Overlay a labeled, real-world background sound event (e.g. 'Dog', 'Wind', "
            "'Siren') from a local AudioSet subset onto a WAV/MP3 clip at a timestamp, "
            "at a target SNR, following the Scaper/DESED soundscape-synthesis "
            "convention. Requires audio_path and label; timestamp, duration, snr_db, "
            "and fade_ms all have defaults."
        )

    @classmethod
    def parameter_schema(cls) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "audio_path": {"type": "string"},
                "label": {
                    "type": "string",
                    "description": "AudioSet ontology label to search for, e.g. 'Dog', 'Wind', 'Siren'.",
                },
                "timestamp": {
                    "type": "string",
                    "format": "HH:MM:SS.mmm",
                    "description": "Onset of the inserted event. Random if omitted.",
                },
                "duration": {
                    "type": "number",
                    "description": "Event duration in seconds (looped/trimmed to fit). Uses the source clip's own length if omitted.",
                },
                "snr_db": {
                    "type": "number",
                    "description": f"Event loudness relative to the local background, in dB (default {DEFAULT_SNR_DB}).",
                },
                "fade_ms": {
                    "type": "number",
                    "description": f"Onset/offset fade duration, in ms (default {DEFAULT_FADE_MS}).",
                },
                "audio_dir": {"type": "string", "description": "Override the local AudioSet WAV directory."},
                "metadata_path": {"type": "string", "description": "Override the AudioSet metadata JSON path."},
                "output_path": {"type": "string"},
            },
            "required": ["audio_path", "label"],
        }

    @classmethod
    def execute(cls, parameters: Dict[str, Any]) -> Dict[str, Any]:
        cls.validate_parameters(parameters)
        try:
            return insert_background_event(
                parameters["audio_path"],
                parameters.get("output_path"),
                label=parameters["label"],
                timestamp=parameters.get("timestamp"),
                duration=parameters.get("duration"),
                snr_db=float(parameters.get("snr_db", DEFAULT_SNR_DB)),
                fade_ms=float(parameters.get("fade_ms", DEFAULT_FADE_MS)),
                audio_dir=parameters.get("audio_dir"),
                metadata_path=parameters.get("metadata_path"),
            )
        except AudioEditError as exc:
            raise ToolValidationError(str(exc)) from exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Insert a labeled AudioSet background event into a clip.")
    parser.add_argument("input_file", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument("--timestamp", default=None, help="HH:MM:SS.mmm onset; random if omitted")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--snr_db", type=float, default=DEFAULT_SNR_DB)
    parser.add_argument("--fade_ms", type=float, default=DEFAULT_FADE_MS)
    parser.add_argument("--audio_dir", default=None)
    parser.add_argument("--metadata_path", default=None)
    args = parser.parse_args()

    result = insert_background_event(
        args.input_file,
        args.output_file,
        label=args.label,
        timestamp=args.timestamp,
        duration=args.duration,
        snr_db=args.snr_db,
        fade_ms=args.fade_ms,
        audio_dir=args.audio_dir,
        metadata_path=args.metadata_path,
    )
    print(f"處理完成！已插入 '{result['event_label']}' 於 {result['onset_timestamp']}，已儲存至: {result['output_path']}")
