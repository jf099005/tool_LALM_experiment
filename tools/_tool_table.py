"""Single source of truth for tool_name -> (module, class name).

Consumed two different ways by two different callers, which is exactly why this
table holds no imports of its own (safe to read in any conda env):

- `tools/__init__.py` eagerly imports every listed class via
  `importlib.import_module` to build `TOOL_CLASSES`/`TOOL_NAME_TO_CLASS`. Safe
  because every tool module guards its own heavy ML deps (torch, DeepFilterNet,
  AudioSR, sam_audio) with a top-level try/except and only fails at call time,
  not import time.
- `tools/tool_execute.py` loads exactly one entry per subprocess invocation via
  `importlib.util.spec_from_file_location`, so a narrow single-purpose conda env
  (e.g. only DeepFilterNet installed) never has to import unrelated tools'
  modules at all.

Previously this mapping was hand-duplicated in three places (`tools/__init__.py`'s
`TOOL_CLASSES`, `tools/tool_execute.py`'s `_TOOL_MODULES`, and
`testing_tool_use_benchmark/tool_executor.py`'s `_CLASSED_TOOLS`) and had drifted:
`TOOL_CLASSES` was missing `DenoiseTool` entirely, `denoise` pointed at a
`tools/denoise.py` re-export shim instead of the real implementation, and
`tool_executor.py` tried (inside a try/except that silently swallowed the
failure) to import a `HumanVoiceAmplifyTool` that has never existed in
`human_voice_enhance.py` -- not carried over here since it isn't real.
"""

from __future__ import annotations

from typing import Dict, Tuple

TOOL_MODULES: Dict[str, Tuple[str, str]] = {
    "asr": ("tools.asr", "ASRTool"),
    "clipping": ("tools.clipping", "ClippingTool"),
    "denoise": ("tools.denoise", "DenoiseTool"),
    "amplitude_normalize": ("tools.normalize", "AmplitudeNormalizeTool"),
    "loudness_normalize": ("tools.normalize", "LoudnessNormalizeTool"),
    "remove_dc_offset": ("tools.normalize", "DCOffsetRemovalTool"),
    "spectral_normalize": ("tools.normalize", "SpectralNormalizeTool"),
    "trim_silence": ("tools.normalize", "TrimSilenceTool"),
    "pre_emphasis": ("tools.normalize", "PreEmphasisTool"),
    "source_separation": ("tools.source_separation", "SourceSeparationTool"),
    "extract_target": ("tools.extract_remove_target", "ExtractTargetTool"),
    "remove_target": ("tools.extract_remove_target", "RemoveTargetTool"),
    "human_voice_enhance": ("tools.human_voice_enhance", "HumanVoiceEnhanceTool"),
    "super_resolution": ("tools.super_resolution", "SuperResolutionTool"),
    "pitch_shift": ("tools.pitch_time", "PitchShiftTool"),
    "time_stretch": ("tools.pitch_time", "TimeStretchTool"),
}
