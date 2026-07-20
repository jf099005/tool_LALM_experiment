"""General-purpose tool-calling interface for the trained/zero-shot LALM.

Drives the same multi-turn "model emits a tool call -> execute it for real ->
feed the real result back as the next turn" loop that
`tool_use_training/gen_1st_stage_data/build_dataset.py` teaches during SFT and
`testing_tool_use_benchmark/run_eval.py` drives at eval time, but generalized
to an arbitrary natural-language instruction over one or more input audios
instead of only the fixed source/target reconstruction task the benchmark
measures, and against the real tools in `tools/` (not the random-parameter
synthetic registry used to generate training data).

Typical usage::

    from interface import ToolCallingAgent
    from interface.engine import SwiftEngine

    engine = SwiftEngine(model="Qwen/Qwen2.5-Omni-7B", adapter_dir="output/checkpoint-500")
    agent = ToolCallingAgent(engine)
    result = agent.run(
        instruction="Remove the background noise and normalize the loudness.",
        audio_paths=["/path/to/input.wav"],
        work_dir="/work/u1501463/interface_runs/sample_0",
    )
    print(result.final_audio_path, result.stop_reason)

`engine` (`SwiftEngine`/`VLLMEngine`) is imported lazily by callers, not
re-exported here, since it pulls in ms-swift/vllm -- only needed if you're
actually running a model rather than, say, unit-testing `executor.py` alone.
"""

from __future__ import annotations

from .agent import AgentResult, Step, ToolCallingAgent
from .executor import ToolExecutionError, UnknownToolError

__all__ = [
    "ToolCallingAgent",
    "AgentResult",
    "Step",
    "ToolExecutionError",
    "UnknownToolError",
]
