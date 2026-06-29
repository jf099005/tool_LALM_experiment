python build_dataset.py \
    --sources audioset \
    --num-samples 10   \
    --min-tools 1 \
    --max-tools 2   \
    --seed 0   \
    --output-dir /work/u1501463/gen_tool_usage_QA/audio   \
    --output-file ~/tool_use_LALM/generate/gen_tool_usage_QA/tool_usage_qa_audioset.json \
    --swift-output-file ~/tool_use_LALM/train.jsonl
