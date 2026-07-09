python build_dataset.py \
    --sources audioset \
    --num-samples 10000   \
    --min-tools 1 \
    --max-tools 2   \
    --seed 0   \
    --output-dir /work/u1501463/gen_tool_usage_QA/audio   \
    --output-file ../tool_usage_qa_audioset.json \
    --swift-output-file ../train.jsonl
