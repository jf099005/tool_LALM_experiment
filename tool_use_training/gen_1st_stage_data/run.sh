stage1_dir=/work/u1501463/stage1_RL/
# stage1_dir=./exp/
python build_dataset.py \
    --sources audioset \
    --num-samples 5000   \
    --min-tools 1 \
    --max-tools 1   \
    --seed 0   \
    --workers 24   \
    --output-dir ${stage1_dir}/audio   \
    --output-file ${stage1_dir}/tool_usage_qa_audioset.json \
    --swift-output-file ${stage1_dir}/train.jsonl
