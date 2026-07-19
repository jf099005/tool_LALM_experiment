stage1_dir=/work/u1501463/tool_use_stage1/
# stage1_dir=./exp/
python build_dataset.py \
    --sources audioset \
    --num-samples 20000   \
    --min-tools 1 \
    --max-tools 1   \
    --seed 0   \
    --workers 24   \
    --output-dir ${stage1_dir}/audio   \
    --output-file ${stage1_dir}/tool_usage_qa_audioset.json \
    --swift-output-file ${stage1_dir}/train.jsonl
