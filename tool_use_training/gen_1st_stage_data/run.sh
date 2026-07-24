# stage1_dir=/work/u1501463/stage1_RL/
stage1_dir=./exp/

# Base model this data is being generated for -- used to auto-detect its own
# official default system message (chat_template.json/tokenizer_config.json)
# as the base of the training system prompt. Point this at whichever model
# dir stage1_training/ actually trains from; swap it when training a
# different model.
base_model_dir=/work/u1501463/model_qwen_25/

python build_dataset.py \
    --sources audioset \
    --num-samples 10   \
    --min-tools 1 \
    --max-tools 1   \
    --seed 0   \
    --workers 24   \
    --output-dir ${stage1_dir}/audio   \
    --output-file ${stage1_dir}/tool_usage_qa_audioset.json \
    --swift-output-file ${stage1_dir}/train.jsonl \
    --system-prompt-model-dir ${base_model_dir} \
    --system-prompt-model-type qwen2_5_omni
