swift sft   \
    --model Qwen/Qwen2.5-Omni-7B   \
    --dataset ./train.jsonl   \
    --tuner_type lora   \
    --output_dir output   \
    --max_length 4096
