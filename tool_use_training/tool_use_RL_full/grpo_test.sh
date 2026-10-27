model_dir=/work/u1501463/Qwen3-Omni-30B-A3B-Instruct/
adapter_path=/work/u1501463/stage1_training/v0-20260814-011059/checkpoint-100/

ENABLE_AUDIO_OUTPUT=False \
CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
swift rlhf \
    --rlhf_type grpo \
    --model ${model_dir} \
    --adapters ${adapter_path} \
    --template qwen2_5_omni \
    --target_modules all-linear \
    --lora_rank 8 \
    --lora_alpha 32 \
    --quant_method bnb \
    --quant_bits 4 \
    --bnb_4bit_quant_type nf4 \
    --bnb_4bit_compute_dtype bfloat16 \
    --bnb_4bit_use_double_quant true \
    --dataset  /work/u1501463/stage1_qwen3_omni_RL/train.jsonl \
    --external_plugins reward.py \
    --reward_funcs tool_format tool_accuracy tool_audio_closeness \
    --reward_weights 0.2 0.5 0.3 \
    --num_generations 2 \
    --generation_batch_size 2 \
    --local_rollout_forward_batch_size 1 \
    --max_length 4096 \
    --deepspeed ds_zero3_offload_param_only.json \
    --gradient_checkpointing true
