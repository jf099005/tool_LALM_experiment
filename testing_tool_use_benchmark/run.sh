model_path=/work/u1501463/Qwen3-Omni-30B-A3B-Instruct/
adapter_path=/work/u1501463/stage1_training/v0-20260814-011059/checkpoint-100/
~/miniconda3/envs/swift_qwen3/bin/python run_eval.py \
    --benchmark-file benchmark.json \
    --model $model_path \
    --adapter-dir $adapter_path \
    --model-type qwen3_omni_moe \
    --system-prompt-model-type qwen3_omni \
    --output-file results_qwen3.json \
    --dtype float16