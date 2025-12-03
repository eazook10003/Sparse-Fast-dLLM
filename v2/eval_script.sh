# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
model_path=Efficient-Large-Model/Fast_dLLM_v2_7B

# task=mmlu
# accelerate launch eval.py --tasks ${task} --batch_size 1 --num_fewshot 5 \
# --confirm_run_unsafe_code --model fast_dllm_v2 --fewshot_as_multiturn --apply_chat_template \
# --model_args model_path=${model_path}

# task=gpqa_main_n_shot
# accelerate launch eval.py --tasks ${task} --batch_size 1 \
# --confirm_run_unsafe_code --model fast_dllm_v2 --fewshot_as_multiturn --apply_chat_template \
# --model_args model_path=${model_path}

# Sparse cache parameters
use_sparse=True           # True = sparse cache, False = baseline (dense cache only)
keep_ratio=0.5            # Ratio of KV cache to keep (0.0-1.0)
pool_kernel_size=3        # Kernel size for max pooling in importance scoring
delay_step=1              # Steps before applying sparse cache

task=gsm8k
accelerate launch eval.py --tasks ${task} --batch_size 32 --num_fewshot 0 --limit 100 \
--confirm_run_unsafe_code --model fast_dllm_v2 --fewshot_as_multiturn --apply_chat_template \
--model_args model_path=${model_path},threshold=0.9,show_speed=True,use_sparse=True,use_block_cache=False,keep_ratio=1.0,pool_kernel_size=3,delay_step=1


# task=minerva_math
# accelerate launch eval.py --tasks ${task} --batch_size 32 --num_fewshot 0 \
# --confirm_run_unsafe_code --model fast_dllm_v2 --fewshot_as_multiturn --apply_chat_template \
# --model_args model_path=${model_path},threshold=1,show_speed=True

# task=ifeval
# accelerate launch eval.py --tasks ${task} --batch_size 32 \
# --confirm_run_unsafe_code --model fast_dllm_v2 --fewshot_as_multiturn --apply_chat_template \
# --model_args model_path=${model_path},threshold=1,show_speed=True
