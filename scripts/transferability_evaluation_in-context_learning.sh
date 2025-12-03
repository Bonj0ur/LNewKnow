cd src/in-context_learning/transfer_study/
CUDA_VISIBLE_DEVICES=0 python query_new_knowledge.py \
    --input_path ../../../datasets/new_knowledge_datasets/generated_new_knowledge.json \
    --save_dir ../../../results/in-context_learning/transfer_study/generated_new_knowledge/Aya-Expanse-8B \
    --mode generated_new_knowledge \
    --backend vllm \
    --vllm_model ../../../models/Aya-Expanse-8B
CUDA_VISIBLE_DEVICES=0 python query_new_knowledge.py \
    --input_path ../../../datasets/new_knowledge_datasets/generated_new_knowledge.json \
    --save_dir ../../../results/in-context_learning/transfer_study/generated_new_knowledge/Llama-3.1-8B-Instruct \
    --mode generated_new_knowledge \
    --backend vllm \
    --vllm_model ../../../models/Llama-3.1-8B-Instruct
CUDA_VISIBLE_DEVICES=0 python query_new_knowledge.py \
    --input_path ../../../datasets/new_knowledge_datasets/generated_new_knowledge.json \
    --save_dir ../../../results/in-context_learning/transfer_study/generated_new_knowledge/Qwen3-8B \
    --mode generated_new_knowledge \
    --backend vllm \
    --vllm_model ../../../models/Qwen3-8B
python query_new_knowledge.py \
    --input_path ../../../datasets/new_knowledge_datasets/generated_new_knowledge.json \
    --save_dir ../../../results/in-context_learning/transfer_study/generated_new_knowledge/GPT-4o-Mini-2024-07-18 \
    --mode generated_new_knowledge \
    --backend openai \
    --openai_model gpt-4o-mini-2024-07-18
python judge_new_knowledge.py \
    --root_dir ../../../results/in-context_learning/transfer_study/generated_new_knowledge/Aya-Expanse-8B
python judge_new_knowledge.py \
    --root_dir ../../../results/in-context_learning/transfer_study/generated_new_knowledge/Llama-3.1-8B-Instruct
python judge_new_knowledge.py \
    --root_dir ../../../results/in-context_learning/transfer_study/generated_new_knowledge/Qwen3-8B
python judge_new_knowledge.py \
    --root_dir ../../../results/in-context_learning/transfer_study/generated_new_knowledge/GPT-4o-Mini-2024-07-18
python parse_results.py \
    --result_dir ../../../results/in-context_learning/transfer_study/generated_new_knowledge/ \
    --model_name Aya-Expanse-8B
python parse_results.py \
    --result_dir ../../../results/in-context_learning/transfer_study/generated_new_knowledge/ \
    --model_name Llama-3.1-8B-Instruct
python parse_results.py \
    --result_dir ../../../results/in-context_learning/transfer_study/generated_new_knowledge/ \
    --model_name Qwen3-8B
python parse_results.py \
    --result_dir ../../../results/in-context_learning/transfer_study/generated_new_knowledge/ \
    --model_name GPT-4o-Mini-2024-07-18
python visualization_matrix.py \
    --result_dir ../../../results/in-context_learning/transfer_study/generated_new_knowledge/ \
    --model_name Aya-Expanse-8B
python visualization_matrix.py \
    --result_dir ../../../results/in-context_learning/transfer_study/generated_new_knowledge/ \
    --model_name Llama-3.1-8B-Instruct
python visualization_matrix.py \
    --result_dir ../../../results/in-context_learning/transfer_study/generated_new_knowledge/ \
    --model_name Qwen3-8B
python visualization_matrix.py \
    --result_dir ../../../results/in-context_learning/transfer_study/generated_new_knowledge/ \
    --model_name GPT-4o-Mini-2024-07-18
python visualization_inequality.py \
    --result_dir ../../../results/in-context_learning/transfer_study/generated_new_knowledge/ \
    --model_name Aya-Expanse-8B
python visualization_inequality.py \
    --result_dir ../../../results/in-context_learning/transfer_study/generated_new_knowledge/ \
    --model_name Llama-3.1-8B-Instruct
python visualization_inequality.py \
    --result_dir ../../../results/in-context_learning/transfer_study/generated_new_knowledge/ \
    --model_name Qwen3-8B
python visualization_inequality.py \
    --result_dir ../../../results/in-context_learning/transfer_study/generated_new_knowledge/ \
    --model_name GPT-4o-Mini-2024-07-18
# --------------------------------------------------------------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python query_new_knowledge.py \
    --input_path ../../../datasets/new_knowledge_datasets/real_new_knowledge.json \
    --save_dir ../../../results/in-context_learning/transfer_study/real_new_knowledge/Aya-Expanse-8B \
    --mode real_new_knowledge \
    --backend vllm \
    --vllm_model ../../../models/Aya-Expanse-8B
CUDA_VISIBLE_DEVICES=0 python query_new_knowledge.py \
    --input_path ../../../datasets/new_knowledge_datasets/real_new_knowledge.json \
    --save_dir ../../../results/in-context_learning/transfer_study/real_new_knowledge/Llama-3.1-8B-Instruct \
    --mode real_new_knowledge \
    --backend vllm \
    --vllm_model ../../../models/Llama-3.1-8B-Instruct
CUDA_VISIBLE_DEVICES=0 python query_new_knowledge.py \
    --input_path ../../../datasets/new_knowledge_datasets/real_new_knowledge.json \
    --save_dir ../../../results/in-context_learning/transfer_study/real_new_knowledge/Qwen3-8B \
    --mode real_new_knowledge \
    --backend vllm \
    --vllm_model ../../../models/Qwen3-8B
python query_new_knowledge.py \
    --input_path ../../../datasets/new_knowledge_datasets/real_new_knowledge.json \
    --save_dir ../../../results/in-context_learning/transfer_study/real_new_knowledge/GPT-4o-Mini-2024-07-18 \
    --mode real_new_knowledge \
    --backend openai \
    --openai_model gpt-4o-mini-2024-07-18
python judge_new_knowledge.py \
    --root_dir ../../../results/in-context_learning/transfer_study/real_new_knowledge/Aya-Expanse-8B
python judge_new_knowledge.py \
    --root_dir ../../../results/in-context_learning/transfer_study/real_new_knowledge/Llama-3.1-8B-Instruct
python judge_new_knowledge.py \
    --root_dir ../../../results/in-context_learning/transfer_study/real_new_knowledge/Qwen3-8B
python judge_new_knowledge.py \
    --root_dir ../../../results/in-context_learning/transfer_study/real_new_knowledge/GPT-4o-Mini-2024-07-18
python parse_results.py \
    --result_dir ../../../results/in-context_learning/transfer_study/real_new_knowledge/ \
    --model_name Aya-Expanse-8B
python parse_results.py \
    --result_dir ../../../results/in-context_learning/transfer_study/real_new_knowledge/ \
    --model_name Llama-3.1-8B-Instruct
python parse_results.py \
    --result_dir ../../../results/in-context_learning/transfer_study/real_new_knowledge/ \
    --model_name Qwen3-8B
python parse_results.py \
    --result_dir ../../../results/in-context_learning/transfer_study/real_new_knowledge/ \
    --model_name GPT-4o-Mini-2024-07-18
python visualization_matrix.py \
    --result_dir ../../../results/in-context_learning/transfer_study/real_new_knowledge/ \
    --model_name Aya-Expanse-8B
python visualization_matrix.py \
    --result_dir ../../../results/in-context_learning/transfer_study/real_new_knowledge/ \
    --model_name Llama-3.1-8B-Instruct
python visualization_matrix.py \
    --result_dir ../../../results/in-context_learning/transfer_study/real_new_knowledge/ \
    --model_name Qwen3-8B
python visualization_matrix.py \
    --result_dir ../../../results/in-context_learning/transfer_study/real_new_knowledge/ \
    --model_name GPT-4o-Mini-2024-07-18
python visualization_inequality.py \
    --result_dir ../../../results/in-context_learning/transfer_study/real_new_knowledge/ \
    --model_name Aya-Expanse-8B
python visualization_inequality.py \
    --result_dir ../../../results/in-context_learning/transfer_study/real_new_knowledge/ \
    --model_name Llama-3.1-8B-Instruct
python visualization_inequality.py \
    --result_dir ../../../results/in-context_learning/transfer_study/real_new_knowledge/ \
    --model_name Qwen3-8B
python visualization_inequality.py \
    --result_dir ../../../results/in-context_learning/transfer_study/real_new_knowledge/ \
    --model_name GPT-4o-Mini-2024-07-18