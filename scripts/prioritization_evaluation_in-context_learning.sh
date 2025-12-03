cd src/in-context_learning/conflict_study
CUDA_VISIBLE_DEVICES=0 python query_new_knowledge_local.py \
    --data_path ../../../datasets/new_knowledge_datasets/generated_new_knowledge.json \
    --conflict_path ../../../datasets/new_knowledge_datasets/generated_conflict_study/selected_pairs.json \
    --save_root ../../../results/in-context_learning/conflict_study/generated_new_knowledge/Aya-Expanse-8B \
    --vllm_model ../../../models/Aya-Expanse-8B \
    --mode generated_new_knowledge
CUDA_VISIBLE_DEVICES=0 python query_new_knowledge_local.py \
    --data_path ../../../datasets/new_knowledge_datasets/generated_new_knowledge.json \
    --conflict_path ../../../datasets/new_knowledge_datasets/generated_conflict_study/selected_pairs.json \
    --save_root ../../../results/in-context_learning/conflict_study/generated_new_knowledge/Llama-3.1-8B-Instruct \
    --vllm_model ../../../models/Llama-3.1-8B-Instruct \
    --mode generated_new_knowledge
CUDA_VISIBLE_DEVICES=0 python query_new_knowledge_local.py \
    --data_path ../../../datasets/new_knowledge_datasets/generated_new_knowledge.json \
    --conflict_path ../../../datasets/new_knowledge_datasets/generated_conflict_study/selected_pairs.json \
    --save_root ../../../results/in-context_learning/conflict_study/generated_new_knowledge/Qwen3-8B \
    --vllm_model ../../../models/Qwen3-8B \
    --mode generated_new_knowledge
python query_new_knowledge_openai.py \
    --data_path ../../../datasets/new_knowledge_datasets/generated_new_knowledge.json \
    --conflict_path ../../../datasets/new_knowledge_datasets/generated_conflict_study/selected_pairs.json \
    --save_root ../../../results/in-context_learning/conflict_study/generated_new_knowledge/GPT-4o-Mini-2024-07-18 \
    --mode generated_new_knowledge
python judge_conflict.py \
    --root_dir ../../../results/in-context_learning/conflict_study/generated_new_knowledge/Aya-Expanse-8B
python judge_conflict.py \
    --root_dir ../../../results/in-context_learning/conflict_study/generated_new_knowledge/Llama-3.1-8B-Instruct
python judge_conflict.py \
    --root_dir ../../../results/in-context_learning/conflict_study/generated_new_knowledge/Qwen3-8B
python judge_conflict.py \
    --root_dir ../../../results/in-context_learning/conflict_study/generated_new_knowledge/GPT-4o-Mini-2024-07-18
python parse_results.py \
    --model_folder ../../../results/in-context_learning/conflict_study/generated_new_knowledge/Aya-Expanse-8B
python parse_results.py \
    --model_folder ../../../results/in-context_learning/conflict_study/generated_new_knowledge/Llama-3.1-8B-Instruct
python parse_results.py \
    --model_folder ../../../results/in-context_learning/conflict_study/generated_new_knowledge/Qwen3-8B
python parse_results.py \
    --model_folder ../../../results/in-context_learning/conflict_study/generated_new_knowledge/GPT-4o-Mini-2024-07-18
python visualization_barplot.py \
    --root_dir ../../../results/in-context_learning/conflict_study/generated_new_knowledge/Aya-Expanse-8B
python visualization_barplot.py \
    --root_dir ../../../results/in-context_learning/conflict_study/generated_new_knowledge/Llama-3.1-8B-Instruct
python visualization_barplot.py \
    --root_dir ../../../results/in-context_learning/conflict_study/generated_new_knowledge/Qwen3-8B
python visualization_barplot.py \
    --root_dir ../../../results/in-context_learning/conflict_study/generated_new_knowledge/GPT-4o-Mini-2024-07-18
python aggregate_results.py \
    --base_dir ../../../results/in-context_learning/conflict_study/generated_new_knowledge \
    --model_name Aya-Expanse-8B
python aggregate_results.py \
    --base_dir ../../../results/in-context_learning/conflict_study/generated_new_knowledge \
    --model_name Llama-3.1-8B-Instruct
python aggregate_results.py \
    --base_dir ../../../results/in-context_learning/conflict_study/generated_new_knowledge \
    --model_name Qwen3-8B
python aggregate_results.py \
    --base_dir ../../../results/in-context_learning/conflict_study/generated_new_knowledge \
    --model_name GPT-4o-Mini-2024-07-18
python visualization_boxplot.py \
    --root_dir ../../../results/in-context_learning/conflict_study/generated_new_knowledge \
    --save_path ../../../results/in-context_learning/conflict_study/generated_new_knowledge/box.png
# --------------------------------------------------------------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python query_new_knowledge_local.py \
    --data_path ../../../datasets/new_knowledge_datasets/real_new_knowledge.json \
    --conflict_path ../../../datasets/new_knowledge_datasets/real_conflict_study/selected_pairs.json \
    --save_root ../../../results/in-context_learning/conflict_study/real_new_knowledge/Aya-Expanse-8B \
    --vllm_model ../../../models/Aya-Expanse-8B \
    --mode real_new_knowledge
CUDA_VISIBLE_DEVICES=0 python query_new_knowledge_local.py \
    --data_path ../../../datasets/new_knowledge_datasets/real_new_knowledge.json \
    --conflict_path ../../../datasets/new_knowledge_datasets/real_conflict_study/selected_pairs.json \
    --save_root ../../../results/in-context_learning/conflict_study/real_new_knowledge/Llama-3.1-8B-Instruct \
    --vllm_model ../../../models/Llama-3.1-8B-Instruct \
    --mode real_new_knowledge
CUDA_VISIBLE_DEVICES=0 python query_new_knowledge_local.py \
    --data_path ../../../datasets/new_knowledge_datasets/real_new_knowledge.json \
    --conflict_path ../../../datasets/new_knowledge_datasets/real_conflict_study/selected_pairs.json \
    --save_root ../../../results/in-context_learning/conflict_study/real_new_knowledge/Qwen3-8B \
    --vllm_model ../../../models/Qwen3-8B \
    --mode real_new_knowledge
python query_new_knowledge_openai.py \
    --data_path ../../../datasets/new_knowledge_datasets/real_new_knowledge.json \
    --conflict_path ../../../datasets/new_knowledge_datasets/real_conflict_study/selected_pairs.json \
    --save_root ../../../results/in-context_learning/conflict_study/real_new_knowledge/GPT-4o-Mini-2024-07-18 \
    --mode real_new_knowledge
python judge_conflict.py \
    --root_dir ../../../results/in-context_learning/conflict_study/real_new_knowledge/Aya-Expanse-8B
python judge_conflict.py \
    --root_dir ../../../results/in-context_learning/conflict_study/real_new_knowledge/Llama-3.1-8B-Instruct
python judge_conflict.py \
    --root_dir ../../../results/in-context_learning/conflict_study/real_new_knowledge/Qwen3-8B
python judge_conflict.py \
    --root_dir ../../../results/in-context_learning/conflict_study/real_new_knowledge/GPT-4o-Mini-2024-07-18
python parse_results.py \
    --model_folder ../../../results/in-context_learning/conflict_study/real_new_knowledge/Aya-Expanse-8B
python parse_results.py \
    --model_folder ../../../results/in-context_learning/conflict_study/real_new_knowledge/Llama-3.1-8B-Instruct
python parse_results.py \
    --model_folder ../../../results/in-context_learning/conflict_study/real_new_knowledge/Qwen3-8B
python parse_results.py \
    --model_folder ../../../results/in-context_learning/conflict_study/real_new_knowledge/GPT-4o-Mini-2024-07-18
python visualization_barplot.py \
    --root_dir ../../../results/in-context_learning/conflict_study/real_new_knowledge/Aya-Expanse-8B
python visualization_barplot.py \
    --root_dir ../../../results/in-context_learning/conflict_study/real_new_knowledge/Llama-3.1-8B-Instruct
python visualization_barplot.py \
    --root_dir ../../../results/in-context_learning/conflict_study/real_new_knowledge/Qwen3-8B
python visualization_barplot.py \
    --root_dir ../../../results/in-context_learning/conflict_study/real_new_knowledge/GPT-4o-Mini-2024-07-18
python aggregate_results.py \
    --base_dir ../../../results/in-context_learning/conflict_study/real_new_knowledge \
    --model_name Aya-Expanse-8B
python aggregate_results.py \
    --base_dir ../../../results/in-context_learning/conflict_study/real_new_knowledge \
    --model_name Llama-3.1-8B-Instruct
python aggregate_results.py \
    --base_dir ../../../results/in-context_learning/conflict_study/real_new_knowledge \
    --model_name Qwen3-8B
python aggregate_results.py \
    --base_dir ../../../results/in-context_learning/conflict_study/real_new_knowledge \
    --model_name GPT-4o-Mini-2024-07-18
python visualization_boxplot.py \
    --root_dir ../../../results/in-context_learning/conflict_study/real_new_knowledge \
    --save_path ../../../results/in-context_learning/conflict_study/real_new_knowledge/box.png