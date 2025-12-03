cd src/in-context_learning/resist_study/
CUDA_VISIBLE_DEVICES=0 python query_general_knowledge.py \
    --input_path ../../../datasets/general_knowledge_datasets/generated_general_knowledge.json \
    --save_dir ../../../results/in-context_learning/resist_study/generated_general_knowledge/Aya-Expanse-8B \
    --backend vllm \
    --vllm_model ../../../models/Aya-Expanse-8B
CUDA_VISIBLE_DEVICES=0 python query_general_knowledge.py \
    --input_path ../../../datasets/general_knowledge_datasets/generated_general_knowledge.json \
    --save_dir ../../../results/in-context_learning/resist_study/generated_general_knowledge/Llama-3.1-8B-Instruct \
    --backend vllm \
    --vllm_model ../../../models/Llama-3.1-8B-Instruct
CUDA_VISIBLE_DEVICES=0 python query_general_knowledge.py \
    --input_path ../../../datasets/general_knowledge_datasets/generated_general_knowledge.json \
    --save_dir ../../../results/in-context_learning/resist_study/generated_general_knowledge/Qwen3-8B \
    --backend vllm \
    --vllm_model ../../../models/Qwen3-8B
python query_general_knowledge.py \
    --input_path ../../../datasets/general_knowledge_datasets/generated_general_knowledge.json \
    --save_dir ../../../results/in-context_learning/resist_study/generated_general_knowledge/GPT-4o-Mini-2024-07-18 \
    --backend openai \
    --openai_model gpt-4o-mini-2024-07-18
python judge_general_knowledge.py \
    --root_dir ../../../results/in-context_learning/resist_study/generated_general_knowledge/Aya-Expanse-8B
python judge_general_knowledge.py \
    --root_dir ../../../results/in-context_learning/resist_study/generated_general_knowledge/Llama-3.1-8B-Instruct
python judge_general_knowledge.py \
    --root_dir ../../../results/in-context_learning/resist_study/generated_general_knowledge/Qwen3-8B
python judge_general_knowledge.py \
    --root_dir ../../../results/in-context_learning/resist_study/generated_general_knowledge/GPT-4o-Mini-2024-07-18
python parse_results.py \
    --result_dir ../../../results/in-context_learning/resist_study/generated_general_knowledge/ \
    --model_name Aya-Expanse-8B
python parse_results.py \
    --result_dir ../../../results/in-context_learning/resist_study/generated_general_knowledge/ \
    --model_name Llama-3.1-8B-Instruct
python parse_results.py \
    --result_dir ../../../results/in-context_learning/resist_study/generated_general_knowledge/ \
    --model_name Qwen3-8B
python parse_results.py \
    --result_dir ../../../results/in-context_learning/resist_study/generated_general_knowledge/ \
    --model_name GPT-4o-Mini-2024-07-18
python visualization_inequality.py \
    --result_dir ../../../results/in-context_learning/resist_study/generated_general_knowledge/ \
    --model_name Aya-Expanse-8B \
    --baseline_path ../../../results/verify/generated_general_knowledge/summary_accuracy_table.csv
python visualization_inequality.py \
    --result_dir ../../../results/in-context_learning/resist_study/generated_general_knowledge/ \
    --model_name Llama-3.1-8B-Instruct \
    --baseline_path ../../../results/verify/generated_general_knowledge/summary_accuracy_table.csv
python visualization_inequality.py \
    --result_dir ../../../results/in-context_learning/resist_study/generated_general_knowledge/ \
    --model_name Qwen3-8B \
    --baseline_path ../../../results/verify/generated_general_knowledge/summary_accuracy_table.csv
python visualization_inequality.py \
    --result_dir ../../../results/in-context_learning/resist_study/generated_general_knowledge/ \
    --model_name GPT-4o-Mini-2024-07-18 \
    --baseline_path ../../../results/verify/generated_general_knowledge/summary_accuracy_table.csv
# --------------------------------------------------------------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python query_general_knowledge.py \
    --input_path ../../../datasets/general_knowledge_datasets/real_general_knowledge.json \
    --save_dir ../../../results/in-context_learning/resist_study/real_general_knowledge/Aya-Expanse-8B \
    --backend vllm \
    --vllm_model ../../../models/Aya-Expanse-8B
CUDA_VISIBLE_DEVICES=0 python query_general_knowledge.py \
    --input_path ../../../datasets/general_knowledge_datasets/real_general_knowledge.json \
    --save_dir ../../../results/in-context_learning/resist_study/real_general_knowledge/Llama-3.1-8B-Instruct \
    --backend vllm \
    --vllm_model ../../../models/Llama-3.1-8B-Instruct
CUDA_VISIBLE_DEVICES=0 python query_general_knowledge.py \
    --input_path ../../../datasets/general_knowledge_datasets/real_general_knowledge.json \
    --save_dir ../../../results/in-context_learning/resist_study/real_general_knowledge/Qwen3-8B \
    --backend vllm \
    --vllm_model ../../../models/Qwen3-8B
python query_general_knowledge.py \
    --input_path ../../../datasets/general_knowledge_datasets/real_general_knowledge.json \
    --save_dir ../../../results/in-context_learning/resist_study/real_general_knowledge/GPT-4o-Mini-2024-07-18 \
    --backend openai \
    --openai_model gpt-4o-mini-2024-07-18
python judge_general_knowledge.py \
    --root_dir ../../../results/in-context_learning/resist_study/real_general_knowledge/Aya-Expanse-8B
python judge_general_knowledge.py \
    --root_dir ../../../results/in-context_learning/resist_study/real_general_knowledge/Llama-3.1-8B-Instruct
python judge_general_knowledge.py \
    --root_dir ../../../results/in-context_learning/resist_study/real_general_knowledge/Qwen3-8B
python judge_general_knowledge.py \
    --root_dir ../../../results/in-context_learning/resist_study/real_general_knowledge/GPT-4o-Mini-2024-07-18
python parse_results.py \
    --result_dir ../../../results/in-context_learning/resist_study/real_general_knowledge/ \
    --model_name Aya-Expanse-8B
python parse_results.py \
    --result_dir ../../../results/in-context_learning/resist_study/real_general_knowledge/ \
    --model_name Llama-3.1-8B-Instruct
python parse_results.py \
    --result_dir ../../../results/in-context_learning/resist_study/real_general_knowledge/ \
    --model_name Qwen3-8B
python parse_results.py \
    --result_dir ../../../results/in-context_learning/resist_study/real_general_knowledge/ \
    --model_name GPT-4o-Mini-2024-07-18
python visualization_inequality.py \
    --result_dir ../../../results/in-context_learning/resist_study/real_general_knowledge/ \
    --model_name Aya-Expanse-8B \
    --baseline_path ../../../results/verify/real_general_knowledge/summary_accuracy_table.csv
python visualization_inequality.py \
    --result_dir ../../../results/in-context_learning/resist_study/real_general_knowledge/ \
    --model_name Llama-3.1-8B-Instruct \
    --baseline_path ../../../results/verify/real_general_knowledge/summary_accuracy_table.csv
python visualization_inequality.py \
    --result_dir ../../../results/in-context_learning/resist_study/real_general_knowledge/ \
    --model_name Qwen3-8B \
    --baseline_path ../../../results/verify/real_general_knowledge/summary_accuracy_table.csv
python visualization_inequality.py \
    --result_dir ../../../results/in-context_learning/resist_study/real_general_knowledge/ \
    --model_name GPT-4o-Mini-2024-07-18 \
    --baseline_path ../../../results/verify/real_general_knowledge/summary_accuracy_table.csv