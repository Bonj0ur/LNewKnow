cd src/fine-tuning/resist_study/
python build_datasets.py \
    --input ../../../datasets/general_knowledge_datasets/generated_general_knowledge.json \
    --output_dir ../../../datasets/general_knowledge_datasets/generated_resist_study
python upload_openai_datasets.py \
    --folder ../../../datasets/general_knowledge_datasets/generated_resist_study/train \
    --output ../../utils/generated_resist_study_datasets.json
bash scripts/fine-tuning_generated_local_Aya.sh
bash scripts/fine-tuning_generated_local_Llama.sh
bash scripts/fine-tuning_generated_local_Qwen.sh
python fine-tuning_openai.py \
    --start_epoch 0 \
    --model_record ../../utils/generated_resist_study_models.json \
    --dataset_record ../../utils/generated_resist_study_datasets.json \ 
    --suffix generated_resist_study
bash scripts/query_robust_generated_local_Aya.sh
bash scripts/query_robust_generated_local_Llama.sh
bash scripts/query_robust_generated_local_Qwen.sh
python query_robust_openai.py \
    --test_dir ../../../datasets/general_knowledge_datasets/generated_resist_study/test \
    --model_dict ../../utils/generated_resist_study_models.json \
    --output_dir ../../../results/fine-tuning/resist_study/generated_general_knowledge/GPT-4o-Mini-2024-07-18
python judge_robust.py \
    --root_dir ../../../results/fine-tuning/resist_study/generated_general_knowledge/Aya-Expanse-8B
python judge_robust.py \
    --root_dir ../../../results/fine-tuning/resist_study/generated_general_knowledge/Llama-3.1-8B-Instruct
python judge_robust.py \
    --root_dir ../../../results/fine-tuning/resist_study/generated_general_knowledge/Qwen3-8B
python judge_robust.py \
    --root_dir ../../../results/fine-tuning/resist_study/generated_general_knowledge/GPT-4o-Mini-2024-07-18
python parse_results.py \
    --result_dir ../../../results/fine-tuning/resist_study/generated_general_knowledge \
    --model_name Aya-Expanse-8B
python parse_results.py \
    --result_dir ../../../results/fine-tuning/resist_study/generated_general_knowledge \
    --model_name Llama-3.1-8B-Instruct
python parse_results.py \
    --result_dir ../../../results/fine-tuning/resist_study/generated_general_knowledge \
    --model_name Qwen3-8B
python parse_results.py \
    --result_dir ../../../results/fine-tuning/resist_study/generated_general_knowledge \
    --model_name GPT-4o-Mini-2024-07-18
python visualization_inequality.py \
    --result_dir ../../../results/fine-tuning/resist_study/generated_general_knowledge \
    --model_name Aya-Expanse-8B \
    --baseline_path ../../../results/verify/generated_general_knowledge/summary_accuracy_table.csv
python visualization_inequality.py \
    --result_dir ../../../results/fine-tuning/resist_study/generated_general_knowledge \
    --model_name Llama-3.1-8B-Instruct \
    --baseline_path ../../../results/verify/generated_general_knowledge/summary_accuracy_table.csv
python visualization_inequality.py \
    --result_dir ../../../results/fine-tuning/resist_study/generated_general_knowledge \
    --model_name Qwen3-8B \
    --baseline_path ../../../results/verify/generated_general_knowledge/summary_accuracy_table.csv
python visualization_inequality.py \
    --result_dir ../../../results/fine-tuning/resist_study/generated_general_knowledge \
    --model_name GPT-4o-Mini-2024-07-18 \
    --baseline_path ../../../results/verify/generated_general_knowledge/summary_accuracy_table.csv
# --------------------------------------------------------------------------------------------------------------------------------
python build_datasets.py \
    --input ../../../datasets/general_knowledge_datasets/real_general_knowledge.json \
    --output_dir ../../../datasets/general_knowledge_datasets/real_resist_study
python upload_openai_datasets.py \
    --folder ../../../datasets/general_knowledge_datasets/real_resist_study/train \
    --output ../../utils/real_resist_study_datasets.json
bash scripts/fine-tuning_real_local_Aya.sh
bash scripts/fine-tuning_real_local_Llama.sh
bash scripts/fine-tuning_real_local_Qwen.sh
python fine-tuning_openai.py \
    --start_epoch 0 \
    --model_record ../../utils/real_resist_study_models.json \
    --dataset_record ../../utils/real_resist_study_datasets.json \ 
    --suffix real_resist_study
bash scripts/query_robust_real_local_Aya.sh
bash scripts/query_robust_real_local_Llama.sh
bash scripts/query_robust_real_local_Qwen.sh
python query_robust_openai.py \
    --test_dir ../../../datasets/general_knowledge_datasets/real_resist_study/test \
    --model_dict ../../utils/real_resist_study_models.json \
    --output_dir ../../../results/fine-tuning/resist_study/real_general_knowledge/GPT-4o-Mini-2024-07-18
python judge_robust.py \
    --root_dir ../../../results/fine-tuning/resist_study/real_general_knowledge/Aya-Expanse-8B
python judge_robust.py \
    --root_dir ../../../results/fine-tuning/resist_study/real_general_knowledge/Llama-3.1-8B-Instruct
python judge_robust.py \
    --root_dir ../../../results/fine-tuning/resist_study/real_general_knowledge/Qwen3-8B
python judge_robust.py \
    --root_dir ../../../results/fine-tuning/resist_study/real_general_knowledge/GPT-4o-Mini-2024-07-18
python parse_results.py \
    --result_dir ../../../results/fine-tuning/resist_study/real_general_knowledge \
    --model_name Aya-Expanse-8B
python parse_results.py \
    --result_dir ../../../results/fine-tuning/resist_study/real_general_knowledge \
    --model_name Llama-3.1-8B-Instruct
python parse_results.py \
    --result_dir ../../../results/fine-tuning/resist_study/real_general_knowledge \
    --model_name Qwen3-8B
python parse_results.py \
    --result_dir ../../../results/fine-tuning/resist_study/real_general_knowledge \
    --model_name GPT-4o-Mini-2024-07-18
python visualization_inequality.py \
    --result_dir ../../../results/fine-tuning/resist_study/real_general_knowledge \
    --model_name Aya-Expanse-8B \
    --baseline_path ../../../results/verify/real_general_knowledge/summary_accuracy_table.csv
python visualization_inequality.py \
    --result_dir ../../../results/fine-tuning/resist_study/real_general_knowledge \
    --model_name Llama-3.1-8B-Instruct \
    --baseline_path ../../../results/verify/real_general_knowledge/summary_accuracy_table.csv
python visualization_inequality.py \
    --result_dir ../../../results/fine-tuning/resist_study/real_general_knowledge \
    --model_name Qwen3-8B \
    --baseline_path ../../../results/verify/real_general_knowledge/summary_accuracy_table.csv
python visualization_inequality.py \
    --result_dir ../../../results/fine-tuning/resist_study/real_general_knowledge \
    --model_name GPT-4o-Mini-2024-07-18 \
    --baseline_path ../../../results/verify/real_general_knowledge/summary_accuracy_table.csv