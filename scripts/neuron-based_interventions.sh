cd src/lang-specific_neurons/
python flores_data_preparation.py \
    --model_name Llama-3.1-8B-Instruct \
    --save_dir ./outputs/token_ids
python flores_data_preparation.py \
    --model_name Aya-Expanse-8B \
    --save_dir ./outputs/token_ids
python flores_data_preparation.py \
    --model_name Qwen3-8B \
    --save_dir ./outputs/token_ids
bash scripts/get_activation.sh
python language_similarity.py \
    --model_name Aya-Expanse-8B
python language_similarity.py \
    --model_name Llama-3.1-8B-Instruct
python language_similarity.py \
    --model_name Qwen3-8B
CUDA_VISIBLE_DEVICES=0 python neuron_manipulation_generation.py \
    --model_name Aya-Expanse-8B \
    --input_dir ../../results/in-context_learning/transfer_study/generated_new_knowledge \
    --save_dir ../../results/neuron_manipulation/ \
    --deactivation
CUDA_VISIBLE_DEVICES=0 python neuron_manipulation_generation.py \
    --model_name Llama-3.1-8B-Instruct \
    --input_dir ../../results/in-context_learning/transfer_study/generated_new_knowledge \
    --save_dir ../../results/neuron_manipulation/ \
    --deactivation
CUDA_VISIBLE_DEVICES=0 python neuron_manipulation_generation.py \
    --model_name Qwen3-8B \
    --input_dir ../../results/in-context_learning/transfer_study/generated_new_knowledge \
    --save_dir ../../results/neuron_manipulation/ \
    --deactivation
CUDA_VISIBLE_DEVICES=0 python neuron_manipulation_generation.py \
    --model_name Aya-Expanse-8B \
    --input_dir ../../results/in-context_learning/transfer_study/generated_new_knowledge \
    --save_dir ../../results/neuron_manipulation/ \
    --activation
CUDA_VISIBLE_DEVICES=0 python neuron_manipulation_generation.py \
    --model_name Llama-3.1-8B-Instruct \
    --input_dir ../../results/in-context_learning/transfer_study/generated_new_knowledge \
    --save_dir ../../results/neuron_manipulation/ \
    --activation
CUDA_VISIBLE_DEVICES=0 python neuron_manipulation_generation.py \
    --model_name Qwen3-8B \
    --input_dir ../../results/in-context_learning/transfer_study/generated_new_knowledge \
    --save_dir ../../results/neuron_manipulation/ \
    --activation
python neuron_manipulation_judge.py \
    --root_dir ../../results/neuron_manipulation/deactivation/Aya-Expanse-8B
python neuron_manipulation_judge.py \
    --root_dir ../../results/neuron_manipulation/deactivation/Llama-3.1-8B-Instruct
python neuron_manipulation_judge.py \
    --root_dir ../../results/neuron_manipulation/deactivation/Qwen3-8B
python neuron_manipulation_judge.py \
    --root_dir ../../results/neuron_manipulation/activation/Aya-Expanse-8B
python neuron_manipulation_judge.py \
    --root_dir ../../results/neuron_manipulation/activation/Llama-3.1-8B-Instruct
python neuron_manipulation_judge.py \
    --root_dir ../../results/neuron_manipulation/activation/Qwen3-8B
python neuron_manipulation_parse_results.py \
    --result_dir ../../results/neuron_manipulation/deactivation/ \
    --model_name Aya-Expanse-8B
python neuron_manipulation_parse_results.py \
    --result_dir ../../results/neuron_manipulation/deactivation/ \
    --model_name Llama-3.1-8B-Instruct
python neuron_manipulation_parse_results.py \
    --result_dir ../../results/neuron_manipulation/deactivation/ \
    --model_name Qwen3-8B
python neuron_manipulation_parse_results.py \
    --result_dir ../../results/neuron_manipulation/activation/ \
    --model_name Aya-Expanse-8B
python neuron_manipulation_parse_results.py \
    --result_dir ../../results/neuron_manipulation/activation/ \
    --model_name Llama-3.1-8B-Instruct
python neuron_manipulation_parse_results.py \
    --result_dir ../../results/neuron_manipulation/activation/ \
    --model_name Qwen3-8B
python performance_measurement.py \
    --mode deactivation \
    --model_name Aya-Expanse-8B \
    --baseline_dir ../../results/in-context_learning/transfer_study/generated_new_knowledge \
    --manipulation_dir ../../results/neuron_manipulation/
python performance_measurement.py \
    --mode deactivation \
    --model_name Llama-3.1-8B-Instruct \
    --baseline_dir ../../results/in-context_learning/transfer_study/generated_new_knowledge \
    --manipulation_dir ../../results/neuron_manipulation/
python performance_measurement.py \
    --mode deactivation \
    --model_name Qwen3-8B \
    --baseline_dir ../../results/in-context_learning/transfer_study/generated_new_knowledge \
    --manipulation_dir ../../results/neuron_manipulation/
python performance_measurement.py \
    --mode activation \
    --model_name Aya-Expanse-8B \
    --baseline_dir ../../results/in-context_learning/transfer_study/generated_new_knowledge \
    --manipulation_dir ../../results/neuron_manipulation/
python performance_measurement.py \
    --mode activation \
    --model_name Llama-3.1-8B-Instruct \
    --baseline_dir ../../results/in-context_learning/transfer_study/generated_new_knowledge \
    --manipulation_dir ../../results/neuron_manipulation/
python performance_measurement.py \
    --mode activation \
    --model_name Qwen3-8B \
    --baseline_dir ../../results/in-context_learning/transfer_study/generated_new_knowledge \
    --manipulation_dir ../../results/neuron_manipulation/