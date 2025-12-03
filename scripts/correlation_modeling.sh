cd src/cor_stats
python agg_per_pair_stats.py \
    --model_name Aya-Expanse-8B
python agg_per_pair_stats.py \
    --model_name Llama-3.1-8B-Instruct
python agg_per_pair_stats.py \
    --model_name Qwen3-8B
python agg_per_pair_stats.py \
    --model_name GPT-4o-Mini-2024-07-18
python agg_per_lang_stats.py \
    --model_name Aya-Expanse-8B
python agg_per_lang_stats.py \
    --model_name Llama-3.1-8B-Instruct
python agg_per_lang_stats.py \
    --model_name Qwen3-8B
python agg_per_lang_stats.py \
    --model_name GPT-4o-Mini-2024-07-18
python agg_transfer_results.py \
    --result_dir ../../results
python transfer_cor_analysis.py \
    --model_name Aya-Expanse-8B \
    --mode_name generated_new_knowledge \
    --method_name in-context_learning
python transfer_cor_analysis.py \
    --model_name Llama-3.1-8B-Instruct \
    --mode_name generated_new_knowledge \
    --method_name in-context_learning
python transfer_cor_analysis.py \
    --model_name Qwen3-8B \
    --mode_name generated_new_knowledge \
    --method_name in-context_learning
python transfer_cor_analysis.py \
    --model_name GPT-4o-Mini-2024-07-18 \
    --mode_name generated_new_knowledge \
    --method_name in-context_learning
python transfer_cor_analysis.py \
    --model_name Aya-Expanse-8B \
    --mode_name real_new_knowledge \
    --method_name in-context_learning
python transfer_cor_analysis.py \
    --model_name Llama-3.1-8B-Instruct \
    --mode_name real_new_knowledge \
    --method_name in-context_learning
python transfer_cor_analysis.py \
    --model_name Qwen3-8B \
    --mode_name real_new_knowledge \
    --method_name in-context_learning
python transfer_cor_analysis.py \
    --model_name GPT-4o-Mini-2024-07-18 \
    --mode_name real_new_knowledge \
    --method_name in-context_learning
python transfer_cor_analysis.py \
    --model_name Aya-Expanse-8B \
    --mode_name generated_new_knowledge \
    --method_name fine-tuning
python transfer_cor_analysis.py \
    --model_name Llama-3.1-8B-Instruct \
    --mode_name generated_new_knowledge \
    --method_name fine-tuning
python transfer_cor_analysis.py \
    --model_name Qwen3-8B \
    --mode_name generated_new_knowledge \
    --method_name fine-tuning
python transfer_cor_analysis.py \
    --model_name GPT-4o-Mini-2024-07-18 \
    --mode_name generated_new_knowledge \
    --method_name fine-tuning
python transfer_cor_analysis.py \
    --model_name Aya-Expanse-8B \
    --mode_name real_new_knowledge \
    --method_name fine-tuning
python transfer_cor_analysis.py \
    --model_name Llama-3.1-8B-Instruct \
    --mode_name real_new_knowledge \
    --method_name fine-tuning
python transfer_cor_analysis.py \
    --model_name Qwen3-8B \
    --mode_name real_new_knowledge \
    --method_name fine-tuning
python transfer_cor_analysis.py \
    --model_name GPT-4o-Mini-2024-07-18 \
    --mode_name real_new_knowledge \
    --method_name fine-tuning
# --------------------------------------------------------------------------------------------------------------------------------
python agg_quality_results.py \
    --result_dir ../../results/
python quality_cor_analysis.py \
    --setting "fine-tuning|effective_performance" \
    --mode_name generated \
    --model_name Aya-Expanse-8B
python quality_cor_analysis.py \
    --setting "fine-tuning|robust_performance" \
    --mode_name generated \
    --model_name Aya-Expanse-8B
python quality_cor_analysis.py \
    --setting "in-context_learning|robust_performance" \
    --mode_name generated \
    --model_name Aya-Expanse-8B
python quality_cor_analysis.py \
    --setting "fine-tuning|effective_performance" \
    --mode_name real \
    --model_name Aya-Expanse-8B
python quality_cor_analysis.py \
    --setting "fine-tuning|robust_performance" \
    --mode_name real \
    --model_name Aya-Expanse-8B
python quality_cor_analysis.py \
    --setting "in-context_learning|robust_performance" \
    --mode_name real \
    --model_name Aya-Expanse-8B
python quality_cor_analysis.py \
    --setting "fine-tuning|effective_performance" \
    --mode_name generated \
    --model_name Llama-3.1-8B-Instruct
python quality_cor_analysis.py \
    --setting "fine-tuning|robust_performance" \
    --mode_name generated \
    --model_name Llama-3.1-8B-Instruct
python quality_cor_analysis.py \
    --setting "in-context_learning|robust_performance" \
    --mode_name generated \
    --model_name Llama-3.1-8B-Instruct
python quality_cor_analysis.py \
    --setting "fine-tuning|effective_performance" \
    --mode_name real \
    --model_name Llama-3.1-8B-Instruct
python quality_cor_analysis.py \
    --setting "fine-tuning|robust_performance" \
    --mode_name real \
    --model_name Llama-3.1-8B-Instruct
python quality_cor_analysis.py \
    --setting "in-context_learning|robust_performance" \
    --mode_name real \
    --model_name Llama-3.1-8B-Instruct
python quality_cor_analysis.py \
    --setting "fine-tuning|effective_performance" \
    --mode_name generated \
    --model_name Qwen3-8B
python quality_cor_analysis.py \
    --setting "fine-tuning|robust_performance" \
    --mode_name generated \
    --model_name Qwen3-8B
python quality_cor_analysis.py \
    --setting "in-context_learning|robust_performance" \
    --mode_name generated \
    --model_name Qwen3-8B
python quality_cor_analysis.py \
    --setting "fine-tuning|effective_performance" \
    --mode_name real \
    --model_name Qwen3-8B
python quality_cor_analysis.py \
    --setting "fine-tuning|robust_performance" \
    --mode_name real \
    --model_name Qwen3-8B
python quality_cor_analysis.py \
    --setting "in-context_learning|robust_performance" \
    --mode_name real \
    --model_name Qwen3-8B
python quality_cor_analysis.py \
    --setting "fine-tuning|effective_performance" \
    --mode_name generated \
    --model_name GPT-4o-Mini-2024-07-18
python quality_cor_analysis.py \
    --setting "fine-tuning|robust_performance" \
    --mode_name generated \
    --model_name GPT-4o-Mini-2024-07-18
python quality_cor_analysis.py \
    --setting "in-context_learning|robust_performance" \
    --mode_name generated \
    --model_name GPT-4o-Mini-2024-07-18
python quality_cor_analysis.py \
    --setting "fine-tuning|effective_performance" \
    --mode_name real \
    --model_name GPT-4o-Mini-2024-07-18
python quality_cor_analysis.py \
    --setting "fine-tuning|robust_performance" \
    --mode_name real \
    --model_name GPT-4o-Mini-2024-07-18
python quality_cor_analysis.py \
    --setting "in-context_learning|robust_performance" \
    --mode_name real \
    --model_name GPT-4o-Mini-2024-07-18