# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import tiktoken
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from utils import TOKENIZERS_BY_TYPE
from transformers import AutoTokenizer
from typing import Dict, List, Any, Optional, Union

def get_tokenizer(model_type):
    model = TOKENIZERS_BY_TYPE[model_type]
    if model.startswith('gpt'):
        enc = tiktoken.encoding_for_model(model)
        return enc, "openai"
    else:
        return AutoTokenizer.from_pretrained(model), "hf"

class MorphScore:
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        self.config = {
            # Filtering Flags
            'data_dir': './datasets/morphological_clean_datasets/',
            'unique_only': True,
            'stem_eq_lemma': True,
            'exclude_numbers': True,
            'language_subset': [],
            'splits': ['train', 'dev', 'test'],
            # Scoring Flags
            'freq_scale': False,
            'exclude_single_tok': True,
            'exclude_single_morpheme': True,
            'single_tok_point': 1,
            'correct_point': 1,
            'partial_point': 0.5,
            # Breakdown Flags
            'by_split': False,
            'by_pos': False,
            # Tokenizer Settings
            'subword_prefix': ''
        }

        if config_path:
            self._load_config(config_path)
        
        self.config.update(kwargs)

        self._validate_config()

    def _load_config(self, config_path: str):
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
            self.config.update(file_config)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")

    def _validate_config(self):
        if not isinstance(self.config['single_tok_point'], (int, float)):
            raise ValueError("single_tok_point must be numeric")
        if not isinstance(self.config['correct_point'], (int, float)):
            raise ValueError("correct_point must be numeric")
        if not isinstance(self.config['partial_point'], (int, float)):
            raise ValueError("partial_point must be numeric")        
        if not isinstance(self.config['language_subset'], list):
            raise ValueError("language_subset must be a list")
        if not isinstance(self.config['splits'], list):
            raise ValueError("splits must be a list")

    def _load_dataset(self, language_or_filename: str) -> pd.DataFrame:
        if not language_or_filename.endswith('.csv'):
            language = language_or_filename.lower()
            dataset_path = Path(self.config['data_dir']) / f'{language}_data.csv'
        else:
            dataset_path = Path(self.config['data_dir']) / language_or_filename
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found for language: {language}") 
        return pd.read_csv(dataset_path)

    def _filter_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        filtered_df = dataset.copy()
        if 'data_split' in filtered_df.columns:
            split_dfs = []
            for split in self.config['splits']:
                split_dfs.append(filtered_df[filtered_df['data_split'].str.startswith(split)])
            filtered_df = pd.concat(split_dfs)

        if self.config['unique_only'] and 'unique' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['unique'] == 'unique']

        if self.config['stem_eq_lemma'] and all(col in filtered_df.columns for col in ['stem', 'lemma']):
            filtered_df = filtered_df[filtered_df['stem'] == filtered_df['lemma']]

        if self.config['exclude_numbers'] and 'wordform' in filtered_df.columns:
            filtered_df = filtered_df[~filtered_df['wordform'].astype(str).str.contains(r'\d')]
        
        return filtered_df

    def morph_eval(self, morphemes: List[str], tokens: List[str]) -> float:
        if len(tokens) == 1:
            return (np.nan, np.nan) if self.config['exclude_single_tok'] else (self.config['single_tok_point'], self.config['single_tok_point']) 
        
        all_pred_boundaries = []
        idx = 0
        for t in range(len(tokens)):
            tok = tokens[t]
            this_idx = idx + len(tok)
            all_pred_boundaries.append(this_idx)
            idx = this_idx
        
        if len(morphemes) == 2:
            gold_boundary_idx = len(morphemes[0])
            if gold_boundary_idx in all_pred_boundaries:
                return self.config['correct_point'], 1 / len(all_pred_boundaries)
            else:
                return 0, 0

        elif len(morphemes) == 3:
            gold_boundary_indices = [len(morphemes[0]), len(morphemes[0]) + len(morphemes[1])]
            if gold_boundary_indices[0] in all_pred_boundaries and gold_boundary_indices[1] in all_pred_boundaries:
                return self.config['correct_point'], 2 / len(all_pred_boundaries)
            elif gold_boundary_indices[0] in all_pred_boundaries or gold_boundary_indices[1] in all_pred_boundaries:
                return self.config['partial_point'], 1 / len(all_pred_boundaries)
            else:
                return 0, 0

        else:
            if self.config['exclude_single_morpheme']:
                return (np.nan, np.nan)
            else:
                return (self.config['single_tok_point'], self.config['single_tok_point']) if morphemes == tokens else (0, 0)

    def get_morphscore(self, dataset: pd.DataFrame, tok_type, tokenizer, return_df: bool = False) -> tuple:
        required_cols = ['stem', 'lemma', 'preceding_part', 'following_part', 'wordform']
        if not all(col in dataset.columns for col in required_cols):
            raise ValueError(f"Dataset must contain columns: {required_cols}")
        
        if tok_type == 'hf':
            special_toks = tokenizer.special_tokens_map.values()
        elif tok_type == 'openai':
            special_toks = list(tokenizer._special_tokens.keys())

        points_morphscore_recall = []
        points_morphscore_precision = []
        weights = []
        token_char_ratios = []
        matched_subwords = []
        gold_subwords = []
        pred_subwords = []

        def add_nan_values():
            points_morphscore_recall.append(np.nan)
            points_morphscore_precision.append(np.nan)
            matched_subwords.append(np.nan)
            gold_subwords.append(np.nan)
            pred_subwords.append(np.nan)
            weights.append(np.nan)
            token_char_ratios.append(np.nan)
        
        for idx in range(len(dataset)):
            row = dataset.iloc[idx]
            prefix = row['preceding_part']
            suffix = row['following_part']
            stem = row['stem']
            wordform = row['wordform']
            norm_freq = float(row['word_freq_norm'])

            if not isinstance(wordform, str):
                if pd.isna(wordform):
                    add_nan_values()
                    continue
                wordform = str(wordform).strip()
                if not wordform:
                    add_nan_values()
                    continue

            if not wordform or wordform.isspace():
                add_nan_values()
                continue

            morphemes = []
            if not pd.isna(prefix):
                morphemes.append(prefix)
            morphemes.append(stem)
            if not isinstance(suffix, float):
                morphemes.append(suffix)
            
            if tok_type == 'hf':
                token_output = tokenizer(wordform)
                token_ids = token_output['input_ids']
                tokens = [tokenizer.decode(token_id) for token_id in token_ids]
            elif tok_type == 'openai':
                token_ids = tokenizer.encode(wordform)
                tokens = [tokenizer.decode([token_id]) for token_id in token_ids]
            tokens = [t for t in tokens if t not in special_toks]
            if self.config['subword_prefix']:
                tokens = [t.replace(self.config['subword_prefix'], '') for t in tokens]
            
            if len(wordform) > 0:
                token_char_ratios.append(len(tokens) / len(wordform))
            else:
                token_char_ratios.append(np.nan)
            
            point_recall, point_precision = self.morph_eval(morphemes, tokens)

            if self.config['freq_scale'] and not np.isnan(point_recall):
                weights.append(norm_freq)
            elif not np.isnan(point_recall):
                weights.append(1)
            else:
                weights.append(np.nan)
            
            points_morphscore_recall.append(point_recall)
            points_morphscore_precision.append(point_precision)

            n_matched = len(set(tokens) & set(morphemes))
            n_gold = len(morphemes)
            n_pred = len(tokens)
            matched_subwords.append(n_matched)
            gold_subwords.append(n_gold)
            pred_subwords.append(n_pred)
        
        dataset['morphscore_recall'] = points_morphscore_recall
        dataset['morphscore_precision'] = points_morphscore_precision
        dataset['token_char_ratio'] = token_char_ratios
        dataset['matched_subwords'] = matched_subwords
        dataset['gold_subwords'] = gold_subwords
        dataset['pred_subwords'] = pred_subwords

        new_dataset = dataset.dropna(subset=[
            'morphscore_recall', 'morphscore_precision',
            'token_char_ratio', 'matched_subwords',
            'gold_subwords', 'pred_subwords'
        ])
        valid_weights = [w for w in weights if not np.isnan(w)]
        assert len(new_dataset) == len(valid_weights)

        weighted_recall_points = [p * w for p, w in zip(new_dataset['morphscore_recall'], valid_weights)]
        weighted_precision_points = [p * w for p, w in zip(new_dataset['morphscore_precision'], valid_weights)]

        mean_morphscore_recall = float(np.sum(weighted_recall_points) / np.sum(valid_weights))
        mean_morphscore_precision = float(np.sum(weighted_precision_points) / np.sum(valid_weights))

        n_matched = np.sum(new_dataset['matched_subwords'])
        n_gold = np.sum(new_dataset['gold_subwords'])
        n_pred = np.sum(new_dataset['pred_subwords'])
        micro_precision = float(n_matched / n_pred)
        micro_recall = float(n_matched / n_gold)
        if micro_precision + micro_recall == 0:
            micro_f1 = 0.0
        else:
            micro_f1 = float(2 * micro_precision * micro_recall / (micro_precision + micro_recall))
        
        all_precs = [row['matched_subwords'] / row['pred_subwords'] for _, row in new_dataset.iterrows()]
        all_recalls = [row['matched_subwords'] / row['gold_subwords'] for _, row in new_dataset.iterrows()]
        macro_precision = float(np.mean(all_precs))
        macro_recall = float(np.mean(all_recalls))
        if (macro_precision + macro_recall) == 0:
            macro_f1 = 0.0
        else:
            macro_f1 = float(2 * macro_precision * macro_recall / (macro_precision + macro_recall))
        
        mean_token_char_ratio = np.mean(new_dataset['token_char_ratio']) if len(new_dataset['token_char_ratio']) > 0 else 0.0

        results = {
            'morphscore_recall': mean_morphscore_recall,
            'morphscore_precision': mean_morphscore_precision,
            'morphscore_recall_std': np.std(weighted_recall_points) if len(weighted_recall_points) > 1 else 0.0,
            'morphscore_precision_std': np.std(weighted_precision_points) if len(weighted_precision_points) > 1 else 0.0,
            'mean_token_char_ratio': mean_token_char_ratio,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'num_samples': len(new_dataset)
        }

        return (results, new_dataset) if return_df else results

    def eval(self, tok_type, tokenizer, return_df: bool = False) -> Dict[str, Any]:
        results_per_lang = {'config': self.config.copy()}
        new_dataset = None

        if self.config['language_subset']:
            languages = self.config['language_subset']
        else:
            languages = os.listdir(self.config['data_dir'])

        for language in languages:
            dataset = self._load_dataset(language)
            filtered_data = self._filter_dataset(dataset)

            if len(filtered_data) == 0:
                results_per_lang[language] = {'num_samples': 0, 'error': 'No samples after filtering'}
                continue

            if return_df:
                results, new_dataset = self.get_morphscore(filtered_data, tok_type, tokenizer, return_df)
            else:
                results = self.get_morphscore(filtered_data, tok_type, tokenizer, return_df)
            
            if self.config['by_split'] and 'data_split' in filtered_data.columns:
                results['by_split'] = {}
                for split in filtered_data['data_split'].unique():
                    split_data = filtered_data[filtered_data['data_split'] == split]
                    split_results = self.get_morphscore(split_data, tok_type, tokenizer, return_df=False)
                    results['by_split'][split] = split_results

            if self.config['by_pos'] and 'pos' in filtered_data.columns:
                results['by_pos'] = {}
                for pos in filtered_data['pos'].unique():
                    pos_data = filtered_data[filtered_data['pos'] == pos]
                    pos_results = self.get_morphscore(pos_data, tok_type, tokenizer, return_df=False)
                    results['by_pos'][pos] = pos_results
        
            results_per_lang[language] = results
        
        if return_df:
            return results_per_lang, new_dataset
        else:
            return results_per_lang

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",type=str)
    parser.add_argument("--freq_scale",action="store_true")
    parser.add_argument("--exclude_single_tok",action="store_true")
    parser.add_argument("--exclude_single_morpheme",action="store_true")
    parser.add_argument("--language_subset",type=list,default=["danish", "english", "french", "hindi", "italian", "japanese", "korean", "mandarin", "portuguese", "scottish_gaelic", "spanish", "swedish", "tamil", "thai", "welsh"])
    args = parser.parse_args()

    tokenizer, tok_type = get_tokenizer(args.model_name)
    morph_score = MorphScore(language_subset=args.language_subset,freq_scale=args.freq_scale,exclude_single_tok=args.exclude_single_tok,exclude_single_morpheme=args.exclude_single_morpheme)
    results = morph_score.eval(tok_type,tokenizer,return_df=False)

    args.output_path = f'../results/tokenization_analysis/morphological_alignment/{args.model_name}/morphscore_freq_scale_{int(args.freq_scale)}_exclude_single_tok_{int(args.exclude_single_tok)}_exclude_single_morpheme_{int(args.exclude_single_morpheme)}.json'
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)