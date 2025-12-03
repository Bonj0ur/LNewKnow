# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import argparse
import pandas as pd
from functools import reduce

LANG_OPTIMIZED = {
    "Aya-Expanse-8B": ["Chinese", "English", "French", "Hindi", "Italian", "Japanese", "Korean", "Portuguese", "Spanish"],
    "Llama-3.1-8B-Instruct": ["English", "French", "Italian", "Portuguese", "Hindi", "Spanish", "Thai"],
    "Qwen3-8B": ["English", "French", "Portuguese", "Swedish", "Danish", "Spanish", "Italian", "Welsh", "Hindi", "Chinese", "Tamil", "Thai", "Japanese", "Korean", "Swahili"]
}

LANG3_MAP = {
    "eng": "English",
    "zho": "Chinese",
    "jpn": "Japanese",
    "fra": "French",
    "spa": "Spanish",
    "ita": "Italian",
    "por": "Portuguese",
    "swe": "Swedish",
    "kor": "Korean",
    "dan": "Danish",
    "tha": "Thai",
    "hin": "Hindi",
    "tam": "Tamil",
    "mon": "Mongolian",
    "cym": "Welsh",
    "swa": "Swahili",
    "tuk": "Turkmen",
    "gla": "Scottish Gaelic",
    "zul": "Zulu"
}

LANG2_MAP = {
    "en": "English",
    "zh_CN": "Chinese",
    "ja": "Japanese",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "sv": "Swedish",
    "ko": "Korean",
    "da": "Danish",
    "th": "Thai",
    "hi": "Hindi",
    "ta": "Tamil",
    "mn": "Mongolian",
    "cy": "Welsh",
    "sw": "Swahili",
    "tk": "Turkmen",
    "gd": "Scottish Gaelic",
    "zu": "Zulu"
}

MORPHSCORE_MAP = {
    "danish": "Danish",
    "english": "English",
    "french": "French",
    "hindi": "Hindi",
    "italian": "Italian",
    "japanese": "Japanese",
    "korean": "Korean",
    "mandarin": "Chinese",
    "portuguese": "Portuguese",
    "scottish_gaelic": "Scottish Gaelic",
    "spanish": "Spanish",
    "swedish": "Swedish",
    "tamil": "Tamil",
    "thai": "Thai",
    "welsh": "Welsh"
}

def load_training_proportion(path):
    df = pd.read_csv(path)
    df = df.rename(columns={"primary_language": "language"})
    df["language"] = df["language"].map(LANG3_MAP)
    df["data_proportion"] = df["mean ± std"].str.split("±").str[0].astype(float)
    return df[["language", "data_proportion"]]

def load_model_optimized(model_name):
    langs = list(LANG3_MAP.values())
    optimized = [1.0 if lang in LANG_OPTIMIZED.get(model_name, []) else 0.0 for lang in langs]
    return pd.DataFrame({"language": langs, "optimized": optimized})

def load_average_rank(path):
    with open(path) as f:
        data = json.load(f)["Average Rank"]
    df = pd.DataFrame.from_dict(data, orient="index", columns=["average_rank"]).reset_index()
    df = df[~df["index"].str.contains("All")]
    df = df.rename(columns={"index": "language"})
    df["language"] = df["language"].map(LANG2_MAP)
    return df

def load_efficiency(path):
    with open(path) as f:
        data = json.load(f)
    rows = []
    for lang, res in data.items():
        rows.append({
            "language": LANG2_MAP.get(lang),
            "renyi_efficiency": res["renyi_score"],
            "token_count": res["token_count"]
        })
    return pd.DataFrame(rows)

def load_morphscore(path):
    with open(path) as f:
        data = json.load(f)
    rows = []
    for lang, res in data.items():
        if lang in MORPHSCORE_MAP:
            rows.append({
                "language": MORPHSCORE_MAP[lang],
                "recall": res["morphscore_recall"],
                "precision": res["morphscore_precision"],
                "num_samples": res["num_samples"]
            })
    return pd.DataFrame(rows)

def load_all_morphscores(base_dir, model_name):
    dfs = []
    for i in range(2):
        for j in range(2):
            filename = f"morphscore_freq_scale_{i}_exclude_single_tok_{j}_exclude_single_morpheme_{j}.json"
            path = os.path.join(base_dir, model_name, filename)
            df = load_morphscore(path)
            df = df.rename(columns={
                "recall": f"morphscore_recall_f{i}_tok{j}_morpheme{j}",
                "precision": f"morphscore_precision_f{i}_tok{j}_morpheme{j}",
                "num_samples": f"morphscore_num_samples_f{i}_tok{j}_morpheme{j}"
            })
            dfs.append(df)
    return dfs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--training_proportion_path", default="../linguistic_stats/results/cc-stats/cc_proportion.csv")
    parser.add_argument("--token_average_rank_dir", default="../linguistic_stats/results/tokenization_analysis/overlap/")
    parser.add_argument("--token_efficiency_dir", default="../linguistic_stats/results/tokenization_analysis/efficiency/flores-dev/")
    parser.add_argument("--token_morphological_alignment_dir", default="../linguistic_stats/results/tokenization_analysis/morphological_alignment/")
    args = parser.parse_args()

    save_path = f"./datasets/per_lang_stats_{args.model_name}.csv"
    model = args.model_name

    all_dfs = [
        load_training_proportion(args.training_proportion_path),
        load_model_optimized(model),
        load_average_rank(os.path.join(args.token_average_rank_dir, model, "tokenizer_properties.json")),
        load_efficiency(os.path.join(args.token_efficiency_dir, model, "renyi_scores.json")),
        *load_all_morphscores(args.token_morphological_alignment_dir, model)
    ]

    merged_df = reduce(lambda left, right: pd.merge(left, right, on="language", how="outer"), all_dfs)
    merged_df.to_csv(save_path, index=False)