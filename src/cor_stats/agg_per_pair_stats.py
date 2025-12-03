# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import argparse
import pandas as pd
from functools import reduce

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

def load_and_melt(path, value_name, lang_map=LANG3_MAP):
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.map(lambda x: lang_map.get(x.strip(), x.strip()))
    df.columns = df.columns.map(lambda x: lang_map.get(x.strip(), x.strip()))
    return df.reset_index().melt(id_vars="index", var_name="target", value_name=value_name).rename(columns={"index": "source"})

def load_token_overlap(path):
    with open(path) as f:
        data = json.load(f)["JSD"]
    df = pd.DataFrame.from_dict(data, orient="index", columns=["token_overlap_jsd_sym"]).reset_index()
    df[['source', 'target']] = df['index'].str.split("-", expand=True)
    df = df.drop(columns=["index"])
    df = df[~df['source'].str.contains("All") & ~df['target'].str.contains("All")]
    df["source"] = df["source"].map(LANG2_MAP)
    df["target"] = df["target"].map(LANG2_MAP)
    df_sym = df.rename(columns={"source": "target", "target": "source"})
    diag_langs = [LANG2_MAP[k] for k in LANG2_MAP]
    df_diag = pd.DataFrame({"source": diag_langs, "target": diag_langs, "token_overlap_jsd_sym": 0.0})
    return pd.concat([df, df_sym, df_diag], ignore_index=True)

def load_token_alignability(path):
    with open(path) as f:
        data = json.load(f)
    records = []
    for src, targets in data.items():
        for tgt, res in targets.items():
            records.append({
                "source": LANG2_MAP.get(src, src),
                "target": LANG2_MAP.get(tgt, tgt),
                "one_to_one_asym": res.get("one-to-one", None),
                "eflomal_sym": res.get("eflomal", None)
            })
    return pd.DataFrame(records)

def load_neuron_overlap(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.map(lambda x: LANG2_MAP.get(x.strip(), x.strip()))
    df.columns = df.columns.map(lambda x: LANG2_MAP.get(x.strip(), x.strip()))
    return df.reset_index().melt(id_vars="index", var_name="target", value_name="neuron_overlap_sym").rename(columns={"index": "source"})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--geography_dist_path", default="../linguistic_stats/results/linguistic_features/geo_distances.csv")
    parser.add_argument("--phylogeny_dist_path", default="../linguistic_stats/results/linguistic_features/fam_distances.csv")
    parser.add_argument("--syntax_dist_path", default="../linguistic_stats/results/linguistic_features/syntax_knn_distances.csv")
    parser.add_argument("--phonology_dist_path", default="../linguistic_stats/results/linguistic_features/phonology_knn_distances.csv")
    parser.add_argument("--inventory_dist_path", default="../linguistic_stats/results/linguistic_features/inventory_knn_distances.csv")
    parser.add_argument("--neuron_overlap_dir", default="../lang-specific_neurons/outputs/")
    parser.add_argument("--token_overlap_dir", default="../linguistic_stats/results/tokenization_analysis/overlap/")
    parser.add_argument("--token_alignability_dir", default="../linguistic_stats/results/tokenization_analysis/alignability/flores-dev/")
    args = parser.parse_args()

    model = args.model_name
    save_path = f"./datasets/per_pair_stats_{model}.csv"

    # Paths
    neuron_overlap_path = os.path.join(args.neuron_overlap_dir, f"overlap_matrix_{model}.csv")
    token_overlap_path = os.path.join(args.token_overlap_dir, model, "tokenizer_properties.json")
    token_alignability_path = os.path.join(args.token_alignability_dir, model, "align_scores.json")

    # Load all dataframes
    all_dfs = [
        load_and_melt(args.geography_dist_path, "geo_dist_sym"),
        load_and_melt(args.phylogeny_dist_path, "phylogeny_dist_sym"),
        load_and_melt(args.syntax_dist_path, "syntax_dist_sym"),
        load_and_melt(args.phonology_dist_path, "phonology_dist_sym"),
        load_and_melt(args.inventory_dist_path, "inventory_dist_sym"),
        load_token_overlap(token_overlap_path),
        load_token_alignability(token_alignability_path),
    ]

    neuron_df = load_neuron_overlap(neuron_overlap_path)
    if neuron_df is not None:
        all_dfs.append(neuron_df)
    
    # Merge all pairwise data
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=["source", "target"], how="outer"), all_dfs)
    merged_df.to_csv(save_path, index=False)