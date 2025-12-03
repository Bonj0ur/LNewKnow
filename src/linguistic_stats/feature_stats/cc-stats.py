# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input file does not exist: {args.input}")

    df = pd.read_csv(args.input, usecols=["crawl", "primary_language", "%pages/crawl"])

    target_languages = [
        "eng", "jpn", "zho", "spa", "fra", "ita", "por", "kor", "swe",
        "dan", "tam", "mon", "cym", "swa", "zul", "tuk", "gla", "tha", "hin"
    ]

    df = df[df["primary_language"].isin(target_languages)]
    pivot_df = df.pivot(index="primary_language", columns="crawl", values="%pages/crawl")

    has_nan = pivot_df.isna().any().any()
    print("Contains NaN:", has_nan)

    mean_per_lang = pivot_df.mean(axis=1, skipna=True)
    std_per_lang = pivot_df.std(axis=1, skipna=True)

    stats_df = pd.DataFrame({
        "mean": mean_per_lang,
        "std": std_per_lang
    }).sort_values(by="mean", ascending=False)

    stats_df["mean ± std"] = stats_df.apply(
        lambda row: f"{row['mean']:.4f} ± {row['std']:.4f}", axis=1
    )

    formatted_output = stats_df[["mean ± std"]]
    print("\nLanguage statistics (mean ± std):")
    print(formatted_output)

    if args.output:
        formatted_output.to_csv(args.output)
        print(f"\nSaved to: {args.output}")

if __name__ == "__main__":
    main()