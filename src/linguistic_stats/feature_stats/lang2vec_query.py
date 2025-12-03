# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import argparse
import numpy as np
import pandas as pd
import lang2vec.lang2vec as l2v

def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    return u.dot(v)

def get_distance_matrix(langs_list, dist_type):
    print(f"Computing {dist_type} distances for {len(langs_list)} languages...")
    vecs_dict = l2v.get_features(langs_list, dist_type, minimal=True)
    vecs = {lang: np.array(vecs_dict[lang]) for lang in langs_list}
    distance_matrix = pd.DataFrame(index=langs_list, columns=langs_list, dtype=float)

    for lang1 in langs_list:
        for lang2 in langs_list:
            if lang1 == lang2:
                distance_matrix.loc[lang1, lang2] = 0.0
            else:
                sim = cosine_similarity(vecs[lang1], vecs[lang2])
                dist = 1 - sim
                distance_matrix.loc[lang1, lang2] = dist
    
    return distance_matrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--dist_types", nargs="+", default=["syntax_knn", "phonology_knn", "inventory_knn", "fam", "geo"])
    parser.add_argument("--langs_list", nargs="+", default=["eng", "zho", "jpn", "fra", "spa", "ita", "por", "swe", "kor", "dan", "tha", "hin", "tam", "mon", "cym", "swa", "tuk", "gla", "zul"])
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    for dist_type in args.dist_types:
        df = get_distance_matrix(args.langs_list, dist_type)
        out_path = os.path.join(args.save_dir, f"{dist_type}_distances.csv")
        df.to_csv(out_path)
        print(f"Saved {dist_type} distance matrix to {out_path}")