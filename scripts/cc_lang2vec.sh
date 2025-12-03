cd src/linguistic_stats/feature_stats/
python cc-stats.py --input ../datasets/cc-stats/languages.csv --output ../results/cc-stats/cc_proportion.csv
python lang2vec_query.py --save_dir ../results/linguistic_features