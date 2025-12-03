cd src/linguistic_stats/tokenization_analysis/
bash scripts/prep_flores_tok_files.sh
python download_non_en_pairs_MultiCCAligned.py
bash scripts/prep_multi-cc_tok_files.sh
bash scripts/prep_opus-100_tok_files.sh
git clone https://github.com/robertostling/eflomal.git
cd eflomal
python -m pip install .
git clone https://github.com/clab/fast_align.git
mkdir fast_align/build
cd fast_align/build
cmake .. && make
bash scripts/align_prior_opus-100.sh
bash scripts/align_prior_multi-cc.sh
bash scripts/align_prior_flores.sh
bash scripts/rename_file.sh
bash scripts/alignability_calc.sh
bash scripts/overlap_calc.sh
bash scripts/efficiency_calc.sh
bash scripts/morphscore_calc.sh