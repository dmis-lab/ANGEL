# activate virtual environment
source /usr/miniconda3/etc/profile.d/conda.sh
conda init 
conda activate positive_only  

# download umls vectorizor
wget -O ./umls_tfidf_vectorizer.joblib https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/linkers/2023-04-23/umls/tfidf_vectorizer.joblib


# download datasets from GenBioEL (NCBI-disease BC5CDR COMETA AskAPatient)
gdown "https://drive.google.com/uc?id=1JWYMdwxp7_ZZRGAO-ENmgUNirx9-nX32" -O processed_data.zip
unzip processed_data.zip
rm processed_data.zip
rm -rf ./__MACOSX

# download processed medmentions datasets
# If you want to download original medmentions (2017 ver) use below
gdown "https://drive.google.com/uc?id=1UXcPwwWaGUxCvyL7b4orbUVqbboKhGro" -O mm.zip     # 2020 ver
# gdown "https://drive.google.com/uc?id=1IrMlO4yh7d4tGO4BgThNdKtg_1opEXTF" -O mm.zip   # 2017 ver
unzip mm.zip -d ./benchmarks
rm mm.zip

echo "Download Dataset Complete!"

# make synonym-aware dataset and trie
DATASET=(ncbi cometa bc5cdr aap mm)

for dataset in "${DATASET[@]}"; do
    python ./trie/create_trie_and_target_kb.py \
    -dataset $dataset \
    -base_dir ./benchmarks

    python make_syn.py \
    -dataset $dataset \
    -num_k 3 
done
