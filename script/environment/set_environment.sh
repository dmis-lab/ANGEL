# Check virtual environment
POS_ENV="positive_only"
NEG_ENV="negative_aware"

if conda info --envs | grep -q "^$POS_ENV "; then
    echo "The Conda environment '$POS_ENV' already exists."
else
    echo "The Conda environment '$POS_ENV' does not exist. Creating it now..."
    conda env create -f script/environment/environment_pos.yml
fi

if conda info --envs | grep -q "^$NEG_ENV "; then
    echo "The Conda environment '$NEG_ENV' already exists."
else
    echo "The Conda environment '$NEG_ENV' does not exist. Creating it now..."
    conda env create -f script/environment/environment_neg.yml
fi