#!/bin/bash
# Please consult the README.md file for instructions on how to run the benchmark.
source $HOME/kmeans_env/bin/activate
tokenizer_name=$1
shift
script_args="$@"

scp $HOME/projects/def-ravanelm/datasets/genres.tar.gz $SLURM_TMPDIR
cd  $SLURM_TMPDIR
tar -xzvf genres.tar.gz

cd $HOME/sb_dasb_music/
python benchmarks/DASB/GTZAN/linear/train_${tokenizer_name}.py  benchmarks/DASB/GTZAN/linear/hparams/train_${tokenizer_name}.yaml  --data_folder=$SLURM_TMPDIR/data/ $@   