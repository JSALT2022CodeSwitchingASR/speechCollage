#!/bin/bash
#SBATCH -J V2audioGen
#SBATCH -p cpu-all
#SBATCH -c 40 #number of CPUs needed

module load gcc6 slurm cmake

#. /alt-asr/shchowdhury/tools/miniconda3/etc/profile.d/conda.sh && conda deactivate && conda activate /jsalt1/exp/wp2/audio_cs_aug/exp1/cs_generated_audio/tmp/dorsaenv_clone


inputlist=$1
outdir=$2
data=$3
proc=$4
#exp_suffix=$2
#outdir='/jsalt1/exp/wp2/audio_cs_aug/exp1/audio_data_generated_outdir/v2_seame_'$exp_suffix

mkdir -p $outdir

python3 src/generate_bigram_norm.py \
  --input $inputlist \
  --output $outdir \
  --data $data \
  --process $4



