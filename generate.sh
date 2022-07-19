#!/bin/bash
#set -e
#set -u
#set -o pipefail
stage=3
stop_stage=3

input_dir=./data/seame
out_dir=./exp/seame_gen_cs_unigram
process=40

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  ###steps for creating supervision, recording and bins ::
	python src/setup_recording_dict.py $input_dir/wav.scp ./data/ #$input_dir
	python src/setup_supervision_dict.py $input_dir/ctm.mono ./data/recording_dict.json ./data/ #$input_dir
	python src/setup_bins.py ./data/supervisions.json ./data/recording_dict.json ./data/ #$input_dir
fi



if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  mkdir -p $input_dir/splits
  split -n 10 -d $input_dir/text $input_dir/splits
fi
##Unigram generation
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  array=( $( ls $input_dir/splits/. ) )

  for i in "${array[@]}"; do
    echo "$i"
    sbatch sbatch_unigram_seame.sh $input_dir/splits/$i $out_dir $process
  done
fi
##+ smoothing


