#!/bin/bash
#set -e
#set -u
#set -o pipefail
stage=3
stop_stage=4

input_dir=./data/arcs
out_dir=./exp/ar_gen_cs_bigram_cleaned
process=40
unit='bi'

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  ###steps for creating supervision, recording and bins ::
	python src/setup_recording_dict.py $input_dir/wav.scp $input_dir
	python src/setup_supervision_dict.py $input_dir/ctm.mono $input_dir/recording_dict.json $input_dir
	python src/setup_bins.py $input_dir/supervisions.json $input_dir/recording_dict.json $input_dir 10 #$input_dir

	if [ $unit='bi' ]; then
		python src/setup_bigram_sup_dict.py $input_dir/ctm.mono $input_dir/recording_dict.json $input_dir
		python src/setup_bins.py $input_dir/bigram_supervisions.json $input_dir/recording_dict.json $input_dir 2
		python src/get_vocab.py $input_dir/bigram_supervisions.json $input_dir/bigram_vocab.json 	
   		python src/get_vocab.py $input_dir/supervisions.json $input_dir/unigram_vocab.json
	fi
fi



if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  mkdir -p $input_dir/splits
rm -r $input_dir/splits/*
  split -n 1 -d $input_dir/text $input_dir/splits/
  echo "Done spliting"
fi

##Unigram generation
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  #array=( $( ls $input_dir/splits/. ) )
   #array= ( "02" "03" )
  for i in 02 03; do #"${array[@]}"; do "${array[@]}"; do
    echo "$i"
    sbatch sbatch_unigram_ar.sh $input_dir/splits/$i $out_dir $input_dir/ $process
  done
fi

##Bigram generation
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  #array=( $( ls $input_dir/splits/. ) )
  #array= ( "02" "03" )
  for i in 02 03; do #"${array[@]}"; do
    echo "$i"
    sbatch sbatch_bigram.sh $input_dir/splits/$i $out_dir $input_dir/ $process
  done
fi
##+ smoothing


