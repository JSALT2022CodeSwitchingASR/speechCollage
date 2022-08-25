#!/bin/bash
#set -e
#set -u
#set -o pipefail
stage=3
stop_stage=3

input_dir=./data/seame #change
out_dir=./exp/seame_gen_cs_unigram_new #change
process=40 #change
unit="uni" #or 'bi' 
norm=true #to do normalizing and scaling instead of search, false for searching with binning 
echo "$unit"="bi"
if $norm ; then 
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        ###steps for creating supervision, recording and bins ::
	python src/setup_recording_dict.py $input_dir/wav.scp $input_dir
	python src/setup_supervision_dict.py $input_dir/ctm.mono $input_dir/recording_dict.json $input_dir 
        echo "unigrams done"
	if [ "$unit" = "bi" ]; then
		python src/setup_bigram_sup_dict.py $input_dir/ctm.mono $input_dir/recording_dict.json $input_dir
	        echo "bigrams done"
        fi
    fi
else
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        ###steps for creating supervision, recording and bins ::
        python src/setup_recording_dict.py $input_dir/wav.scp $input_dir
        python src/setup_supervision_dict.py $input_dir/ctm.mono $input_dir/recording_dict.json $input_dir 
        python src/setup_bins.py $input_dir/supervisions.json $input_dir/recording_dict.json $input_dir 10
        echo "unigrams done, not norm"
        if [ "$unit" = "bi" ]; then
                python src/setup_bigram_sup_dict.py $input_dir/ctm.mono $input_dir/recording_dict.json $input_dir
                python src/get_vocab.py $input_dir/supervisions.json $input_dir/unigram_vocab.json 
                python src/get_vocab.py $input_dir/bigram_supervisions.json $input_dir/bigram_vocab.json 
                python src/setup_bins.py $input_dir/bigram_supervisions.json $input_dir/recording_dict.json $input_dir 2 
                echo "bigrams done, not norm"
        fi
    fi
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  mkdir -p $input_dir/splits
  split -n 10 -d $input_dir/text $input_dir/splits/
fi
echo "splits done"
##generation
if [ "$unit" = "uni" ]; then
   if $norm ; then
       if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
           array=( $( ls $input_dir/splits/. ) )

           for i in "${array[@]}"; do
               echo "$i"
               sbatch sbatch_unigram_norm.sh $input_dir/splits/$i $input_dir  $out_dir $process
           done
           echo "generation done uni norm"
       fi
   else 
       if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
           array=( $( ls $input_dir/splits/. ) )

           for i in "${array[@]}"; do
               echo "$i"
               sbatch sbatch_unigram.sh $input_dir/splits/$i $input_dir  $out_dir $process
           done
           echo "generation done uni bins"
       fi
   fi
else
   if $norm ; then
       if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
           array=( $( ls $input_dir/splits/. ) )
       
           for i in "${array[@]}"; do
               echo "$i"
               sbatch sbatch_bigram_norm.sh $input_dir/splits/$i $input_dir  $out_dir $process
           done
           echo "generation done bi norm"
       fi
   else
       if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
           array=( $( ls $input_dir/splits/. ) )

           for i in "${array[@]}"; do
               echo "$i"
               sbatch sbatch_bigram.sh $input_dir/splits/$i $input_dir  $out_dir $process
           done
           echo "generation done bi bins"
       fi
   fi
fi
##+ smoothing
