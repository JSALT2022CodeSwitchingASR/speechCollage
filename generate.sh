#!/bin/bash
#set -e
#set -u
#set -o pipefail
. ./cmd.sh
. ./path.sh
stage=3
stop_stage=3

cs_text=data/seame/text
mono_input_dir=./data/seame2
odir=./exp/seame_gen_cs_unigram
process=5
nj=40

. ./utils/parse_options.sh

mkdir -p $odir/log
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  ###steps for creating supervision, recording and bins ::
	python src/setup_recording_dict.py $mono_input_dir/wav.scp ./data/ #$input_dir
	python src/setup_supervision_dict.py $mono_input_dir/ctm.mono ./data/recording_dict.json ./data/ #$input_dir
	python src/setup_bins.py ./data/supervisions.json ./data/recording_dict.json ./data/ #$input_dir
fi

##Unigram generation
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  for i in `seq 1 ${nj}`; do
    ${train_cmd} ${odir}/log/generate.${i}.log \
      ./utils/split_scp.pl -j ${nj} $((${i} - 1)) ${cs_text} \| \
      sbatch_unigram_seame.sh /dev/stdin $odir $process &
  done
fi
##+ smoothing


