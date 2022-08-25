#!/usr/bin/env python3
# Copyright (c) Dorsa Z

# Apache 2.0


import json
import random
from lhotse import *
import torchaudio 
from torchaudio import * 
import torch 
import os.path
import numpy as np
import sys
from datetime import datetime

random.seed(10)

# %Modified by shammur
def load_dicts_modified(sup_dict_path, rec_dict_path):
    supervisions = json.load(open(sup_dict_path))
    recordings = json.load(open(rec_dict_path))
    energies = []
    for rec in recordings.keys():
        _, energy = recordings[rec]
        energies.append(energy)
    mean=np.mean(energies)
    std=np.std(energies)
    #print(mean,std)
    return supervisions, recordings,mean,std

# Todo: fix! make modular
def take_random(token,sups,recordings,energy_to_match, first,mean,std):
     matched_sups = sups[token]
     sup = random.sample(matched_sups, 1)[0]
     #print(token)
     energy = recordings[sup[1]][1]
     if(not first):
         #print(energy)
         
         #print(float(energy_to_match/(np.sqrt(energy))))
         recording = Recording.from_file(recordings[sup[1]][0]).perturb_volume(factor=float(energy_to_match/np.sqrt(energy)))
         #print(np.sqrt(audio.audio_energy(recording.load_audio())))
         sup = SupervisionSegment(id=sup[0], recording_id=sup[1], start=sup[2], duration=sup[3], channel=0,
                                     text=sup[0])
         c = MonoCut(id=sup.id, start=sup.start, duration=sup.duration, channel=sup.channel, recording=recording,
                        supervisions=[sup])
         return c, energy_to_match
     else:
        
         rand_en=np.random.normal(loc=mean, scale=std)
         while(rand_en<0.01):
             rand_en=np.random.normal(loc=mean, scale=std)
         #print(rand_en)
         recording=Recording.from_file(recordings[sup[1]][0]).perturb_volume(factor=float(rand_en/np.sqrt(energy)))
         sup = SupervisionSegment(id=sup[0], recording_id=sup[1], start=sup[2], duration=sup[3], channel=0,
                                     text=sup[0])
         c = MonoCut(id=sup.id, start=sup.start, duration=sup.duration, channel=sup.channel, recording=recording,
                        supervisions=[sup])
         return c,rand_en
             
def create_cs_audio(generated_text, output_directory_path, supervisions, recordings,mean,std): 
    length = len(generated_text)
    transcripts=[]
    alignments={}
    for i in range(length):
        line = generated_text[i].split()
        file_name = line[0]

        start_time = datetime.now()
        transcript=file_name + ' '
        alignment=[]
        sentence_token = line[1:]
        cut = None
        energy_to_match = 0.0
        for j in range(len(sentence_token)):
            token = sentence_token[j]
            print(token)
            if (token in supervisions):
                #print('here')
                transcript += (token+ ' ')
                if not cut: 
                    c,e=take_random(token,supervisions,recordings,energy_to_match,True,mean,std)
                    cut=c
                    alignment.append((token,cut.recording.id, cut.start,cut.duration))
                    energy_to_match=e			
                else:
                    c,e=take_random(token,supervisions,recordings,energy_to_match,False,mean,std)
                    cut=cut.append(c)
                    alignment.append((token,c.recording.id,c.start,c.duration))


                 

        end_time = datetime.now()
        delta = (end_time - start_time)
        print('making sentence time: ', delta)

        start_time = datetime.now()
        if(cut):
            #cut.save_audio(output_directory_path + '/' + file_name + '.wav')
            transcripts.append(transcript.strip())
            alignments[file_name]=alignment
            torchaudio.save(output_directory_path+'/'+file_name+'.wav', torch.from_numpy(cut.load_audio()),sample_rate=16000, encoding="PCM_S", bits_per_sample=16)
        end_time = datetime.now()
        delta = (end_time - start_time)

        print('saving audio time: ', delta)

    with open(output_directory_path+'/transcripts.txt','w') as f:
        for t in transcripts:
            f.write(t+'\n')
    with open(output_directory_path+'/alignments.json', 'w') as f: 
        json.dump(alignments,f) 


if __name__ == "__main__":
    sup_dict_path = sys.argv[1]
    rec_dict_path = sys.argv[2]
    #bins_dict_path = sys.argv[3]
    # non_freq_dict_path = sys.argv[3]i
    # sup_bin_1_dict_path = sys.argv[4]
    # sup_bin_2_dict_path = sys.argv[5]
    # sup_bin_3_dict_path = sys.argv[6]
    # sup_bin_4_dict_path = sys.argv[7]
    # sup_bin_5_dict_path = sys.argv[8]

    input_path = sys.argv[3]
    output_path = sys.argv[4]

    supervisions, recordings,mean,std= load_dicts_modified(
        sup_dict_path, rec_dict_path)
        # non_freq_dict_path, sup_bin_1_dict_path, sup_bin_2_dict_path, sup_bin_3_dict_path,
        # sup_bin_4_dict_path, sup_bin_5_dict_path)
    generated_text = open(input_path, 'r').readlines()
    create_cs_audio(generated_text, output_path, supervisions, recordings,mean,std)
