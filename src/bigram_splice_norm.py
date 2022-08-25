import json
from itertools import groupby
from operator import itemgetter
import re
import random 
import torchaudio 
from torchaudio import * 
from lhotse import *
import torch 
import os.path 
import numpy as np 
from langdetect import detect
import sys 
from datetime import datetime
random.seed(50)

def load_dicts(rec_dict_path,unigram_sup_path,bigram_sup_path):

    recordings=json.load(open(rec_dict_path))
    print('here1')
    uni_sups = json.load(open(unigram_sup_path))
    bi_sups=json.load(open(bigram_sup_path))
    energies = []
    for rec in recordings.keys():
        _, energy = recordings[rec]
        energies.append(energy)
    mean=np.mean(energies)
    std=np.std(energies)
    #print(mean,std)
    return uni_sups,bi_sups, recordings,mean,std

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
def isEnglishWord(word):
    # we use wordnet as well as enchant as wordnet check fails for contractions
    # and enchant check fails for british spelling
    # chinese character check 
    return re.sub(r'[\u4e00-\u9fff]+', '', word)==word and re.sub(r'[\u0600-\u06FF\s]+', '', word)==word
def find_boundaries(line):
    ranges={}
    ranges['en']=[]
    ranges['oth']=[]
    en_indices=[]
    oth_indices=[]
    for i in range(len(line)):
        word =line[i]
        if(isEnglishWord(word)):
            en_indices.append(i)
        else:
            oth_indices.append(i)

    for k, g in groupby(enumerate(en_indices), lambda ix : ix[0] - ix[1]):
        rangs=list(map(itemgetter(1), g))
        r=(rangs[0],rangs[-1])
        ranges['en'].append(r)
    for k, g in groupby(enumerate(oth_indices), lambda ix : ix[0] - ix[1]):
        rangs=list(map(itemgetter(1), g))
        r=(rangs[0],rangs[-1])
        ranges['oth'].append(r)
    r=ranges['en'] + ranges['oth']
    r.sort(key=lambda ix:ix[0])
    #print(r)
    return r

def create_segments(ranges, line,uni_sups,bi_sups):
    segments=[]
    for (b,e) in ranges:
        seg=line[b:e+1]
        #print(seg)
        if(len(seg) ==1):
            segments.append(seg)
        elif(len(seg)==2):
            if(' '.join(seg) in bi_sups):
                #if(random.randint(0,1)==1):
                    #segments.append([seg[0]])
                    #segments.append([seg[1]])
                #else:
                segments.append(seg)
            else:
                segments.append([seg[0]])
                segments.append([seg[1]])

        elif(len(seg)>=3):
            #length=len(seg)
            i=0
            while(i<len(seg)):
                ngram=random.randint(1,min(len(seg)-i,2))
                if(ngram==1):
                    sub_seg=seg[i:i+1]
                    #seg=seg[i+1:]
                    segments.append(sub_seg)
                    i+=1
                if(ngram==2):
                    sub_seg=seg[i:i+2]
                    #seg=seg[i+2:]
                    if(' '.join(sub_seg) in bi_sups):
                        segments.append(sub_seg)
                    else:
                        segments.append([sub_seg[0]])
                        segments.append([sub_seg[1]])
                    i+=2
    return segments

def create_cs_audio(generated_text,output_directory_path,recordings,unigram_supervisions, bigram_supervisions,mean,std):
    length=len(generated_text)
    transcripts=[]
    alignments={}
    for i in range(length):
        line=generated_text[i].split()
        filename=line[0]
        ranges=find_boundaries(line[1:])
        #print(line[1:])
        segments=create_segments(ranges,line[1:],unigram_supervisions,bigram_supervisions)
        print(segments)
        start_time=datetime.now()
        transcript=filename+' '
        cut=None
        alignment=[]
        energy_to_match=0.0
        for j in range(len(segments)):
            seg=segments[j]
            l=len(seg)
            seg=' '.join(seg)
            token=seg
            #print(seg)
            if(l==1 and seg in unigram_supervisions):
                if(not cut):
                    c,e=take_random(token,unigram_supervisions,recordings,energy_to_match,True,mean,std)
                    cut=c
                    alignment.append((token,cut.recording.id, cut.start,cut.duration))
                    energy_to_match=e  
                else:
                    c,e=take_random(token,unigram_supervisions,recordings,energy_to_match,False,mean,std)
                    cut=cut.append(c)
                    alignment.append((token,c.recording.id,c.start,c.duration)) 
                transcript+=(seg+ ' ')
            if(l==2 and seg in bigram_supervisions):
                if(not cut):
                   c,e=take_random(token,bigram_supervisions,recordings,energy_to_match,True,mean,std)
                   cut=c
                   alignment.append((token,cut.recording.id, cut.start,cut.duration))
                   energy_to_match=e
                else:
                   c,e=take_random(token,bigram_supervisions,recordings,energy_to_match,False,mean,std)
                   cut=cut.append(c)
                   alignment.append((token,c.recording.id,c.start,c.duration))
                transcript+=(seg+ ' ')
        end_time=datetime.now()
        delta = (end_time-start_time)
        print('making sentence time: ', delta)
        start_time=datetime.now()
        if(cut):
            #cut.save_audio(output_directory_path+'/bi_'+filename+'.wav')	
            transcripts.append(transcript.strip())
            alignments[filename]=alignment
            torchaudio.save(output_directory_path+'/bi_'+filename+'.wav',torch.from_numpy(cut.load_audio()),sample_rate=16000, encoding="PCM_S", bits_per_sample=16)
        end_time=datetime.now()
        delta = (end_time-start_time)

        print('saving audio time: ', delta)

    with open(output_directory_path+'/bi_transcripts.txt','w') as f:
        for t in transcripts:
            f.write(t+'\n')
    with open(output_directory_path+'/alignments.json', 'w') as f:
        json.dump(alignments,f)
'''if __name__ == "__main__":
    rec_dict_path='/jsalt1/exp/wp2/audio_cs_aug/exp1/speech_gen_wp/data/recording_dict.json'
    unigram_sups_path='/jsalt1/exp/wp2/audio_cs_aug/exp1/speech_gen_wp/data/unigram_vocab.json'
    unigram_bins_path='/jsalt1/exp/wp2/audio_cs_aug/exp1/speech_gen_wp/data/unigram_bins.json'
    bigram_sups_path='/jsalt1/exp/wp2/audio_cs_aug/exp1/speech_gen_wp/data/bigram_vocab.json'
    bigram_bins_path='/jsalt1/exp/wp2/audio_cs_aug/exp1/speech_gen_wp/data/bigram_bins.json'
    
    recordings,uni_sups,uni_bins,bi_sups,bi_bins,percents=load_dicts(rec_dict_path,unigram_sups_path,unigram_bins_path,bigram_sups_path,bigram_bins_path)
    print('here')
  
    output_directory_path='.'
    generated_text=open('/jsalt1/exp/wp2/audio_cs_aug/exp1/speech_gen_wp/dummy_2.txt','r').readlines()[0:1]
    create_cs_audio(generated_text,output_directory_path,recordings,uni_sups,uni_bins,bi_sups,bi_bins,percents)'''

if __name__=="__main__":
    rec_dict_path=sys.argv[1]
    unigram_sup_path=sys.argv[2]
    bigram_sup_path=sys.argv[3]
     
    #recordings,uni_sups,uni_bins,bi_sups,bi_bins,percents=load_dicts(rec_dict_path,unigram_v_path,unigram_bins_path,bigram_v_path,bigram_bins_path) 
    uni_sups,bi_sups, recordings,mean,std=load_dicts(rec_dict_path, unigram_sup_path, bigram_sup_path)
    output_directory_path='.'
    generated_text=open('/jsalt1/exp/wp2/audio_cs_aug/exp1/speech_gen_wp/dummy_2.txt','r').readlines()[0:1]
    create_cs_audio(generated_text,output_directory_path,recordings,uni_sups, bi_sups,mean,std)
