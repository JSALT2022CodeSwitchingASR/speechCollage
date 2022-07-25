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

def load_dicts(rec_dict_path,unigram_v_path,unigram_bins_path,bigram_v_path,bigram_bins_path):

    recs=json.load(open(rec_dict_path))
    print('here1')
    uni_v=set(json.load(open(unigram_v_path)))
    print('here2')
    uni_bins=json.load(open(unigram_bins_path))
    print('here3')
    bi_v=set(json.load(open(bigram_v_path)))
    print('here4')
    bi_bins=json.load(open(bigram_bins_path))
    energies=[]
    for rec in recs.keys():
            _,energy=recs[rec]
            energies.append(energy)

    percentiles=[0.0,25.0,50.0,75.0,100.0]

    np_arr=np.array(energies)
    percents=np.percentile(np_arr,percentiles)

    return recs,uni_v,uni_bins,bi_v,bi_bins,percents
def find_bin(energy,percentiles):

    if(energy>=percentiles[0] and energy<percentiles[1]):
        return 1
    elif(energy>=percentiles[1] and energy < percentiles[2]):
        return 2

    elif(energy >=percentiles[2] and energy < percentiles[3]):
        return 3
    elif(energy >= percentiles[3] and energy<percentiles[4]):
        return 4
    else:
        return 5
def take_random(token,bin,recordings):
     matched_sups = bin[token]
     sup = random.sample(matched_sups, 1)[0]
     energy_to_match = recordings[sup[1]][1]
     recording = Recording.from_file(recordings[sup[1]][0])
     sup = SupervisionSegment(id=sup[0], recording_id=sup[1], start=sup[2], duration=sup[3], channel=0,
                                     text=sup[0])
     c = MonoCut(id=sup.id, start=sup.start, duration=sup.duration, channel=sup.channel, recording=recording,
                        supervisions=[sup])
     return c, energy_to_match

def take_first(token,bin,recordings):
     matched_sups = bin[token]
     sup = matched_sups[0]
     energy_to_match = recordings[sup[1]][1]
     recording = Recording.from_file(recordings[sup[1]][0])
     sup = SupervisionSegment(id=sup[0], recording_id=sup[1], start=sup[2], duration=sup[3], channel=0,
                                     text=sup[0])
     c = MonoCut(id=sup.id, start=sup.start, duration=sup.duration, channel=sup.channel, recording=recording,
                        supervisions=[sup])
     return c, energy_to_match

def take_last(token,bin,recordings):
     matched_sups = bin[token]
     sup = matched_sups[-1]
     energy_to_match = recordings[sup[1]][1]
     recording = Recording.from_file(recordings[sup[1]][0])
     sup = SupervisionSegment(id=sup[0], recording_id=sup[1], start=sup[2], duration=sup[3], channel=0,
                                     text=sup[0])
     c = MonoCut(id=sup.id, start=sup.start, duration=sup.duration, channel=sup.channel, recording=recording,
                        supervisions=[sup])
     return c, energy_to_match

def find_token(token, b, bins, recordings):
   offsets=[0,1,-1,2,-2,3,-3,4,-4,5,-5]
   for offset in offsets:
      bin_id='bin'+str(b+offset)
      if(bin_id in bins and token in bins[bin_id] and len(bins[bin_id][token])>=1):
          if(offset<0):
              return take_last(token,bins[bin_id],recordings)
          elif(offset>0):
              return take_first(token,bins[bin_id],recordings)
          else:
              return take_random(token,bins[bin_id],recordings)

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

def create_segments(ranges, line,uni_v,bi_v):
    segments=[]
    for (b,e) in ranges:
        seg=line[b:e+1]
        #print(seg)
        if(len(seg) ==1):
            segments.append(seg)
        elif(len(seg)==2):
            if(' '.join(seg) in bi_v):
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
                    if(' '.join(sub_seg) in bi_v):
                        segments.append(sub_seg)
                    else:
                        segments.append([sub_seg[0]])
                        segments.append([sub_seg[1]])
                    i+=2
    return segments

def create_cs_audio(generated_text,output_directory_path,recordings,uni_v,uni_bins,bi_v,bi_bins,percents):
    length=len(generated_text)
    transcripts=[]
    alignments={}
    for i in range(length):
        line=generated_text[i].split()
        filename=line[0]
        ranges=find_boundaries(line[1:])
        #print(line[1:])
        segments=create_segments(ranges,line[1:],uni_v,bi_v)
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
            if(l==1 and seg in uni_v):
                if(not cut):
                    if(seg in uni_bins['low_freq']):
                        #print('here1')
                        c,e=take_random(seg,uni_bins['low_freq'],recordings)
                        cut=c
                        #print(seg,c.recording.id, cut.start,cut.duration)
                        alignment.append((token,c.recording.id, c.start,c.duration))
                        energy_to_match=e 
                    else:
                        #print('here2')
                        random_bin=random.randint(1,5)
                        c,e=find_token(seg,random_bin,uni_bins,recordings)
                        cut=c
                        alignment.append((token,c.recording.id, c.start,c.duration))
                        #print(seg,c.recording.id, cut.start,cut.duration)
                        energy_to_match=e
                else:
                    if(seg in uni_bins['low_freq']):
                        c,e=take_random(seg,uni_bins['low_freq'],recordings)
                        energy_to_match=e
                        alignment.append((token,c.recording.id, c.start,c.duration))
                        #print(seg,c.recording.id, c.start,c.duration)
                        cut=cut.append(c)
                    else:
                        b=find_bin(energy_to_match,percents)
                        c,e=find_token(seg,b,uni_bins,recordings)
                        alignment.append((token,c.recording.id, c.start,c.duration))
                        cut=cut.append(c)
                        #print(seg,c.recording.id, c.start,c.duration)
                        energy_to_match=e
                transcript+=(seg+ ' ')
            if(l==2 and seg in bi_v):
                if(not cut):
                    #print('here3')
                    if(seg in bi_bins['low_freq']):
                        c,e=take_random(seg,bi_bins['low_freq'],recordings)
                        cut=c
                        alignment.append((token,c.recording.id, c.start,c.duration))
                        #print(seg,c.recording.id, c.start,c.duration)
                        energy_to_match=e  
                    else:
                        random_bin=random.randint(1,5)
                        c,e=find_token(seg,random_bin,bi_bins,recordings)
                        cut=c
                        alignment.append((token,c.recording.id, c.start,c.duration))
                        #print(seg,c.recording.id, cut.start,cut.duration)
                        energy_to_match=e
                else:
                    #print('here4')
                    if(seg in bi_bins['low_freq']):
                        c,e=take_random(seg,bi_bins['low_freq'],recordings)
                        energy_to_match=e
                        alignment.append((token,c.recording.id, c.start,c.duration))
                        print(seg,c.recording.id, c.start,c.duration)
                        cut=cut.append(c)
                    else:
                        b=find_bin(energy_to_match,percents)
                        c,e=find_token(seg,b,bi_bins,recordings)
                        cut=cut.append(c)
                        alignment.append((token,c.recording.id, c.start,c.duration))
                        print(seg,c.recording.id, c.start,c.duration)
                        energy_to_match=e


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
    unigram_v_path=sys.argv[2]
    unigram_bins_path=sys.argv[3]
    bigram_v_path=sys.argv[4]
    bigram_bins_path=sys.argv[5]
     
    recordings,uni_sups,uni_bins,bi_sups,bi_bins,percents=load_dicts(rec_dict_path,unigram_v_path,unigram_bins_path,bigram_v_path,bigram_bins_path) 

    output_directory_path='.'
    generated_text=open('/jsalt1/exp/wp2/audio_cs_aug/exp1/speech_gen_wp/dummy_2.txt','r').readlines()[0:1]
    create_cs_audio(generated_text,output_directory_path,recordings,uni_sups,uni_bins,bi_sups,bi_bins,percents)
