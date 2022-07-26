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
import codecs
from datetime import datetime

random.seed(10)

# %Modified by shammur
def load_dicts_modified(sup_dict_path, rec_dict_path, bin_dict_path):
    supervisions = json.load(open(sup_dict_path))
    recordings = json.load(open(rec_dict_path))
    bins = json.load(open(bin_dict_path))

    #non_freq_sups, sups_bin_1, sups_bin_2, sups_bin_3, sups_bin_4, sups_bin_5= bins['low_freq'], bins['bin1'], bins['bin2'], bins['bin3'], bins['bin4'], bins['bin5']

    energies = []
    for rec in recordings.keys():
        _, energy = recordings[rec]
        energies.append(energy)

    percentiles = [0.0, 25.0, 50.0, 75.0, 100.0]

    np_arr = np.array(energies)
    percents = np.percentile(np_arr, percentiles)
    return supervisions, recordings, bins, percents

# Original load dicts func
def load_dicts(sup_dict_path, rec_dict_path, non_freq_dict_path, sup_bin_1_dict_path, sup_bin_2_dict_path,
               sup_bin_3_dict_path, sup_bin_4_dict_path, sup_bin_5_dict_path):
    supervisions = json.load(open(sup_dict_path))
    recordings = json.load(open(rec_dict_path))
    non_freq_sups = json.load(open(non_freq_dict_path))
    sups_bin_1 = json.load(open(sup_bin_1_dict_path))
    sups_bin_2 = json.load(open(sup_bin_2_dict_path))
    sups_bin_3 = json.load(open(sup_bin_3_dict_path))
    sups_bin_4 = json.load(open(sup_bin_4_dict_path))
    sups_bin_5 = json.load(open(sup_bin_5_dict_path))

    energies = []
    for rec in recordings.keys():
        _, energy = recordings[rec]
        energies.append(energy)

    percentiles = [0.0, 25.0, 50.0, 75.0, 100.0]

    np_arr = np.array(energies)
    percents = np.percentile(np_arr, percentiles)
    return supervisions, recordings, non_freq_sups, sups_bin_1, sups_bin_2, sups_bin_3, sups_bin_4, sups_bin_5, percents


def find_bin(energy, percentiles):
    if (energy >= percentiles[0] and energy < percentiles[1]):
        return 1
    elif (energy >= percentiles[1] and energy < percentiles[2]):
        return 2

    elif (energy >= percentiles[2] and energy < percentiles[3]):
        return 3
    elif (energy >= percentiles[3] and energy < percentiles[4]):
        return 4
    else:
        return 5

# Todo: fix! make modular

def find_token(token, b, sups_bin_1, sups_bin_2, sups_bin_3, sups_bin_4, sups_bin_5, recordings):
    if (b == 1):
        if (token in sups_bin_1 and len(sups_bin_1[token])>=1):
            matched_sups = sups_bin_1[token]
            sup = random.sample(matched_sups, 1)[0]
            energy_to_match = recordings[sup[1]][1]
            recording = recordings[sup[1]][0]
            sup = SupervisionSegment(id=sup[0], recording_id=sup[1], start=sup[2], duration=sup[3], channel=0,
                                     text=sup[0])
            c = MonoCut(id=sup.id, start=sup.start, duration=sup.duration, channel=sup.channel, recording=recording,
                        supervisions=[sup])
            return c, energy_to_match
        elif (token in sups_bin_2 and len(sups_bin_2[token])>=1):
            matched_sups = sups_bin_2[token]
            sup = matched_sups[0]
            energy_to_match = recordings[sup[1]][1]
            recording = recordings[sup[1]][0]
            sup = SupervisionSegment(id=sup[0], recording_id=sup[1], start=sup[2], duration=sup[3], channel=0,
                                     text=sup[0])
            c = MonoCut(id=sup.id, start=sup.start, duration=sup.duration, channel=sup.channel, recording=recording,
                        supervisions=[sup])
            return c, energy_to_match
        elif (token in sups_bin_3 and len(sups_bin_3[token])>=1):
            matched_sups = sups_bin_3[token]
            sup = matched_sups[0]
            energy_to_match = recordings[sup[1]][1]
            recording = recordings[sup[1]][0]
            sup = SupervisionSegment(id=sup[0], recording_id=sup[1], start=sup[2], duration=sup[3], channel=0,
                                     text=sup[0])
            c = MonoCut(id=sup.id, start=sup.start, duration=sup.duration, channel=sup.channel, recording=recording,
                        supervisions=[sup])
            return c, energy_to_match
        elif (token in sups_bin_4 and len(sups_bin_4[token])>=1):
            matched_sups = sups_bin_4[token]
            sup = matched_sups[0]
            energy_to_match = recordings[sup[1]][1]
            recording = recordings[sup[1]][0]
            sup = SupervisionSegment(id=sup[0], recording_id=sup[1], start=sup[2], duration=sup[3], channel=0,
                                     text=sup[0])
            c = MonoCut(id=sup.id, start=sup.start, duration=sup.duration, channel=sup.channel, recording=recording,
                        supervisions=[sup])
            return c, energy_to_match

        elif (token in sups_bin_5 and  len(sups_bin_5[token])>=1):
            matched_sups = sups_bin_5[token]
            sup = matched_sups[0]
            energy_to_match = recordings[sup[1]][1]
            recording = recordings[sup[1]][0]
            sup = SupervisionSegment(id=sup[0], recording_id=sup[1], start=sup[2], duration=sup[3], channel=0,
                                     text=sup[0])
            c = MonoCut(id=sup.id, start=sup.start, duration=sup.duration, channel=sup.channel, recording=recording,
                        supervisions=[sup])
            return c, energy_to_match

    if (b == 2):
        if (token in sups_bin_2 and len(sups_bin_2[token])>=1):
            matched_sups = sups_bin_2[token]
            sup = random.sample(matched_sups, 1)[0]
            energy_to_match = recordings[sup[1]][1]
            recording = recordings[sup[1]][0]
            sup = SupervisionSegment(id=sup[0], recording_id=sup[1], start=sup[2], duration=sup[3], channel=0,
                                     text=sup[0])
            c = MonoCut(id=sup.id, start=sup.start, duration=sup.duration, channel=sup.channel, recording=recording,
                        supervisions=[sup])
            return c, energy_to_match
        elif (token in sups_bin_1 and len(sups_bin_1[token])>=1):
            matched_sups = sups_bin_1[token]
            sup = matched_sups[-1]
            energy_to_match = recordings[sup[1]][1]
            recording = recordings[sup[1]][0]
            sup = SupervisionSegment(id=sup[0], recording_id=sup[1], start=sup[2], duration=sup[3], channel=0,
                                     text=sup[0])
            c = MonoCut(id=sup.id, start=sup.start, duration=sup.duration, channel=sup.channel, recording=recording,
                        supervisions=[sup])
            return c, energy_to_match
        elif (token in sups_bin_3 and len(sups_bin_3[token])>=1):
            matched_sups = sups_bin_3[token]
            sup = matched_sups[0]
            energy_to_match = recordings[sup[1]][1]
            recording = recordings[sup[1]][0]
            sup = SupervisionSegment(id=sup[0], recording_id=sup[1], start=sup[2], duration=sup[3], channel=0,
                                     text=sup[0])
            c = MonoCut(id=sup.id, start=sup.start, duration=sup.duration, channel=sup.channel, recording=recording,
                        supervisions=[sup])
            return c, energy_to_match
        elif (token in sups_bin_4 and len(sups_bin_4[token])>=1):
            matched_sups = sups_bin_4[token]
            sup = matched_sups[0]
            energy_to_match = recordings[sup[1]][1]
            recording = recordings[sup[1]][0]
            sup = SupervisionSegment(id=sup[0], recording_id=sup[1], start=sup[2], duration=sup[3], channel=0,
                                     text=sup[0])
            c = MonoCut(id=sup.id, start=sup.start, duration=sup.duration, channel=sup.channel, recording=recording,
                        supervisions=[sup])
            return c, energy_to_match

        elif (token in sups_bin_5 and len(sups_bin_5[token])>=1):
            matched_sups = sups_bin_5[token]
            sup = matched_sups[0]
            energy_to_match = recordings[sup[1]][1]
            recording = recordings[sup[1]][0]
            sup = SupervisionSegment(id=sup[0], recording_id=sup[1], start=sup[2], duration=sup[3], channel=0,
                                     text=sup[0])
            c = MonoCut(id=sup.id, start=sup.start, duration=sup.duration, channel=sup.channel, recording=recording,
                        supervisions=[sup])
            return c, energy_to_match

    if (b == 3):
        if (token in sups_bin_3 and len(sups_bin_3[token])>=1):
            matched_sups = sups_bin_3[token]
            sup = random.sample(matched_sups, 1)[0]
            energy_to_match = recordings[sup[1]][1]
            recording = recordings[sup[1]][0]
            sup = SupervisionSegment(id=sup[0], recording_id=sup[1], start=sup[2], duration=sup[3], channel=0,
                                     text=sup[0])
            c = MonoCut(id=sup.id, start=sup.start, duration=sup.duration, channel=sup.channel, recording=recording,
                        supervisions=[sup])
            return c, energy_to_match
        elif (token in sups_bin_2 and len(sups_bin_2[token])>=1):
            matched_sups = sups_bin_2[token]
            sup = matched_sups[-1]
            energy_to_match = recordings[sup[1]][1]
            recording = recordings[sup[1]][0]
            sup = SupervisionSegment(id=sup[0], recording_id=sup[1], start=sup[2], duration=sup[3], channel=0,
                                     text=sup[0])
            c = MonoCut(id=sup.id, start=sup.start, duration=sup.duration, channel=sup.channel, recording=recording,
                        supervisions=[sup])
            return c, energy_to_match

        elif (token in sups_bin_4 and len(sups_bin_4[token])>=1):
            matched_sups = sups_bin_4[token]
            sup = matched_sups[0]
            energy_to_match = recordings[sup[1]][1]
            recording = recordings[sup[1]][0]
            sup = SupervisionSegment(id=sup[0], recording_id=sup[1], start=sup[2], duration=sup[3], channel=0,
                                     text=sup[0])
            c = MonoCut(id=sup.id, start=sup.start, duration=sup.duration, channel=sup.channel, recording=recording,
                        supervisions=[sup])
            return c, energy_to_match
        elif (token in sups_bin_1 and len(sups_bin_1[token])>=1):
            matched_sups = sups_bin_1[token]
            sup = matched_sups[-1]
            energy_to_match = recordings[sup[1]][1]
            recording = recordings[sup[1]][0]
            sup = SupervisionSegment(id=sup[0], recording_id=sup[1], start=sup[2], duration=sup[3], channel=0,
                                     text=sup[0])
            c = MonoCut(id=sup.id, start=sup.start, duration=sup.duration, channel=sup.channel, recording=recording,
                        supervisions=[sup])
            return c, energy_to_match

        elif (token in sups_bin_5 and len(sups_bin_5[token])>=1):
            matched_sups = sups_bin_5[token]
            sup = matched_sups[0]
            energy_to_match = recordings[sup[1]][1]
            recording = recordings[sup[1]][0]
            sup = SupervisionSegment(id=sup[0], recording_id=sup[1], start=sup[2], duration=sup[3], channel=0,
                                     text=sup[0])
            c = MonoCut(id=sup.id, start=sup.start, duration=sup.duration, channel=sup.channel, recording=recording,
                        supervisions=[sup])
            return c, energy_to_match

    if (b == 4):
        if (token in sups_bin_4 and len(sups_bin_4[token])>=1):
            matched_sups = sups_bin_4[token]
            sup = random.sample(matched_sups, 1)[0]
            energy_to_match = recordings[sup[1]][1]
            recording = recordings[sup[1]][0]
            sup = SupervisionSegment(id=sup[0], recording_id=sup[1], start=sup[2], duration=sup[3], channel=0,
                                     text=sup[0])
            c = MonoCut(id=sup.id, start=sup.start, duration=sup.duration, channel=sup.channel, recording=recording,
                        supervisions=[sup])
            return c, energy_to_match
        elif (token in sups_bin_3 and len(sups_bin_3[token])>=1):
            matched_sups = sups_bin_3[token]
            sup = matched_sups[-1]
            energy_to_match = recordings[sup[1]][1]
            recording = recordings[sup[1]][0]
            sup = SupervisionSegment(id=sup[0], recording_id=sup[1], start=sup[2], duration=sup[3], channel=0,
                                     text=sup[0])
            c = MonoCut(id=sup.id, start=sup.start, duration=sup.duration, channel=sup.channel, recording=recording,
                        supervisions=[sup])
            return c, energy_to_match

        elif (token in sups_bin_5 and len(sups_bin_5[token])>=1):
            matched_sups = sups_bin_5[token]
            sup = matched_sups[0]
            energy_to_match = recordings[sup[1]][1]
            recording = recordings[sup[1]][0]
            sup = SupervisionSegment(id=sup[0], recording_id=sup[1], start=sup[2], duration=sup[3], channel=0,
                                     text=sup[0])
            c = MonoCut(id=sup.id, start=sup.start, duration=sup.duration, channel=sup.channel, recording=recording,
                        supervisions=[sup])
            return c, energy_to_match


        elif (token in sups_bin_2 and len(sups_bin_2[token])>=1):
            matched_sups = sups_bin_2[token]
            sup = matched_sups[-1]
            energy_to_match = recordings[sup[1]][1]
            recording = recordings[sup[1]][0]
            sup = SupervisionSegment(id=sup[0], recording_id=sup[1], start=sup[2], duration=sup[3], channel=0,
                                     text=sup[0])
            c = MonoCut(id=sup.id, start=sup.start, duration=sup.duration, channel=sup.channel, recording=recording,
                        supervisions=[sup])
            return c, energy_to_match
        elif (token in sups_bin_1 and len(sups_bin_1[token])>=1):
            matched_sups = sups_bin_1[token]
            sup = matched_sups[-1]
            energy_to_match = recordings[sup[1]][1]
            recording = recordings[sup[1]][0]
            sup = SupervisionSegment(id=sup[0], recording_id=sup[1], start=sup[2], duration=sup[3], channel=0,
                                     text=sup[0])
            c = MonoCut(id=sup.id, start=sup.start, duration=sup.duration, channel=sup.channel, recording=recording,
                        supervisions=[sup])
            return c, energy_to_match

    if (b == 5):
        if (token in sups_bin_5 and len(sups_bin_5[token])>=1):
            matched_sups = sups_bin_5[token]
            sup = random.sample(matched_sups, 1)[0]
            energy_to_match = recordings[sup[1]][1]
            recording = recordings[sup[1]][0]
            sup = SupervisionSegment(id=sup[0], recording_id=sup[1], start=sup[2], duration=sup[3], channel=0,
                                     text=sup[0])
            c = MonoCut(id=sup.id, start=sup.start, duration=sup.duration, channel=sup.channel, recording=recording,
                        supervisions=[sup])
            return c, energy_to_match
        elif (token in sups_bin_4 and len(sups_bin_4[token])>=1):
            matched_sups = sups_bin_4[token]
            sup = matched_sups[-1]
            energy_to_match = recordings[sup[1]][1]
            recording = recordings[sup[1]][0]
            sup = SupervisionSegment(id=sup[0], recording_id=sup[1], start=sup[2], duration=sup[3], channel=0,
                                     text=sup[0])
            c = MonoCut(id=sup.id, start=sup.start, duration=sup.duration, channel=sup.channel, recording=recording,
                        supervisions=[sup])
            return c, energy_to_match

        elif (token in sups_bin_3 and len(sups_bin_3[token])>=1):
            matched_sups = sups_bin_3[token]
            sup = matched_sups[-1]
            energy_to_match = recordings[sup[1]][1]
            recording = recordings[sup[1]][0]
            sup = SupervisionSegment(id=sup[0], recording_id=sup[1], start=sup[2], duration=sup[3], channel=0,
                                     text=sup[0])
            c = MonoCut(id=sup.id, start=sup.start, duration=sup.duration, channel=sup.channel, recording=recording,
                        supervisions=[sup])
            return c, energy_to_match


        elif (token in sups_bin_2 and len(sups_bin_2[token])>=1):
            matched_sups = sups_bin_2[token]
            sup = matched_sups[-1]
            energy_to_match = recordings[sup[1]][1]
            recording = recordings[sup[1]][0]
            sup = SupervisionSegment(id=sup[0], recording_id=sup[1], start=sup[2], duration=sup[3], channel=0,
                                     text=sup[0])
            c = MonoCut(id=sup.id, start=sup.start, duration=sup.duration, channel=sup.channel, recording=recording,
                        supervisions=[sup])
            return c, energy_to_match
        elif (token in sups_bin_1 and len(sups_bin_1[token])>=1):
            matched_sups = sups_bin_1[token]
            sup = matched_sups[-1]
            energy_to_match = recordings[sup[1]][1]
            recording = recordings[sup[1]][0]
            sup = SupervisionSegment(id=sup[0], recording_id=sup[1], start=sup[2], duration=sup[3], channel=0,
                                     text=sup[0])
            c = MonoCut(id=sup.id, start=sup.start, duration=sup.duration, channel=sup.channel, recording=recording,
                        supervisions=[sup])
            return c, energy_to_match


def create_cs_audio(generated_text, output_directory_path, supervisions, recordings, non_freq_sups, sups_bin_1,
                    sups_bin_2, sups_bin_3, sups_bin_4, sups_bin_5, percents):
    # generated_text=open(input_file_path,'r').readlines()

    # filename=input_file_path.split('/')[-1]
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
        alignment = []
        for j in range(len(sentence_token)):
            token = sentence_token[j]
            print(token)
            if (token in supervisions):

                if not cut:
                    matched_sups = supervisions[token]
                    # print(matched_sups)
                    sup = random.sample(matched_sups, 1)[0]
                    # print(sup)
                    # print(recordings[sup[1]])
                    energy_to_match = recordings[sup[1]][1]
                    recording = recordings[sup[1]][0]
                    sup = SupervisionSegment(id=sup[0], recording_id=sup[1], start=sup[2], duration=sup[3], channel=0,
                                             text=sup[0])

                    cut = MonoCut(id=sup.id, start=sup.start, duration=sup.duration, channel=sup.channel,
                                  recording=recording, supervisions=[sup])
                    alignment.append((cut.supervisions[0].recording_id, 1, cut.supervisions[0].start, cut.supervisions[0].duration, cut.supervisions[0].text))

                else:
                    if (token in non_freq_sups):
                        matched_sups = non_freq_sups[token]
                        sup = random.sample(matched_sups, 1)[0]
                        energy_to_match = recordings[sup[1]][1]
                        recording = recordings[sup[1]][0]
                        sup = SupervisionSegment(id=sup[0], recording_id=sup[1], start=sup[2], duration=sup[3],
                                                 channel=0, text=sup[0])

                        c = MonoCut(id=sup.id, start=sup.start, duration=sup.duration, channel=sup.channel,
                                    recording=recording, supervisions=[sup])

                        cut = cut.append(c)
                        alignment.append((c.supervisions[0].recording_id, 1, c.supervisions[0].start, c.supervisions[0].duration, c.supervisions[0].text))
                    else:
                        b = find_bin(energy_to_match, percents)
                        c, e = find_token(token, b, bins, recordings)
                        alignment.append((token,c.recording.id, c.start,c.duration))
                        cut = cut.append(c)
                        energy_to_match = e

                        alignment.append((c.supervisions[0].recording_id, 1, c.supervisions[0].start, c.supervisions[0].duration, c.supervisions[0].text))


        end_time = datetime.now()
        delta = (end_time - start_time)
        print('making sentence time: ', delta)


        if cut is not None:
            start_time = datetime.now()
            audio = cut.load_audio()
            end_time = datetime.now()
            delta = (end_time - start_time)
            print('loading audio time: ', delta)


            start_time = datetime.now()
            torchaudio.save(output_directory_path+'/'+file_name+'.wav', torch.from_numpy(audio),sample_rate=16000, encoding="PCM_S", bits_per_sample=16)
            end_time = datetime.now()
            delta = (end_time - start_time)
            print('saving audio time: ', delta)

            with codecs.open(output_directory_path+"/"+file_name+".ctm", "w", "utf-8") as f:
                for line in alignment:
                    print("%s %d %.2f %.2f %s" % line, file=f)


    with open(output_directory_path+'/transcripts.txt','w') as f:
        for t in transcripts:
            f.write(t+'\n')
    with open(output_directory_path+'/alignments.json', 'w') as f: 
        json.dump(alignments,f) 


if __name__ == "__main__":
    sup_dict_path = sys.argv[1]
    rec_dict_path = sys.argv[2]
    bins_dict_path = sys.argv[3]
    # non_freq_dict_path = sys.argv[3]i
    # sup_bin_1_dict_path = sys.argv[4]
    # sup_bin_2_dict_path = sys.argv[5]
    # sup_bin_3_dict_path = sys.argv[6]
    # sup_bin_4_dict_path = sys.argv[7]
    # sup_bin_5_dict_path = sys.argv[8]

    input_path = sys.argv[4]
    output_path = sys.argv[5]

    supervisions, recordings, bins, percents = load_dicts_modified(
        sup_dict_path, rec_dict_path, bins_dict_path)
        # non_freq_dict_path, sup_bin_1_dict_path, sup_bin_2_dict_path, sup_bin_3_dict_path,
        # sup_bin_4_dict_path, sup_bin_5_dict_path)
    generated_text = open(input_path, 'r').readlines()
    create_cs_audio(generated_text, output_path, supervisions, recordings, bins, percents)
