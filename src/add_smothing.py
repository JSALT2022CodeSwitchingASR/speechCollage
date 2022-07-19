#!/usr/bin/env python3
# Copyright (c) Shammur A Chowdhury

# Apache 2.0



import os, sys
import multiprocessing
import time
import argparse

import librosa
import numpy as np
import pywt
import soundfile as sf
import scipy
from scipy.io import wavfile
import math


parser = argparse.ArgumentParser(description='Smothing for generated CS Audio')
# Datasets
parser.add_argument('--input', type=str, required=True,
                    help='wav.scp file')
parser.add_argument('--output', type=str, required=True,
                    help='Output directory')
parser.add_argument('--process', default=25, type=int, metavar='N',
                    help='number of multiprocess to run')



args = parser.parse_args()
print(args)
proc_count=args.process

def madev(d, axis=None):
    """ Mean absolute deviation of a signal """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def wavelet_denoising(x, wavelet='db4', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * madev(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='per')

def add_on_smothing(audio, outdir):
    outfile=outdir+'/'+os.path.basename(audio)
    fs, x = wavfile.read(audio) # Reading audio wave file
    rms_signal=math.sqrt(np.mean(x**2))
    snrdb=35
    snr_deno = 10 ** (snrdb * 0.1)

    rms_noise=math.sqrt(rms_signal**2/snr_deno)
    #Additive White Gaussian Noise (AWGN)
    # std_noise = rms_noise as mean value of AWGN is zero
    noise=np.random.normal(0, rms_noise, x.shape[0])

    ##Add to the signal
    x_nadded=x+noise

    win = scipy.signal.hann(10)
    # win = scipy.signal.blackmanharris(15) #scipy.signal.hann(10)
    filtered = scipy.signal.convolve(x_nadded, win, mode='same') / sum(win)
    # filtered_den = wavelet_denoising(filtered, level=1)
    # sf.write(outfile, filtered_den, fs)
    sf.write(outfile, filtered, fs)


def chunks(list, n):
    return [list[i:i+n] for i in range(0, len(list), n)]


def run_smothing(inaduios, outdir):
    for audio in inaduios:
        add_on_smothing(audio.strip(), outdir)

def main():
    start_time = time.perf_counter()

    inlist=open(args.input, 'r+').readlines()
    outdir=args.output

    total = len(inlist)
    chunk_size = total // proc_count
    slice = chunks(inlist, chunk_size)
    processes = []

    # Creates n processes then starts them
    for i, s in enumerate(slice):
        p = multiprocessing.Process(target=run_smothing, args=(s,outdir))
        p.start()
        processes.append(p)

    # Joins all the processes
    for p in processes:
        p.join()

    finish_time = time.perf_counter()

    print(f"Program finished in {finish_time - start_time} seconds")


if __name__ == "__main__":
    main()
