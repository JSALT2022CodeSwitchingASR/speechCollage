import json 
import numpy as np 
import sys 
#split supervision dict into 5 bins based on energy + 
# a supervisions dict of non freq words


def load_dicts(supervisions_dict_path, recording_dict_path):
	sups=json.load(open(supervisions_dict_path))
	recs=json.load(open(recording_dict_path))
	return sups,recs

def create_non_freq_words_sups(supervisions_dict,threshold):

	sups=supervisions_dict
	non_freq_words=[]
	non_freq_word_sups={}
	for word in sups.keys(): 
		if(len(sups[word])<=threshold):
			non_freq_word_sups[word]=sups[word]
			non_freq_words.append(word)

	l=[sups.pop(word) for word in non_freq_words]
	return non_freq_word_sups,sups

def create_bins_by_percentile(recording_dict, supervisions_dict):

	energies=[]
	for rec in recording_dict.keys():
        	_,energy=recording_dict[rec]
        	energies.append(energy)

	percentiles=[0.0,25.0,50.0,75.0,100.0]

	np_arr=np.array(energies)
	percents=np.percentile(np_arr,percentiles)
	
	sups_bin_1={}
	sups_bin_2={} 
	sups_bin_3={}
	sups_bin_4={}
	sups_bin_5={}
	for word in supervisions_dict.keys(): 
		sups=supervisions_dict[word]
		sups_bin_1[word]=[]
		sups_bin_2[word]=[]
		sups_bin_3[word]=[]
		sups_bin_4[word]=[]
		sups_bin_5[word]=[]
		for sup in sups:
			energy=recording_dict[sup[1]][1]
			if(energy >= percents[0] and energy < percents[1]):
				sups_bin_1[word].append(sup)
			elif(energy >= percents[1] and energy < percents[2]):
				sups_bin_2[word].append(sup)
			elif(energy >= percents[2] and energy<percents[3]):
				sups_bin_3[word].append(sup)
			elif(energy >= percents[3] and energy < percents[4]):
				sups_bin_4[word].append(sup)
			else:
				sups_bin_5[word].append(sup)
		
		sups_bin_1[word].sort(key=lambda s:recording_dict[s[1]][1])
		sups_bin_2[word].sort(key=lambda s:recording_dict[s[1]][1])
		sups_bin_3[word].sort(key=lambda s:recording_dict[s[1]][1])
		sups_bin_4[word].sort(key=lambda s:recording_dict[s[1]][1])
		sups_bin_5[word].sort(key=lambda s:recording_dict[s[1]][1])

		
		if(len(sups_bin_1[word])==0):
			sups_bin_1.pop(word)
		if(len(sups_bin_2[word])==0):
			sups_bin_2.pop(word)
		if(len(sups_bin_3[word])==0):
			sups_bin_3.pop(word)
		if(len(sups_bin_4[word])==0):
			sups_bin_4.pop(word)
		if(len(sups_bin_5[word])==0):
			sups_bin_5.pop(word)	
		
	return sups_bin_1, sups_bin_2, sups_bin_3, sups_bin_4, sups_bin_5
	

if __name__ == "__main__":
	'''sups,recs=load_dicts('/export/home/dzeinal/supervisions.json', '/export/home/dzeinal/recording_dict.json')
	non_freq_sups,sups_new=create_non_freq_words_sups(sups)
	sups_bin_1, sups_bin_2, sups_bin_3, sups_bin_4, sups_bin_5 = create_bins_by_percentile(recs, sups_new)
	
	unigram_bins={}
	unigram_bins['low_freq']=non_freq_sups
	unigram_bins['bin1']=sups_bin_1
	unigram_bins['bin2']=sups_bin_2
	unigram_bins['bin3']=sups_bin_3
	unigram_bins['bin4']=sups_bin_4
	unigram_bins['bin5']=sups_bin_5 


	with open('/export/home/dzeinal/unigram_bins.json','w') as f:
		json.dump(unigram_bins,f)'''
	
	path_to_supervision=sys.argv[1]
	path_to_recording=sys.argv[2]
	output_dir=sys.argv[3]
	sups,recs=load_dicts(path_to_supervision, path_to_recording)
	
	threshold=10 #for unigrams
	non_freq_sups,sups_new=create_non_freq_words_sups(sups, threshold)
	sups_bin_1, sups_bin_2, sups_bin_3, sups_bin_4, sups_bin_5 = create_bins_by_percentile(recs, sups_new)
        
	ngram_bins={}
	ngram_bins['low_freq']=non_freq_sups
	ngram_bins['bin1']=sups_bin_1
	ngram_bins['bin2']=sups_bin_2
	ngram_bins['bin3']=sups_bin_3
	ngram_bins['bin4']=sups_bin_4
	ngram_bins['bin5']=sups_bin_5 
	

	with open(output_dir+'/unigram_bins.json', 'w') as f:
		json.dump(ngram_bins,f) 



