import json 
import os
from lhotse import * 
import sys
#setup recording dictionary give wave.scp file and 
#save it as recording_dict.json in rec_output_folder

def setup_rec_dict(wav_scp_path, rec_output_folder):

	recording_file=open(wav_scp_path, 'r') 

	lines=recording_file.readlines() 
	i=0
	recordings={}
	for line in lines:
		#print(i)
		i+=1
		line=line.split() 
		if(len(line)>2):
			#the number 3 will change depending on the format of your wav.scp	
			if(os.path.isfile(line[3])):
				
				energy=audio.audio_energy(audio.AudioSource(type='file',channels=[0],source=line[3]).load_audio())
				recordings[line[0]]=(line[3], energy)
			
			else: 
				#log audio files that do not exist 
				print(line[3])
		else:
			if(os.path.isfile(line[1])):
				energy=audio.audio_energy(audio.AudioSource(type='file',channels=[0],source=line[1]).load_audio())			
				recordings[line[0]]=(line[1],energy)
			else: 
				#log audio files that do not exist
				print(line[1]) 

	f=open(rec_output_folder+'/recording_dict.json', 'w') 
	json.dump(recordings,f)


if __name__=="__main__":
	setup_rec_dict(sys.argv[1], sys.argv[2])
