import json
import sys 


if __name__ =="__main__": 


	sup_dict_path=sys.argv[1]
	sups=json.load(open(sup_dict_path))
	keys=list(sups.keys())

	output_vocab_path=sys.argv[2]
	with open(output_vocab_path,'w') as f:
		json.dump(keys,f)


