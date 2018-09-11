import torch
import os

def save_model(model, name):        
	torch.save(model, os.path.join("saved_models",name + ".pt")) 
	print(f"A model is saved successfully as {name}!")
        

def load_model(name):
#	try:
#		if iscuda:
		model = torch.load(os.path.join("saved_models",name + ".pt"))

		#model = pickle.load(open(path, "rb"))
		print(f"Model in {name} loaded successfully!")

		return model
#	except:
#		print(f"No available model such as {name}.")
#		exit()