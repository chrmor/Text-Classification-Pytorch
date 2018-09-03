import os
import sys 
import torch
sys.path.append( os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torchtext.data as data
import torchtext.datasets as datasets
import torchtext.vocab as vocab
import model 
import train 
from data_utils.MR import MR
from data_utils.News20 import News20
from data_utils.WE import WE
from data_utils.WE_2 import WE_2
from nltk.corpus import stopwords
import pickle
import pandas as pd
import numpy as np

print("Torch version:"  + torch.__version__)

seedId = 0;
torch.manual_seed(seedId)

iscuda = True

if iscuda:
	import GPUtil

	# Get the first available GPU
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	try:
		deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=100, maxMemory=20)  # return a list of available gpus

	except:
		print('GPU not compatible with NVIDIA-SMI')

	else:
		print(deviceIDs[0])
		os.environ["CUDA_VISIBLE_DEVICES"] = str(deviceIDs[0])

def save_model_path(model, path):        
	torch.save(model, path) 
	print(f"A model is saved successfully as {path}!")
        
def save_model(model, params):
	path = f"saved_models/{params['nn_model']}_{params['max_length']}_{params['embeddings']}_{params['embeddings_dim']}_{params['num_epochs']}_{params['batch_size']}_{params['learning_rate']}_dataset-{params['dataset']}.pt"
	save_model_path(model, path)

def load_model_path(path):
	try:
		if iscuda:
			model = torch.load(path)
		else:
			#model = torch.load(path, map_location=lambda storage, loc: storage)
			model = torch.load(path)
			model = model.cpu()
		#model = pickle.load(open(path, "rb"))
		print(f"Model in {path} loaded successfully!")

		return model
	except:
		print(f"No available model such as {path}.")
		exit()

def load_model(params):
	path = f"saved_models/{params['nn_model']}_{params['max_length']}_{params['embeddings']}_{params['embeddings_dim']}_{params['num_epochs']}_{params['batch_size']}_{params['learning_rate']}_dataset-{params['dataset']}.pt"
	return load_model_path(path)
        
def SST_data_loader(text_field, label_field, vector, b_size, **kwargs):


	train_data, dev_data, test_data = datasets.SST.splits(text_field, label_field, fine_grained = True)
    
	text_field.build_vocab(train_data, dev_data, test_data, vectors= vector)
	label_field.build_vocab(train_data, dev_data, test_data, vectors = vector)

	# print information about the data
	print('len(train)', len(train_data))
	print('len(test)', len(test_data))


	train_loader, dev_loader, test_loader = data.BucketIterator.splits(
		(train_data, dev_data, test_data), batch_sizes = (b_size, len(dev_data), len(test_data)), random_state=0, **kwargs)



	return train_loader, dev_loader, test_loader


def MR_data_loader(text_field, label_field, vector, b_size, **kwargs):

	train_data, dev_data, test_data = MR.splits(text_field, label_field)
	text_field.build_vocab(train_data, dev_data, test_data, vectors= vector)
	label_field.build_vocab(train_data, dev_data, test_data, vectors = vector)

	# print information about the data
	print('len(train)', len(train_data))
	print('len(test)', len(test_data))

	train_loader, dev_loader, test_loader = data.BucketIterator.splits(
		(train_data, dev_data, test_data), batch_sizes = (b_size, len(dev_data), len(test_data)), **kwargs)



	return train_loader, dev_loader, test_loader

def News_20_data_loader(text_field, label_field, vector, b_size, **kwargs):

	train_data, dev_data, test_data = News20.splits(text_field, label_field)
	text_field.build_vocab(train_data, dev_data, test_data, vectors= vector)
	label_field.build_vocab(train_data, dev_data, test_data, vectors = vector)

	# print information about the data
	print('len(train)', len(train_data))
	print('len(test)', len(test_data))

	train_loader, dev_loader, test_loader = data.BucketIterator.splits(
		(train_data, dev_data, test_data), batch_sizes = (b_size, len(dev_data), len(test_data)), **kwargs)

	return train_loader, dev_loader, test_loader

def WE_data_loader(text_field, label_field, vector, b_size, log_file, ds, **kwargs):

	train_data, dev_data, test_data = WE.splits(text_field, label_field, dataset = ds)
	text_field.build_vocab(train_data, dev_data, test_data, vectors= vector)
	label_field.build_vocab(train_data, dev_data, test_data, vectors = vector)
    
	# print information about the data
	print('len(train)', len(train_data))
	print('len(dev)', len(dev_data))
	print('len(test)', len(test_data))
	if (log_file!=None):   
		with open(log_file, 'a') as the_file:
			the_file.write('\n\ndataset' +  ds)
			the_file.write('\nlen(train)' +  str(len(train_data)))
			the_file.write('\nlen(dev)' +  str(len(dev_data)))
			the_file.write('\nlen(test)' + str(len(test_data)))
			the_file.close()

def WE_2_data_loader(text_field, label_field, idx_path, fold, data_path, start, end, prefix, suffix, vector, b_size, log_file, **kwargs):

	train_data, dev_data, test_data = WE_2.splits(text_field, label_field, idx_path, fold, data_path, start, end, prefix, suffix)
	text_field.build_vocab(train_data, dev_data, test_data, vectors= vector)
	label_field.build_vocab(train_data, dev_data, test_data, vectors = vector)
    
	# print information about the data
	print('len(train)', len(train_data))
	print('len(dev)', len(dev_data))
	print('len(test)', len(test_data))
	if (log_file!=None):   
		with open(log_file, 'a') as the_file:
			the_file.write('\n\ndata folder:' +  data_path)            
			the_file.write('\n\nfold indexes:' +  idx_path + "/" + str(fold))
			the_file.write('\nlen(train)' +  str(len(train_data)))
			the_file.write('\nlen(dev)' +  str(len(dev_data)))
			the_file.write('\nlen(test)' + str(len(test_data)))
			the_file.close()

	#train_loader, dev_loader, test_loader = data.BucketIterator.splits(
		#(train_data, dev_data, test_data), batch_sizes = (b_size, len(dev_data), len(test_data)), **kwargs)
	train_loader, dev_loader, test_loader = data.BucketIterator.splits(
		(train_data, dev_data, test_data), batch_sizes = (b_size, b_size, b_size), **kwargs)
    
	return train_loader, dev_loader, test_loader

def clean_str(strings):
    stop_words = list(set(stopwords.words('english')))
    stop_characters = ["`", "\'", "\"", ".", "\(", "\)", "," , '``', "''", '--', '...']
    stop_words.extend(stop_characters)
    filtered_words = [word for word in strings if word not in stop_words]
    return filtered_words


if __name__=='__main__':

	root_path = '../data/'
    
#parameters 
	params = {
    #Setting this to True we load a previously trained model with the same parameters as specified here!
	"load_model": False, 
	"load_model_name": None,         
	"do_training": True,
	"do_eval": True,
	"save_model": True,
	"save_model_name": None,        
	"predict_samples": False,    
	#glove 6B 100 dim / glove 6B 300 dim /glove 42B 300 dim 
	"embeddings": 'glove-6B',#options.model,
	"embeddings_dim": 300,
	#parameter of rnn 
	"num_layer": 25,
	"num_hidden": 128, 
	#param of rcnn
	"num_sm_hidden": 100, 
	"nn_model": 'RCNN',#options.dataset,
	"dropout_p": 0.5,
	"learning_rate": 0.001,       
	"max_length": 1000,
	"num_epochs": 4,
	"batch_size": 20,      
            
	"data_folder": 'json',
	"start": 2010,
	"end": 2018,
	"prefix": 'wiki-events-',
	"suffix": '_multilink_data_id_clean',
	"dataset": '30-fold-8-classes-2010-2018',
	"fold": 1    
}
	ext = '.txt'
	model_name = str(params['nn_model']) + "_" + str(params['max_length']) + "_" + str(params['data_folder']) + "_" + params['dataset'] + "-" + str(params['fold']) + "_" + str(params['embeddings']) + "-" + str(params['embeddings_dim']) + "_es-" + str(params['num_epochs']) + "_bs-" + str(params['batch_size']) + "_lr-" + str(params['learning_rate']) + '_seed' + str(seedId)
	log_file = "logs/" + model_name + ext
	log_file_samples = "logs/" + model_name + "_SAMPLES" + ext    
    
	glove = vocab.GloVe(name = '6B', dim = params['embeddings_dim'])
    
	if (iscuda):
		device_value = 0  #device = - 1 : cpu 
	else:
		device_value = -1 #device = - 1 : cpu 


#	if torch.cuda.is_available() is True:
#		iscuda = True
#	else:
#		iscuda = False
		#device_value = -1 

	# to fix length : fix_length = a 
	text_field = data.Field(lower = True, batch_first = True, fix_length = params['max_length'], preprocessing = clean_str)
	label_field = data.Field(sequential = False)

	dataloader_log_file = None
	if (params['do_training'] or params['do_eval']):
		dataloader_log_file = log_file
	if (True):
		#load data
		print("Load data...")
		#select data set 
		train_loader, dev_loader, test_loader = WE_2_data_loader(text_field, label_field, root_path + params["dataset"], params["fold"], root_path + params["data_folder"], params["start"], params["end"], params["prefix"], params["suffix"], glove, params['batch_size'], dataloader_log_file, device = device_value, repeat = False)
		#train_loader, dev_loader, test_loader, label_list = WE_data_loader(text_field, label_field, glove, params['batch_size'], dataloader_log_file, ds = params['data_folder'] + "/" +  params['dataset'], device = device_value, repeat = False)
	#train_loader, dev_loader, test_loader = News_20_data_loader(text_field, label_field, glove, params['batch_size'], device = device_value, repeat = False)
	#train_loader, dev_loader, test_loader = SST_data_loader(text_field, label_field, glove, params['batch_size'], device = device_value, repeat = False)
	#train_loader, dev_loader, test_loader = MR_data_loader(text_field, label_field, glove, params['batch_size'], device = device_value, repeat = False)


	in_channels = 1 
	out_channels = 2
	voca_size = len(text_field.vocab)
	num_classes = len(label_field.vocab) - 1 
	embed_dim = glove.vectors.size()[1]
	kernel_sizes = [3,4,5]

	embedding_weight = text_field.vocab.vectors

	if (params['do_training'] or params['do_eval']):
		with open(log_file, 'a') as the_file:
			the_file.write("\nModel: " + params['nn_model'])
			the_file.write("\nMax length: " + str(params['max_length']))
			the_file.write("\nbatch_size: " + str(params['batch_size']));
			the_file.write("\nEmbeddings: " + str(params['embeddings']));
			the_file.close()
    
	# model 
	if params['load_model']:
		if params['load_model_name'] != None:        
			print("Loading pre-trained model from " + params['load_model_name'] + " ...")            
			classifier_model = load_model_path(params['load_model_name'])
		else:    
			print("Loading pre-trained model with same params ...")
			classifier_model = load_model(params)
	else:
		print("Init new model...")
		if params['nn_model'] == 'RCNN':
			classifier_model = model.RCNN_Classifier(voca_size, embed_dim, params['num_hidden'], params['num_sm_hidden'], params['num_layer'], num_classes, embedding_weight,iscuda)
		elif params['nn_model'] == 'CNN':
			classifier_model = model.CNNClassifier(in_channels, out_channels, voca_size, embed_dim, num_classes, kernel_sizes, params['dropout_p'], embedding_weight)
		elif params['nn_model'] == 'RNN':
			classifier_model = model.RNNClassifier(voca_size, embed_dim, params['num_hidden'], params['num_layer'], num_classes, embedding_weight, iscuda)

	if iscuda:
		classifier_model = classifier_model.cuda()

	if params['do_training']:    
		# train 
		print("Start Train...")
		train.train(train_loader, dev_loader, classifier_model, iscuda, params['learning_rate'], params['num_epochs'], params['batch_size'], log_file)
		print("Finished Train...")
	if params['save_model']:
		if params['save_model_name']!=None:
			save_model_path(classifier_model, params['save_model_name'])            
		else:       
			save_model(classifier_model, params)
        
	if params['do_eval']:        
		# eval
		print_evaluation_details = False
		ths = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]
		print("Evaluation")
		for th in ths:
			msg = train.eval_treshold_classes(label_field, test_loader, classifier_model, iscuda, print_evaluation_details, th) 
			print(msg)
			with open(log_file, 'a') as the_file:
				the_file.write('\nTest: ' + msg)
				the_file.close()
            
	if params['predict_samples']:
		df = pd.read_csv(root_path + '/samples/wiki-events-2014_multilink_data_clean.csv')
		with open(log_file_samples, 'w') as the_file:
			for index, row in df.iterrows():
				desc = row['Event description']
				text = row['News full text']
				link = row['News link']
				target = row['Event type']
				the_file.write("Event:\n" + desc + "\nOnline news:\n" + link + "\nType: " + target + "\n" )        
				msg = train.predict(text, classifier_model, text_field, label_field, iscuda)
				the_file.write(msg + "\n\n")

