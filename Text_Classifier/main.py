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
import argparse
import csv

def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Text Classification Experiments')
parser.add_argument('--dataset', help='the dataset')
parser.add_argument('--fold', help='the fold')
parser.add_argument('--load_model', type=str2bool, help='load a model?')
parser.add_argument('--load_model_name', help='the name of the model to be loaded')
parser.add_argument('--save_model', type=str2bool, help='save the trained model?')
parser.add_argument('--save_model_name', help='the name of the trained model to be saved')
parser.add_argument('--y_start', type=int, help='The beginning year of the time range of data')
parser.add_argument('--y_end', type=int, help='The ending year of the time range of data')
parser.add_argument('--do_training', type=str2bool, help='Perform training? dafault: True')
parser.add_argument('--do_eval', type=str2bool, help='Perform evaluation on test set? dafault: True')
parser.add_argument('--ths', type=float, nargs='+', help='Thresholds')
parser.add_argument('--cuda', type=str2bool, help='Use CUDA?')
parser.add_argument('--use_gputil', type=str2bool, help='Use GPUtil?')

parser.add_argument('--batch_size', type=int, help='Size of batches')
parser.add_argument('--learning_rate', type=float, help='The learning rate of the gradient descent')
parser.add_argument('--max_length', type=int, help='max length of classifier input text in number of words')
parser.add_argument('--embeddings', help='Pretrained embeddings to use')
parser.add_argument('--embeddings_dim', type=int, help='Lenght of word embedded vectors')
parser.add_argument('--nn_model', help='the model to train (RCNN|CNN)')
parser.add_argument('--dropout_p', type=float, help='Dropout probability')
parser.add_argument('--num_sm_hidden', type=int, help='Dimension of the hidden layer in SM net')

parser.add_argument('--early_stop', type=str2bool, help='If true stops learning as soon as accuracy goes down on dev set')




args = parser.parse_args()

seedId = 0;
torch.manual_seed(seedId)

iscuda = True
use_gputil = True
if args.cuda != None:
	iscuda = args.cuda
if args.use_gputil != None:
	use_gputil = args.use_gputil

if iscuda and use_gputil:
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
        
def save_model(model, name):        
	torch.save(model, os.path.join("saved_models",name + ".pt")) 
	print(f"A model is saved successfully as {name}!")
        

def load_model(name):
	try:
		if iscuda:
			model = torch.load(os.path.join("saved_models",name + ".pt"))
		else:
			#model = torch.load(path, map_location=lambda storage, loc: storage)
			model = torch.load(os.path.join("saved_models",name + ".pt"))
			model = model.cpu()
		#model = pickle.load(open(path, "rb"))
		print(f"Model in {name} loaded successfully!")

		return model
	except:
		print(f"No available model such as {name}.")
		exit()

        
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
			the_file.write('\n\nfold indexes:' +  os.path.join(idx_path,str(fold)))
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

	root_path = os.path.join('..','data')
    
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
	"num_sm_hidden": 100, #PAPER : 100
	"nn_model": 'RCNN',#options.dataset,
	"dropout_p": 0.5,
	"learning_rate": 0.001,  #PAPER: 0.01     
	"max_length": 1000,
	"num_epochs": 4,
	"batch_size": 15,      
            
	"data_folder": 'json',
	"start": 2010,
	"end": 2018,
	"prefix": 'wiki-events-',
	"suffix": '_multilink_data_id_clean',
	"dataset": '30-fold-8-classes',
	"fold": 1,
	"early_stop": True        
}
    
	ths = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]    
    
#COMMAND LINE ARGUMENTS

	if args.fold != None:
		params["fold"] = int(args.fold)
	if args.dataset != None: 
		params["dataset"] = args.dataset
	if args.load_model != None:
		params["load_model"] = args.load_model
	if args.save_model != None:
		params["save_model"] = args.save_model
	if args.load_model_name != None:
		params["load_model_name"] = args.load_model_name
	if args.save_model_name != None:
		params["save_model_name"] = args.save_model_name
	if args.batch_size != None:
		params["batch_size"] = args.batch_size
	if args.num_epochs != None:
		params["num_epochs"] = args.num_epochs  
	if args.nn_model != None:
		params["nn_model"] = args.nn_model
	if args.y_start != None:
		params["start"] = args.y_start
	if args.y_end != None:
		params["end"] = args.y_end 
	if args.do_training != None:
		params["do_training"] = args.do_training
	if args.do_eval != None:
		params["do_eval"] = args.do_eval
	if args.embeddings != None:
		params["embeddings"] = args.embeddings        
	if args.embeddings_dim != None:
		params["embeddings_dim"] = args.embeddings_dim
	if args.learning_rate != None:
		params["learning_rate"] = args.learning_rate
	if args.max_length != None:
		params["max_length"] = args.max_length        
	if args.num_sm_hidden != None:
		params["num_sm_hidden"] = args.num_sm_hidden        
	if args.dropout_p != None:
		params["dropout_p"] = args.dropout_p
	if args.early_stop != None:
		params["early_stop"] = args.early_stop
	if args.ths != None:
		ths = args.ths
    
	experiment_name = f"{params['nn_model']}_{params['max_length']}_{params['num_sm_hidden']}_{params['embeddings']}_{params['embeddings_dim']}_{params['num_epochs']}_{params['batch_size']}_{params['learning_rate']}_{params['dropout_p']}_dataset-{params['dataset']}_fold-{params['fold']}_earlystop-{params['early_stop']}"

if params['save_model_name'] != None:
	log_name = params['save_model_name']
else:
	log_name = experiment_name
log_file = os.path.join("logs",log_name + ".txt")
log_file_samples = os.path.join("logs",log_name + "_SAMPLES.txt")
csv_path = os.path.join("csv",log_name + ".csv")
    
glove = vocab.GloVe(name = '6B', dim = params['embeddings_dim'])
    
if (iscuda):
	device_value = 0  #device = - 1 : cpu 
else:
	device_value = -1 #device = - 1 : cpu 

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
	train_loader, dev_loader, test_loader = WE_2_data_loader(text_field, label_field, os.path.join(root_path,params["dataset"] + '-' + str(params["start"]) + '-' + str(params["end"])), params["fold"], os.path.join(root_path,params["data_folder"]), params["start"], params["end"], params["prefix"], params["suffix"], glove, params['batch_size'], dataloader_log_file, device = device_value, repeat = False)
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
		if params['load_model_name'] != None:
			the_file.write("\nLoaded pre-trained model: " + params['load_model_name'])    
		the_file.write("\nModel: " + params['nn_model'])
		the_file.write("\nMax length: " + str(params['max_length']))
		the_file.write("\nbatch_size: " + str(params['batch_size']));
		the_file.write("\nEmbeddings: " + str(params['embeddings']));
		the_file.close()
    
# model 
if params['load_model']:
	if params['load_model_name'] != None:        
		classifier_model = load_model(params['load_model_name'])
	else:    
		classifier_model = load_model(experiment_name)
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
	train.train(train_loader, dev_loader, classifier_model, iscuda, params['learning_rate'], params['num_epochs'], params['batch_size'], log_file, params["early_stop"])
        
if params['save_model']:
	if params['save_model_name']!=None:
		save_model(classifier_model, params['save_model_name'])            
	else:       
		save_model(classifier_model, experiment_name)
        
if params['do_eval']:        
        
	# init CSV file, write header
	csv_file = open(csv_path,"a") 
	csv_writer = csv.writer(csv_file, delimiter=',')
	csv_header = ['TH','Class', 'Coverage', '#Covered', '#Total', 'Accuracy', '#Correct']
	csv_writer.writerow(csv_header)
        
	print_evaluation_details = False
        
	for th in ths:
		msg, csv_rows = train.eval_treshold_classes(label_field, test_loader, classifier_model, iscuda, print_evaluation_details, th) 
		print(msg)
            
		for row in csv_rows:
			csv_writer.writerow(row)
                
		with open(log_file, 'a') as the_file:
			the_file.write('\n\nEvaluation: ' + msg)
                
	csv_file.close()                
            
if params['predict_samples']:
	df = pd.read_csv(os.path.join(root_path,'samples','wiki-events-2014_multilink_data_clean.csv'))
	with open(log_file_samples, 'w') as the_file:
		for index, row in df.iterrows():
			desc = row['Event description']
			text = row['News full text']
			link = row['News link']
			target = row['Event type']
			the_file.write("Event:\n" + desc + "\nOnline news:\n" + link + "\nType: " + target + "\n" )        
			msg = train.predict(text, classifier_model, text_field, label_field, iscuda)
			the_file.write(msg + "\n\n")

