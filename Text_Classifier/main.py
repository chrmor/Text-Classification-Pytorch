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
from nltk.corpus import stopwords

torch.manual_seed(0)

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
	with open(log_file, 'a') as the_file:
		the_file.write('\n\ndataset' +  ds)
		the_file.write('\nlen(train)' +  str(len(train_data)))
		the_file.write('\nlen(dev)' +  str(len(dev_data)))
		the_file.write('\nlen(test)' + str(len(test_data)))
		the_file.close()
    

	train_loader, dev_loader, test_loader = data.BucketIterator.splits(
		(train_data, dev_data, test_data), batch_sizes = (b_size, len(dev_data), len(test_data)), **kwargs)



	return train_loader, dev_loader, test_loader

def clean_str(strings):
    stop_words = list(set(stopwords.words('english')))
    stop_characters = ["`", "\'", "\"", ".", "\(", "\)", "," , '``', "''", '--', '...']
    stop_words.extend(stop_characters)
    filtered_words = [word for word in strings if word not in stop_words]
    return filtered_words


if __name__=='__main__':

	   
    #glove 6B 100 dim / glove 6B 300 dim /glove 42B 300 dim 
	glove = vocab.GloVe(name = '6B', dim = 100)
	embeddings = 'glove-6B-100'
	iscuda = False
	WE_dataset = '2012-2017-full-text'
	device_value = 'cpu'  	#device = - 1 : cpu 
	batch_size = 20
	nn_model = 'RCNN'
	log_file = 'log' + nn_model + '.txt'
	max_length = 400

#	if torch.cuda.is_available() is True:
#		iscuda = True
#	else:
#		iscuda = False
		#device_value = -1 



	#load data
	print("Load data...")
	# to fix length : fix_length = a 
	text_field = data.Field(lower = True, batch_first = True, fix_length = max_length, preprocessing = clean_str)
	label_field = data.Field(sequential = False)

    #select data set 
	train_loader, dev_loader, test_loader = WE_data_loader(text_field, label_field, glove, batch_size, log_file, ds = WE_dataset, device = device_value, repeat = False)
	#train_loader, dev_loader, test_loader = News_20_data_loader(text_field, label_field, glove, batch_size, device = device_value, repeat = False)
	#train_loader, dev_loader, test_loader = SST_data_loader(text_field, label_field, glove, batch_size, device = device_value, repeat = False)
	#train_loader, dev_loader, test_loader = MR_data_loader(text_field, label_field, glove, batch_size, device = device_value, repeat = False)


    #parameters 
	in_channels = 1 
	out_channels = 2
	voca_size = len(text_field.vocab)
	num_classes = len(label_field.vocab) - 1 
	embed_dim = glove.vectors.size()[1]
	kernel_sizes = [3,4,5]
	dropout_p = 0.5
	embedding_weight = text_field.vocab.vectors

	learnign_rate = 0.001
	num_epochs = 25

	#parameter of rnn 
	num_layer  = 25
	num_hidden = 128

	#param of rcnn
	num_sm_hidden = 100 

	with open(log_file, 'a') as the_file:
		the_file.write("\nModel: " + nn_model)
		the_file.write("\nMax length: " + str(max_length))
		the_file.write("\nbatch_size: " + str(batch_size));
		the_file.write("\nEmbeddings: " + str(embeddings));
		the_file.close()
    
	# model 
	print("Load model...")
	if nn_model == 'RCNN':
		classifier_model = model.RCNN_Classifier(voca_size, embed_dim, num_hidden, num_sm_hidden, num_layer, num_classes, embedding_weight,iscuda)
	elif nn_model == 'CNN':
		classifier_model = model.CNNClassifier(in_channels, out_channels, voca_size, embed_dim, num_classes, kernel_sizes, dropout_p, embedding_weight)
	elif nn_model == 'RNN':
		classifier_model = model.RNNClassifier(voca_size, embed_dim, num_hidden, num_layer, num_classes, embedding_weight, iscuda)
        
         

	if iscuda:
		classifier_model = classifier_model.cuda()

	# train 
	print("Start Train...")
	train.train(train_loader, dev_loader, classifier_model, iscuda, learnign_rate, num_epochs, log_file)

	# eval 
	print("Evaluation")
	msg = train.eval(test_loader, classifier_model, iscuda) 
	print(msg)
	with open(log_file, 'a') as the_file:
		the_file.write('\nTest: ' + msg)
		the_file.close()


	