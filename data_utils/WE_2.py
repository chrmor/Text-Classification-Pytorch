import os 
import sys 
import glob
import random
sys.path.append( os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torchtext.data as data

import json
import gzip
from random import shuffle

def getEvents(start, end, path, prefix, suffix):
	listEvents = []
	for year in range(start,end+1):
		filename= prefix + str(year) + suffix + '.json.gz'
		print("loading file " + os.path.join(path, filename) + " ...")
		with gzip.open(os.path.join(path, filename), "rb") as f:
			events = json.loads(f.read().decode("utf8"))
		print("found " + str(len(events['results'])) + " events...")
		#we slice results to ignore the last element (dummy object)        
		listEvents = listEvents + events['results'][:-1]
		print("total: " + str(len(listEvents)) + " events...")
	return listEvents

def getEventIndex(listEvents):
	index = {}        
	for event in listEvents:
		idx = event['id']    
		index[int(idx)] = event
	return index    
    
def filterEvents(listEvents, min_text_len):
	filteredEvents = []
	for event in listEvents:
		if 'event-type' in event and 'full-text' in event:
			#keep only events with non empty full-text and event-type
			if event['full-text'] and event['event-type'] and len(event['full-text']) > min_text_len:
				filteredEvents.append(event)
	return filteredEvents

class WE_2(data.Dataset):
    
	@staticmethod
	def sort_key(ex):
		return len(ex.text)

	def __init__(self, eventIndex, path, text_field, label_field, examples = None, **kwargs):

		fields = [('text', text_field), ('label', label_field)]
        
		abs_path = os.path.abspath(path)    
		#globe ignores hidden files (e.g. .DS_Store)
		label_list = os.listdir(abs_path)#glob.glob(os.path.join(abs_path, '*'))
		label_list.sort()
        
		if examples is None:
			examples = [] 
			for label in label_list:
				if label.startswith('.'): continue
				#print("Label: " + label)   
				#print("Opening file: " + os.path.join(path, label, "idx.txt"))
				idxs = tuple(open(os.path.join(path, label, "idx.txt"), 'r'))
				for idx in idxs:
					event = eventIndex[int(idx)]
					text = event['full-text']
					examples.append(data.Example.fromlist([text, label], fields))

		super(WE_2, self).__init__(examples, fields, **kwargs)

        
	@classmethod
	def splits(cls, text_field, label_field, idx_path, fold, data_path, start, end, prefix, suffix, **kwargs):
	    """Create dataset objects for splits of the dataset.
	    Arguments:
	        text_field: The field that will be used for the sentence.
	        label_field: The field that will be used for label data.
	        root: The root directory that the dataset's zip archive will be
	            expanded into; therefore the directory in whose trees
	            subdirectory the data files will be stored.
	        train: The filename of the train data. Default: 'train.txt'.
	        validation: The filename of the validation data, or None to not
	            load the validation set. Default: 'dev.txt'.
	        test: The filename of the test data, or None to not load the test
	            set. Default: 'test.txt'.
	        train_subtrees: Whether to use all subtrees in the training set.
	            Default: False.
	        Remaining keyword arguments: Passed to the splits method of
	            Dataset.
	    """
     
	    eventIndex = getEventIndex(getEvents(start, end, data_path, prefix, suffix))    
	    train_path = os.path.join(idx_path,str(fold),"train")
	    test_path = os.path.join(idx_path,str(fold),"test")        
	    train_examples = cls(eventIndex, train_path, text_field, label_field, **kwargs).examples
	    test_examples = cls(eventIndex, test_path, text_field, label_field, **kwargs).examples       
        
	    random.shuffle(train_examples)
	    dev_ratio = 0.1 
	    #test_ratio = 0.1 
	    dev_index = -1 * int(dev_ratio*len(train_examples))
	    #test_index = -1 * int(test_ratio * len(examples))

	    train_data = cls(
	    	eventIndex, train_path, text_field, label_field, examples=train_examples[:dev_index], **kwargs)
	    val_data = cls(
	    	eventIndex, train_path, text_field, label_field, examples=train_examples[dev_index:], **kwargs)
	    test_data = cls(
	    	eventIndex, test_path, text_field, label_field, examples=test_examples, **kwargs)
	    return tuple(d for d in (train_data, val_data, test_data)
	                 if d is not None)

	@classmethod
	def iters(cls, batch_size=32, device=0, root='.data', vectors=None, **kwargs):
	    """Creater iterator objects for splits of the SST dataset.
	    Arguments:
	        batch_size: Batch_size
	        device: Device to create batches on. Use - 1 for CPU and None for
	            the currently active GPU device.
	        root: The root directory that the dataset's zip archive will be
	            expanded into; therefore the directory in whose trees
	            subdirectory the data files will be stored.
	        vectors: one of the available pretrained vectors or a list with each
	            element one of the available pretrained vectors (see Vocab.load_vectors)
	        Remaining keyword arguments: Passed to the splits method.
	    """
	    TEXT = data.Field()
	    LABEL = data.Field(sequential=False)

	    train, val, test = cls.splits(TEXT, LABEL, root=root, **kwargs)

	    TEXT.build_vocab(train, vectors=vectors)
	    LABEL.build_vocab(train)

	    return data.BucketIterator.splits(
	        (train, val, test), batch_size=batch_size, device=device)



