import os 
import sys 
import glob
import random
sys.path.append( os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torchtext.data as data


class WE(data.Dataset):
    
	@staticmethod
	def sort_key(ex):
		return len(ex.text)

	def __init__(self, path, text_field, label_field, examples = None, label_list = None, **kwargs):
		"""Create an  dataset instance given a path and fields.
	    Arguments:
	        path: Path to the data file
	        text_field: The field that will be used for text data.
	        label_field: The field that will be used for label data.
	        Remaining keyword arguments: Passed to the constructor of
	            data.Dataset.
	    """
		fields = [('text', text_field), ('label', label_field)]
		abs_path = os.path.abspath(path)     
		label_list = os.listdir(abs_path)
		label_list.sort()
		if examples is None:
			examples = [] 
			for label in label_list:
				for fname in glob.iglob(os.path.join(path, label, '*')):
					if sys.version_info < (3,):
						f = open(fname, 'r')
					else :
						f = open(fname, 'r', encoding='latin-1')
					text = f.read()
					examples.append(data.Example.fromlist([text, label], fields))

		super(WE, self).__init__(examples, fields, **kwargs)

	def getLabels(dataset = '2012-2017-long-text'):
		path = '.data/' + dataset + "/test"
		abs_path = os.path.abspath(path)     
		label_list = os.listdir(abs_path)
		label_list.sort()        
		return label_list
        
	@classmethod
	def splits(cls, text_field, label_field, root='.data',
	           train='train.txt', validation='dev.txt', test='test.txt', dataset = '2012-2017-long-text', **kwargs):
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
	    dirname = '.data/'
	    path = dirname + dataset
	    train_path = path + "/train"
	    test_path = path + "/test"
	    train_examples = cls(train_path, text_field, label_field, **kwargs).examples
	    test_examples = cls(test_path, text_field, label_field, **kwargs).examples
        
	    random.shuffle(train_examples)
	    dev_ratio = 0.1 
	    #test_ratio = 0.1 
	    dev_index = -1 * int(dev_ratio*len(train_examples))
	    #test_index = -1 * int(test_ratio * len(examples))

	    train_data = None if train is None else cls(
	    	train_path, text_field, label_field, examples=train_examples[:dev_index], **kwargs)
	    val_data = None if validation is None else cls(
	    	train_path, text_field, label_field, examples=train_examples[dev_index:], **kwargs)
	    test_data = None if test is None else cls(
	    	test_path, text_field, label_field, examples=test_examples, **kwargs)
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



