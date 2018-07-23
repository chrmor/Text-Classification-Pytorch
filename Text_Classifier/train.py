import torch 
import torch.autograd as autograd
import torch.nn as nn

def train(train_loader, dev_loader, model, cuda, learnign_rate, num_epochs, batch_size, log_file):

	with open(log_file, 'a') as the_file:
		the_file.write("\nModel: " + str(model))
		the_file.write("\nLearning rate: " + str(learnign_rate));
		the_file.write("\nEpochs: " + str(num_epochs));
		the_file.close()
    
    # gpu runnable 
	if cuda: 
		model.cuda()

	#Parallel
	#torch.nn.DataParallel(model)
        
    # train mode 
	model.train()

	num_batch = len(train_loader)
	step = 0

	#Loss and optimizer 
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr = learnign_rate)

	for epoch in range(num_epochs):
		for i, batch in enumerate(train_loader):
			feature, target = batch.text, batch.label
			if feature.size()[0]!= batch_size:
				continue
			target.data.sub_(1)  # index 
			if cuda:
				feature, target = feature.cuda(), target.cuda()
			#if feature.size()[1] < 5:
			#	continue 

			#Forward, Backward, Optimize 
			optimizer.zero_grad()
			output = model(feature)
			#_, predicted = torch.max(output, 1)
			loss = criterion(output, target)
			loss.backward()
			nn.utils.clip_grad_norm_(model.parameters(), 3, norm_type = 2) # l2 constraint of 3

			optimizer.step()

			step += 1 



			if(step) % 100 == 0:
				msg = 'Epoch [%d/%d], Steps [%d/%d], Loss: %.4f' % (epoch+1, num_epochs, step,  num_batch * num_epochs, loss.item())
				print(msg)
				with open(log_file, 'a') as the_file:
					the_file.write('\n' + msg)
					the_file.close()
    

			if(step) % 500 == 0:
				msg = eval(dev_loader, model, cuda, False)
				print(msg)
				with open(log_file, 'a') as the_file:
					the_file.write('\nDev: ' + msg)
					the_file.close()
				#print(predicted[:10])

def eval_treshold(test_loader, model, cuda, print_details, th):
 	#eval mode 
 	model.eval()

 	#Loss and optimizer 
 	criterion = nn.CrossEntropyLoss()

 	corrects = 0
 	predictions = 0
 	avg_loss = 0 
 	with torch.no_grad(): 
 		for i, batch in enumerate(test_loader):
 			feature, target = batch.text, batch.label
 			#print("Batch: " + str(feature.size()) )            
 			target.data.sub_(1) # index
 			if cuda:
 				feature, target = feature.cuda(), target.cuda()
    
 			output = model(feature)
 			#loss = criterion(output, target) # losses are summed, not average 

 			prediction = torch.max(output, 1)[1].view(target.size()).data
 			th_output = (torch.max(output, 1)[0] >= th)
            
 			th_prediction = []
 			th_target = []
 			ix = 0;
 			for item in th_output.data:
 				if item == 1:
 					th_prediction.append(prediction.data[ix].item())
 					th_target.append(target.data[ix].item())
 					predictions += 1;
 				ix += 1
            
 			corrects += (torch.tensor(th_prediction) == torch.tensor(th_target)).sum()
            
 			if print_details:
 				#avg_loss += loss.item()
 				#print("MAX:" + str(torch.max(output, 1)[0]))
 				print("\nOutPut:\n" + str(output.data))
 				print("Target:\n" + str(target.data))
 				print("Prediction:\n" + str(prediction.data))
 				print("TH_Output:\n" + str(th_output.data))
 				#print("\nPrediction:\n" + str(torch.max(output, 1)))
 				print("Corrects:\n" + str(scores))
 				print("TH Prediction:\n" + str(th_prediction))
 				print("TH Target:\n" + str(th_target))
 				#avg_loss += loss.item()
            
            
 	
 	size = len(test_loader.dataset)
 	accuracy = 100 * float(corrects) / predictions 
 	model.train()
 	msg = '\nTH: {:.2f} Recall: {:.2f}%({}/{}) Accuracy: {:.4f}%({}/{}) \n'.format(th, predictions/size, predictions, size, accuracy, corrects, predictions)
 	return msg 


def eval_treshold_classes(label_list, test_loader, model, cuda, print_details, th):
 	#eval mode 
 	model.eval()

 	#Loss and optimizer 
 	criterion = nn.CrossEntropyLoss()

 	predictions_per_class = {}
 	examples_per_class = {}
 	total_examples_per_class = {}    
 	corrects_per_class = {}
 	for label in label_list:
 		predictions_per_class[label] = 0
 		corrects_per_class[label] = 0
 		examples_per_class[label] = 0
 		total_examples_per_class[label] = 0

    
 	corrects = 0
 	predictions = 0
 	avg_loss = 0 
 	with torch.no_grad(): 
 		for i, batch in enumerate(test_loader):
 			feature, target = batch.text, batch.label
 			#print("Batch: " + str(feature.size()) )            
 			target.data.sub_(1) # index
 			if cuda:
 				feature, target = feature.cuda(), target.cuda()
    
 			output = model(feature)
 			#loss = criterion(output, target) # losses are summed, not average 
 			if target.size()==1:
 				print("batch size: " + str(target.size()))
 				continue
 			prediction = torch.max(output, 1)[1].view(target.size()).data
 			th_output = (torch.max(output, 1)[0] >= th)
            
 			th_prediction = []
 			th_target = []
 			ix = 0;
 			for item in th_output.data:
 				if item == 1:
 					th_prediction.append(prediction.data[ix].item())
 					th_target.append(target.data[ix].item())
 					predictions += 1
 				ix += 1
            
 			t_th_target = torch.tensor(th_target)
 			t_th_prediction = torch.tensor(th_prediction)
 			matches = (t_th_prediction == t_th_target)
 			scores = matches.sum()
 			corrects += scores            

 			count = 0;
 			for label in label_list:
 				t_class_target = t_th_target.clone()
 				examples_per_class[label] += (t_class_target==count).sum()
 				total_examples_per_class[label] += (target==count).sum()                
 				t_class_target[t_class_target!=count] = -1
 				corrects_per_class[label] += (torch.tensor(th_prediction) == t_class_target).sum()
 				count += 1
            
 			if print_details:
 				#avg_loss += loss.item()
 				#print("MAX:" + str(torch.max(output, 1)[0]))
 				print("\nOutPut:\n" + str(output.data))
 				print("Target:\n" + str(target))
 				print("Prediction:\n" + str(prediction.data))                
 				print("Matches:\n" + str(matches.data))
 				print("TH_Output:\n" + str(th_output.data))
 				#print("\nPrediction:\n" + str(torch.max(output, 1)))
 				print("Corrects:\n" + str(scores))
 				print("TH Prediction:\n" + str(th_prediction))
 				print("TH Target:\n" + str(th_target))
 				print("Target class: " + str(t_class_target))                
 				print("Accuracy:\n")
 				for label in label_list:
 					accuracy_class = 100 * float(corrects_per_class[label]) / examples_per_class[label]
 					print(label + ": " + str(accuracy_class))
            
 	
 	size = len(test_loader.dataset)
 	accuracy = 100 * float(corrects) / predictions 
 	model.train()
 	msg = '\nTH: {:.2f} Recall: {:.2f}%({}/{})  Accuracy: {:.4f}%({}/{}) \n'.format(th, predictions/size, predictions, size, accuracy, corrects, predictions)
 	dtl_msg = '\nAccuracy per class:\n'
 	for label in label_list:
 		accuracy_class = 100 * float(corrects_per_class[label].item()) / examples_per_class[label].item()
 		dtl_msg += label + ": " + 'Recall: {:.2f}%({}/{})  Accuracy: {:.4f}%({}/{}) \n'.format(float(examples_per_class[label].item())/float(total_examples_per_class[label]), examples_per_class[label].item(), total_examples_per_class[label],  accuracy_class, corrects_per_class[label].item(), examples_per_class[label].item())
 	return msg + dtl_msg


def eval(test_loader, model, cuda, print_details):
 	#eval mode 
 	model.eval()

 	#Loss and optimizer 
 	criterion = nn.CrossEntropyLoss()

 	corrects = 0
 	avg_loss = 0 
 	with torch.no_grad(): 
 		for i, batch in enumerate(test_loader):
 			feature, target = batch.text, batch.label
 			#print("Batch: " + str(feature.size()) )            
 			target.data.sub_(1) # index
 			if cuda:
 				feature, target = feature.cuda(), target.cuda()
    
 			output = model(feature)
 			#loss = criterion(output, target) # losses are summed, not average 

 			#avg_loss += loss.item()
 			scores = (torch.max(output, 1)[1].view(target.size()).data == target.data).sum()
 			corrects += scores
            
 			if print_details:
 				#avg_loss += loss.item()
 				print("\nOutPut:\n" + str(output))
 				print("Target:\n" + str(target.data))
 				print("Max:\n" + str(torch.max(output, 1)[1].view(target.size()).data))
 				#print("\nPrediction:\n" + str(torch.max(output, 1)))
 				print("Corrects:\n" + str(scores))
 	
 	size = len(test_loader.dataset)
 	accuracy = 100 * float(corrects) / size 
 	model.train()
 	#return '\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, accuracy, corrects, size)
 	return '\nEvaluation - acc: {:.4f}%({}/{}) \n'.format(accuracy, corrects, size)


def predict(sample_text, model, text_field, label_field):

	model.eval()

	text = text_field.preprocess(sample_text)
	text = [[text_field.vocab.stoi[x] for x in text]]
	x = text_field.tensor_type(text)
	x = autograd.Variable(x)
	if cuda:
		x = x.cuda()
	output = model(x)
	_, predicted = torch.max(output, 1)

	return label_field.vocab.itos[predicted.data[0]+1]








