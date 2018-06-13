import torch 
import torch.autograd as autograd
import torch.nn as nn

def train(train_loader, dev_loader, model, cuda, learnign_rate, num_epochs, log_file):

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
			target.data.sub_(1)  # index 
			if cuda:
				feature, target = feature.cuda(), target.cuda()
			#if feature.size()[1] < 5:
			#	continue 

			#Forward, Backward, Optimize 
			optimizer.zero_grad()
			output = model(feature)
            #XXX: unclear fix! but it works
			if len(out_src.size()) < 3:
				output = output.unsqueeze(0)
				target = target.unsqueeze(0)
			#if list(output.size())[0]!=20 or list(output.size())[1]!=8 or target.size()[0]!=20:
				#with open(log_file, 'a') as the_file:
					#the_file.write('\nOutput: ' + str(output.size()) + " target: " + str(target.size()))
					#the_file.close()
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
				msg = eval(dev_loader, model, cuda)
				print(msg)
				with open(log_file, 'a') as the_file:
					the_file.write('\nDev: ' + msg)
					the_file.close()
				#print(predicted[:10])
                          

def eval(test_loader, model, cuda):
 	#eval mode 
 	model.eval()

 	#Loss and optimizer 
 	criterion = nn.CrossEntropyLoss()

 	corrects = 0
 	avg_loss = 0 
 	with torch.no_grad(): 
 		for i, batch in enumerate(test_loader):
 			feature, target = batch.text, batch.label
 			target.data.sub_(1) # index
 			if cuda:
 				feature, target = feature.cuda(), target.cuda()
    
 			output = model(feature)
 			#loss = criterion(output, target) # losses are summed, not average 

 			#avg_loss += loss.item()
 			corrects += (torch.max(output, 1)
                     [1].view(target.size()).data == target.data).sum()
 	
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








