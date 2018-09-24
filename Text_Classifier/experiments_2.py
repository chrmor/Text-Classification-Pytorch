import subprocess

experiments_to_run = []
fixed = "python main.py --cuda True --load_model True --save_model False do_training False --use_gputil False --field full-text"

#experiments_to_run.append(fixed + " --nn_model CNN --field wiki-cats --learning_rate 0.001 --max_length 200 --embeddings_dim 300 --dropout_p 0.5 --batch_size 30 --num_epochs 100 --y_start 2010 --y_end 2018 --dataset 30-fold-8-classes --fold 1 --early_stop True --patience 10 --th 0.0 0.5 0.9")

#experiments_to_run.append(fixed + " --nn_model CNN --field wiki-cats --learning_rate 0.001 --max_length 500 --embeddings_dim 300 --dropout_p 0.5 --batch_size 30 --num_epochs 100 --y_start 2010 --y_end 2018 --dataset 30-fold-8-classes --fold 1 --early_stop True --patience 10 --th 0.0 0.5 0.9")

#experiments_to_run.append(fixed + " --nn_model CNN --field full-text --learning_rate 0.001 --max_length 500 --embeddings_dim 300 --dropout_p 0.5 --batch_size 30 --num_epochs 100 --y_start 2010 --y_end 2018 --dataset 30-fold-8-classes --fold 1 --early_stop True --patience 10 --th 0.0 0.5 0.9")

#experiments_to_run.append(fixed + " --nn_model CNN --field full-text --learning_rate 0.001 --max_length 1000 --embeddings_dim 300 --dropout_p 0.5 --batch_size 30 --num_epochs 100 --y_start 2010 --y_end 2018 --dataset 30-fold-8-classes --fold 1 --early_stop True --patience 10 --th 0.0 0.5 0.9")


#RUN POTENA SERVER


experiments_to_run.append(fixed + " --nn_model RCNN --learning_rate 0.001 --max_length 1000 --embeddings_dim 300 --dropout_p 0.1 --num_sm_hidden 100 --batch_size 15 --num_epochs 100 --y_start 2010 --y_end 2018 --dataset 30-fold-8-classes --early_stop True --patience 3 --th 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 --fold 11 --cuda_device 1")

experiments_to_run.append(fixed + " --nn_model RCNN --learning_rate 0.001 --max_length 1000 --embeddings_dim 300 --dropout_p 0.1 --num_sm_hidden 100 --batch_size 15 --num_epochs 100 --y_start 2010 --y_end 2018 --dataset 30-fold-8-classes --early_stop True --patience 2 --th 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 --fold 12 --cuda_device 1")

experiments_to_run.append(fixed + " --nn_model RCNN --learning_rate 0.001 --max_length 1000 --embeddings_dim 300 --dropout_p 0.1 --num_sm_hidden 100 --batch_size 15 --num_epochs 100 --y_start 2010 --y_end 2018 --dataset 30-fold-8-classes --early_stop True --patience 2 --th 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 --fold 13 --cuda_device 1")

experiments_to_run.append(fixed + " --nn_model RCNN --learning_rate 0.001 --max_length 1000 --embeddings_dim 300 --dropout_p 0.1 --num_sm_hidden 100 --batch_size 15 --num_epochs 100 --y_start 2010 --y_end 2018 --dataset 30-fold-8-classes --early_stop True --patience 2 --th 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 --fold 14 --cuda_device 1")

experiments_to_run.append(fixed + " --nn_model RCNN --learning_rate 0.001 --max_length 1000 --embeddings_dim 300 --dropout_p 0.1 --num_sm_hidden 100 --batch_size 15 --num_epochs 100 --y_start 2010 --y_end 2018 --dataset 30-fold-8-classes --early_stop True --patience 2 --th 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 --fold 15 --cuda_device 1")

experiments_to_run.append(fixed + " --nn_model RCNN --learning_rate 0.001 --max_length 1000 --embeddings_dim 300 --dropout_p 0.1 --num_sm_hidden 100 --batch_size 15 --num_epochs 100 --y_start 2010 --y_end 2018 --dataset 30-fold-8-classes --early_stop True --patience 3 --th 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 --fold 16 --cuda_device 1")

experiments_to_run.append(fixed + " --nn_model RCNN --learning_rate 0.001 --max_length 1000 --embeddings_dim 300 --dropout_p 0.1 --num_sm_hidden 100 --batch_size 15 --num_epochs 100 --y_start 2010 --y_end 2018 --dataset 30-fold-8-classes --early_stop True --patience 2 --th 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 --fold 17 --cuda_device 1")

experiments_to_run.append(fixed + " --nn_model RCNN --learning_rate 0.001 --max_length 1000 --embeddings_dim 300 --dropout_p 0.1 --num_sm_hidden 100 --batch_size 15 --num_epochs 100 --y_start 2010 --y_end 2018 --dataset 30-fold-8-classes --early_stop True --patience 2 --th 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 --fold 18 --cuda_device 1")

experiments_to_run.append(fixed + " --nn_model RCNN --learning_rate 0.001 --max_length 1000 --embeddings_dim 300 --dropout_p 0.1 --num_sm_hidden 100 --batch_size 15 --num_epochs 100 --y_start 2010 --y_end 2018 --dataset 30-fold-8-classes --early_stop True --patience 2 --th 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 --fold 19 --cuda_device 1")

experiments_to_run.append(fixed + " --nn_model RCNN --learning_rate 0.001 --max_length 1000 --embeddings_dim 300 --dropout_p 0.1 --num_sm_hidden 100 --batch_size 15 --num_epochs 100 --y_start 2010 --y_end 2018 --dataset 30-fold-8-classes --early_stop True --patience 2 --th 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 --fold 20 --cuda_device 1")


for exp in experiments_to_run:
    print("Running experiment with args:\n" + exp)
    args = exp.split(" ")
    subprocess.run(args)

