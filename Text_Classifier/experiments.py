import subprocess

experiments_to_run = []
fixed = "python main.py --cuda False --load_model False --save_model True"
experiments_to_run.append(fixed + " --nn_model CNN --learning_rate 0.01 --max_length 200 --embeddings_dim 100 --dropout_p 0.5 --num_sm_hidden 100 --batch_size 30 --num_epochs 10 --y_start 2017 --y_end 2017 --dataset 30-fold-8-classes --fold 1 --early_stop True --patience 2 --th 0.0 0.5 0.9")

experiments_to_run.append(fixed + " --nn_model RCNN --learning_rate 0.01 --max_length 200 --embeddings_dim 100 --dropout_p 0.5 --num_sm_hidden 100 --batch_size 30 --num_epochs 100 --y_start 2010 --y_end 2018 --dataset 30-fold-8-classes --fold 1 --early_stop True --patience 5 --th 0.0")

experiments_to_run.append(fixed + " --nn_model RCNN --learning_rate 0.001 --max_length 200 --embeddings_dim 100 --dropout_p 0.5 --num_sm_hidden 100 --batch_size 30 --num_epochs 100 --y_start 2010 --y_end 2018 --dataset 30-fold-8-classes --fold 1 --early_stop True --patience 5 --th 0.0")

experiments_to_run.append(fixed + " --nn_model RCNN --learning_rate 0.01 --max_length 500 --embeddings_dim 100 --dropout_p 0.5 --num_sm_hidden 100 --batch_size 30 --num_epochs 100 --y_start 2010 --y_end 2018 --dataset 30-fold-8-classes --fold 1 --early_stop True --patience 5 --th 0.0")

experiments_to_run.append(fixed + " --nn_model RCNN --learning_rate 0.001 --max_length 500 --embeddings_dim 100 --dropout_p 0.5 --num_sm_hidden 100 --batch_size 30 --num_epochs 100 --y_start 2010 --y_end 2018 --dataset 30-fold-8-classes --fold 1 --early_stop True --patience 5 --th 0.0")

experiments_to_run.append(fixed + " --nn_model RCNN --learning_rate 0.01 --max_length 1000 --embeddings_dim 100 --dropout_p 0.5 --num_sm_hidden 100 --batch_size 15 --num_epochs 100 --y_start 2010 --y_end 2018 --dataset 30-fold-8-classes --fold 1 --early_stop True --patience 5 --th 0.0")

experiments_to_run.append(fixed + " --nn_model RCNN --learning_rate 0.001 --max_length 1000 --embeddings_dim 100 --dropout_p 0.5 --num_sm_hidden 100 --batch_size 15 --num_epochs 100 --y_start 2010 --y_end 2018 --dataset 30-fold-8-classes --fold 1 --early_stop True --patience 5 --th 0.0")

experiments_to_run.append(fixed + " --nn_model RCNN --learning_rate 0.01 --max_length 2000 --embeddings_dim 100 --dropout_p 0.5 --num_sm_hidden 100 --batch_size 10 --num_epochs 100 --y_start 2010 --y_end 2018 --dataset 30-fold-8-classes --fold 1 --early_stop True --patience 5 --th 0.0")

experiments_to_run.append(fixed + " --nn_model RCNN --learning_rate 0.001 --max_length 2000 --embeddings_dim 100 --dropout_p 0.5 --num_sm_hidden 100 --batch_size 10 --num_epochs 100 --y_start 2010 --y_end 2018 --dataset 30-fold-8-classes --fold 1 --early_stop True -- patience 5 --th 0.0")

for exp in experiments_to_run:
    print("Running experiment with args:\n" + exp)
    args = exp.split(" ")
    subprocess.run(args)

