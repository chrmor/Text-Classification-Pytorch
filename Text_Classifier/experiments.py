import subprocess

experiments_to_run = []
fixed = "python main.py --cuda True --load_model False --save_model True --use_gputil False"

#experiments_to_run.append(fixed + " --nn_model CNN --field wiki-cats --learning_rate 0.001 --max_length 200 --embeddings_dim 300 --dropout_p 0.5 --batch_size 30 --num_epochs 100 --y_start 2010 --y_end 2018 --dataset 30-fold-8-classes --fold 1 --early_stop True --patience 10 --th 0.0 0.5 0.9")

#experiments_to_run.append(fixed + " --nn_model CNN --field wiki-cats --learning_rate 0.001 --max_length 500 --embeddings_dim 300 --dropout_p 0.5 --batch_size 30 --num_epochs 100 --y_start 2010 --y_end 2018 --dataset 30-fold-8-classes --fold 1 --early_stop True --patience 10 --th 0.0 0.5 0.9")

#experiments_to_run.append(fixed + " --nn_model CNN --field full-text --learning_rate 0.001 --max_length 500 --embeddings_dim 300 --dropout_p 0.5 --batch_size 30 --num_epochs 100 --y_start 2010 --y_end 2018 --dataset 30-fold-8-classes --fold 1 --early_stop True --patience 10 --th 0.0 0.5 0.9")

#experiments_to_run.append(fixed + " --nn_model CNN --field full-text --learning_rate 0.001 --max_length 1000 --embeddings_dim 300 --dropout_p 0.5 --batch_size 30 --num_epochs 100 --y_start 2010 --y_end 2018 --dataset 30-fold-8-classes --fold 1 --early_stop True --patience 10 --th 0.0 0.5 0.9")


#RUN POTENA SERVER

#DONE
#experiments_to_run.append(fixed + " --nn_model RCNN --learning_rate 0.01 --max_length 200 --embeddings_dim 100 --dropout_p 0.5 --num_sm_hidden 100 --batch_size 30 --num_epochs 100 --y_start 2010 --y_end 2018 --dataset 30-fold-8-classes --fold 1 --early_stop True --patience 5 --th 0.0")

#DONE
#experiments_to_run.append(fixed + " --nn_model RCNN --learning_rate 0.001 --max_length 200 --embeddings_dim 100 --dropout_p 0.5 --num_sm_hidden 100 --batch_size 30 --num_epochs 100 --y_start 2010 --y_end 2018 --dataset 30-fold-8-classes --fold 1 --early_stop True --patience 5 --th 0.0")

#DONE
#experiments_to_run.append(fixed + " --nn_model RCNN --learning_rate 0.01 --max_length 500 --embeddings_dim 100 --dropout_p 0.5 --num_sm_hidden 100 --batch_size 30 --num_epochs 100 --y_start 2010 --y_end 2018 --dataset 30-fold-8-classes --fold 1 --early_stop True --patience 5 --th 0.0")

#DONE
#experiments_to_run.append(fixed + " --nn_model RCNN --learning_rate 0.001 --max_length 500 --embeddings_dim 100 --dropout_p 0.5 --num_sm_hidden 100 --batch_size 30 --num_epochs 100 --y_start 2010 --y_end 2018 --dataset 30-fold-8-classes --fold 1 --early_stop True --patience 5 --th 0.0")

#DONE
#experiments_to_run.append(fixed + " --nn_model RCNN --learning_rate 0.01 --max_length 1000 --embeddings_dim 100 --dropout_p 0.5 --num_sm_hidden 100 --batch_size 15 --num_epochs 100 --y_start 2010 --y_end 2018 --dataset 30-fold-8-classes --fold 1 --early_stop True --patience 5 --th 0.0")

#DONE
#experiments_to_run.append(fixed + " --nn_model RCNN --learning_rate 0.0001 --max_length 200 --embeddings_dim 100 --dropout_p 0.5 --num_sm_hidden 100 --batch_size 15 --num_epochs 100 --y_start 2010 --y_end 2018 --dataset 30-fold-8-classes --fold 1 --early_stop True --patience 5 --th 0.0")

#DONE
#experiments_to_run.append(fixed + " --nn_model RCNN --learning_rate 0.001 --max_length 1000 --embeddings_dim 100 --dropout_p 0.5 --num_sm_hidden 100 --batch_size 15 --num_epochs 100 --y_start 2010 --y_end 2018 --dataset 30-fold-8-classes --fold 1 --early_stop True --patience 5 --th 0.0")

#DONE
#experiments_to_run.append(fixed + " --nn_model RCNN --learning_rate 0.0001 --max_length 1000 --embeddings_dim 100 --dropout_p 0.5 --num_sm_hidden 100 --batch_size 15 --num_epochs 100 --y_start 2010 --y_end 2018 --dataset 30-fold-8-classes --fold 1 --early_stop True --patience 5 --th 0.0")

#DONE
#experiments_to_run.append(fixed + " --nn_model RCNN --learning_rate 0.001 --max_length 2000 --embeddings_dim 100 --dropout_p 0.5 --num_sm_hidden 100 --batch_size 10 --num_epochs 100 --y_start 2010 --y_end 2018 --dataset 30-fold-8-classes --fold 1 --early_stop True --patience 5 --th 0.0")

#DONE - INTERRUPTED
#experiments_to_run.append(fixed + " --nn_model RCNN --learning_rate 0.001 --max_length 1000 --embeddings_dim 300 --dropout_p 0.5 --num_sm_hidden 100 --batch_size 15 --num_epochs 100 --y_start 2010 --y_end 2018 --dataset 30-fold-8-classes --fold 1 --early_stop True --patience 10 --th 0.0")

#DONE - INTERRRUPTED
#experiments_to_run.append(fixed + " --nn_model RCNN --learning_rate 0.001 --max_length 1500 --embeddings_dim 300 --dropout_p 0.1 --num_sm_hidden 100 --batch_size 10 --num_epochs 100 --y_start 2010 --y_end 2018 --dataset 30-fold-8-classes --fold 1 --early_stop True --patience 10 --th 0.0")

#DONE - BEST
#experiments_to_run.append(fixed + " --nn_model RCNN --learning_rate 0.001 --max_length 1000 --embeddings_dim 300 --dropout_p 0.1 --num_sm_hidden 100 --batch_size 15 --num_epochs 100 --y_start 2010 --y_end 2018 --dataset 30-fold-8-classes --fold 1 --early_stop True --patience 10 --th 0.0")

#DONE - INTERRUPTED
#experiments_to_run.append(fixed + " --nn_model RCNN --learning_rate 0.001 --max_length 1000 --embeddings_dim 300 --dropout_p 0.1 --num_sm_hidden 100 --batch_size 15 --num_epochs 100 --y_start 2010 --y_end 2018 --dataset 30-fold-8-classes --early_stop True --patience 10 --th 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 --fold 2")

experiments_to_run.append(fixed + "--field wiki-cats --nn_model RCNN --learning_rate 0.001 --max_length 200 --embeddings_dim 300 --dropout_p 0.1 --num_sm_hidden 100 --batch_size 30 --num_epochs 100 --y_start 2010 --y_end 2018 --dataset 30-fold-8-classes --early_stop True --patience 10 --th 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 --fold 1")

experiments_to_run.append(fixed + "--field wiki-cats --nn_model RCNN --learning_rate 0.001 --max_length 1000 --embeddings_dim 300 --dropout_p 0.1 --num_sm_hidden 100 --batch_size 15 --num_epochs 100 --y_start 2010 --y_end 2018 --dataset 30-fold-8-classes --early_stop True --patience 10 --th 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 --fold 1")



for exp in experiments_to_run:
    print("Running experiment with args:\n" + exp)
    args = exp.split(" ")
    subprocess.run(args)

