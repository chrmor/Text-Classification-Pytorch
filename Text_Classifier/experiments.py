import subprocess

experiments_to_run = []
experiments_to_run.append("python main.py --cuda False --load_model False --save_model True --num_epochs 1 --nn_model RCNN --y_start 2017 --y_end 2017 --dataset 30-fold-8-classes --fold 1")
experiments_to_run.append("python main.py --cuda True --load_model False --save_model True --num_epochs 3 --nn_model RCNN --y_start 2010 --y_end 2018 --dataset 30-fold-8-classes --fold 1")

for exp in experiments_to_run:
    print("Running experiment with args:\n" + exp)
    args = exp.split(" ")
    subprocess.run(args)

