cd ~/CRL
conda activate CRL
nohup python minilang/run_experiments.py --effective-horizon 112000 --block-size 2 --num-blocks 2 --num-models 32 --evals-per-model 32 --results-dir minilang/results &