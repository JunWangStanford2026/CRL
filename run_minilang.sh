cd ~/CRL
conda activate CRL
git pull
nohup python minilang/run_experiments.py --effective-horizon 3500 --block-size 2 --num-blocks 2 --num-models 32 --evals-per-model 32 --results-dir minilang/results &