cd ~/CRL
conda activate CRL
nohup python minilang/run_experiments.py --effective-horizon 3500 --block-size 2 --num-blocks 2 --num-models 32 --evals-per-model 32 --results-dir minilang/results &
git add .
git commit -m "HRA 3500 Results"
git push origin minilang/cloud
nohup python minilang/run_experiments.py --effective-horizon 7000 --block-size 2 --num-blocks 2 --num-models 32 --evals-per-model 32 --results-dir minilang/results &
git add .
git commit -m "HRA 7000 Results"
git push origin minilang/cloud
nohup python minilang/run_experiments.py --effective-horizon 14000 --block-size 2 --num-blocks 2 --num-models 32 --evals-per-model 32 --results-dir minilang/results &
git add .
git commit -m "HRA 14000 Results"
git push origin minilang/cloud
nohup python minilang/run_experiments.py --effective-horizon 28000 --block-size 2 --num-blocks 2 --num-models 32 --evals-per-model 32 --results-dir minilang/results &
git add .
git commit -m "HRA 28000 Results"
git push origin minilang/cloud
nohup python minilang/run_experiments.py --effective-horizon 56000 --block-size 2 --num-blocks 2 --num-models 32 --evals-per-model 32 --results-dir minilang/results &
git add .
git commit -m "HRA 56000 Results"
git push origin minilang/cloud
nohup python minilang/run_experiments.py --effective-horizon 112000 --block-size 2 --num-blocks 2 --num-models 32 --evals-per-model 32 --results-dir minilang/results &
git add .
git commit -m "HRA 112000 Results"
git push origin minilang/cloud
nohup python minilang/run_experiments.py --effective-horizon 224000 --block-size 2 --num-blocks 2 --num-models 32 --evals-per-model 32 --results-dir minilang/results &
git add .
git commit -m "HRA 224000 Results"
git push origin minilang/cloud