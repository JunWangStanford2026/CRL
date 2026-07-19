from minilang import miniLangShuffle
from model import MiniLM
from utils import generate_compositional, generate_greedy, generate_grpo, generate_thompson_sampling
from algorithms import reinforce, grpo_reinforce, soft_reinforce
import argparse
import torch
import numpy as np

def main(args):
    print(f"Comparing Monolithic with Compositional with effective horizon {args.effective_horizon}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device {device}")

    # Evaluation Environment
    eval_env = miniLangShuffle(num_blocks=args.num_blocks, block_size=args.block_size)
    eval_obs = eval_env.initialize()

    # Vanilla Reinforce

    # GRPO Reinforce
    grpo_models = []
    for _ in range(args.num_models):
        grpo_model = MiniLM(vocab_size=args.num_blocks * args.block_size + 1, embedding_dim=8, hidden_dim=16).to(device)
        rewards, greedy_rewards = grpo_reinforce(num_blocks=args.num_blocks, block_size=args.block_size, batch_size=32,
                                                 model=grpo_model, num_episodes=args.effective_horizon, lr=1e-3)
        grpo_models.append(grpo_model)

    grpo_rewards = np.zeros((args.num_models, args.evals_per_model))
    for i in range(args.num_models * args.evals_per_model):
        grpo_action = generate_greedy(eval_obs, grpo_models[i // args.evals_per_model])[0]
        grpo_reward, eval_obs = eval_env.step(grpo_action)
        grpo_rewards[i // args.evals_per_model, i % args.evals_per_model] = grpo_reward

    # Compositional Reinforce



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--effective-horizon", type=int) # 3500, 7000, 14000, 28000, 56000, 112000, 224000
    parser.add_argument("--block-size", type=int, default=2)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--num-models", type=int, default=32)
    parser.add_argument("--evals-per-model", type=int, default=32)
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()
    main(args)
