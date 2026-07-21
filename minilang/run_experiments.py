from minilang import miniLangShuffle
from model import MiniLM
from utils import generate_compositional, generate_greedy, generate
from algorithms import reinforce, grpo_reinforce, soft_reinforce
import argparse
import torch
import numpy as np
import os

def main(args):
    print(f"Comparing Monolithic with Compositional with effective horizon {args.effective_horizon}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device {device}")

    experiment_dir = f"{args.results_dir}/horizon_{args.effective_horizon}"
    os.makedirs(experiment_dir, exist_ok=True)

    # Evaluation Environment
    eval_env = miniLangShuffle(num_blocks=args.num_blocks, block_size=args.block_size)
    eval_obs = eval_env.initialize()

    # Vanilla Reinforce
    vanilla_models = []
    for _ in range(args.num_models):
        vanilla_model = MiniLM(vocab_size=args.num_blocks * args.block_size + 1, embedding_dim=8, hidden_dim=16).to(device)
        vanilla_training_rewards, vanilla_training_greedy_rewards = reinforce(num_blocks=args.num_blocks, block_size=args.block_size, batch_size=32,
                                            model=vanilla_model, num_episodes=args.effective_horizon, lr=1e-3)
        vanilla_models.append(vanilla_model)
        
    vanilla_rewards = np.zeros((args.num_models, args.evals_per_model))
    with open(f"{experiment_dir}/vanilla_eval_logs.txt", "a", encoding="utf-8") as file:
        for i in range(args.num_models * args.evals_per_model):
            vanilla_action = generate_greedy(eval_obs, vanilla_models[i // args.evals_per_model])[0]
            file.write(f"Prompt: {eval_obs}, Response: {vanilla_action}, ")
            vanilla_reward, eval_obs = eval_env.step(vanilla_action)
            file.write(f"Reward: {np.mean(vanilla_reward)}\n")
            vanilla_rewards[i // args.evals_per_model, i % args.evals_per_model] = np.mean(vanilla_reward)

    np.save(f"{experiment_dir}/vanilla_rewards.npy", vanilla_rewards)

    # GRPO Reinforce
    grpo_models = []
    for _ in range(args.num_models):
        grpo_model = MiniLM(vocab_size=args.num_blocks * args.block_size + 1, embedding_dim=8, hidden_dim=16).to(device)
        grpo_training_rewards, grpo_training_greedy_rewards = grpo_reinforce(num_blocks=args.num_blocks, block_size=args.block_size, batch_size=32,
                                                 model=grpo_model, num_episodes=args.effective_horizon, lr=1e-3)
        grpo_models.append(grpo_model)

    grpo_rewards = np.zeros((args.num_models, args.evals_per_model))
    with open(f"{experiment_dir}/grpo_eval_logs.txt", "a", encoding="utf-8") as file:
        for i in range(args.num_models * args.evals_per_model):
            grpo_action = generate_greedy(eval_obs, grpo_models[i // args.evals_per_model])[0]
            file.write(f"Prompt: {eval_obs}, Response: {grpo_action}, ")
            grpo_reward, eval_obs = eval_env.step(grpo_action)
            file.write(f"Reward: {np.mean(grpo_reward)}\n")
            grpo_rewards[i // args.evals_per_model, i % args.evals_per_model] = np.mean(grpo_reward)

    np.save(f"{experiment_dir}/grpo_rewards.npy", grpo_rewards)

    # Compositional Reinforce
    model_0s = []
    for _ in range(args.num_models):
        model_0 = MiniLM(vocab_size=args.num_blocks * args.block_size + 1, embedding_dim=8, hidden_dim=8).to(device)
        model_0_training_rewards, model_0_training_greedy_rewards = soft_reinforce(num_blocks=args.num_blocks, block_size=args.block_size, reward_index=0,
                                                                                   model=model_0, alpha=0.1, batch_size=32,
                                                                                   num_episodes=(args.effective_horizon * 2) // 7, lr=1e-3)
        model_0s.append(model_0)

    
    model_1s = []
    for _ in range(args.num_models):
        model_1 = MiniLM(vocab_size=args.num_blocks * args.block_size + 1, embedding_dim=8, hidden_dim=8).to(device)
        model_1_training_rewards, model_1_training_greedy_rewards = soft_reinforce(num_blocks=args.num_blocks, block_size=args.block_size, reward_index=1,
                                                                                   model=model_1, alpha=0.1, batch_size=32,
                                                                                   num_episodes=(args.effective_horizon * 12) // 7, lr=1e-3)
        model_1s.append(model_1)

    compositional_rewards = np.zeros((args.num_models, args.evals_per_model))
    with open(f"{experiment_dir}/compositional_eval_logs.txt", "a", encoding="utf-8") as file:
        for i in range(args.num_models * args.evals_per_model):
            compositional_action = generate_compositional(eval_obs, model_0s[i // args.evals_per_model], model_1s[i // args.evals_per_model])
            file.write(f"Prompt: {eval_obs}, Response: {compositional_action}, ")
            compositional_reward, eval_obs = eval_env.step(compositional_action)
            file.write(f"Reward: {compositional_reward}\n")
            compositional_rewards[i // args.evals_per_model, i % args.evals_per_model] = np.mean(compositional_reward)

    np.save(f"{experiment_dir}/compositional_rewards.npy", compositional_rewards)

    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--effective-horizon", type=int) # 3500, 7000, 14000, 28000, 56000, 112000, 224000
    parser.add_argument("--block-size", type=int, default=2)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--num-models", type=int, default=32)
    parser.add_argument("--evals-per-model", type=int, default=32)
    parser.add_argument("--results-dir", type=str, default="minilang/results")
    args = parser.parse_args()
    main(args)
