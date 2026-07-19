import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import generate_thompson_sampling, generate_grpo, generate_greedy
from minilang import miniLangShuffle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def reinforce(num_blocks, block_size, model, policy=generate_thompson_sampling, num_episodes=1000, lr=0.001):
    '''
    Update rule: theta += lr * reward * grad(log_prob(response))
    returns: a list of rewards obtained in each episode
    '''
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    env = miniLangShuffle(num_blocks, block_size)
    observation = env.initialize()
    rewards = []

    for episode in tqdm(range(num_episodes)):
        optimizer.zero_grad()
        response, log_prob = policy(observation, model)
        reward, observation = env.step(response)
        # reward = np.sum(reward)
        rewards.append(np.mean(reward))
        loss = -torch.tensor(reward).dot(log_prob)  # negative for gradient ascent
        loss.backward()
        optimizer.step()

    return rewards


def grpo_reinforce(num_blocks, block_size, model, num_episodes=1000, lr=0.001, batch_size=16):
    '''
    Update rule: theta += lr * (reward - GRPO Baseline) * grad(log_prob(response))
    returns: a list of rewards obtained in each episode
    '''
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    env = miniLangShuffle(num_blocks, block_size)
    observation = env.initialize()
    greedy_rewards = []
    rewards = []

    for episode in tqdm(range(num_episodes)):
        optimizer.zero_grad()
        response, log_prob = generate_grpo(observation, model, batch_size) # (B, N), (B, N)
        greedy_response, greedy_log_prob = generate_greedy(observation, model)
        greedy_reward = np.mean(env.inspect_reward(greedy_response))
        greedy_rewards.append(greedy_reward)
        reward, observation = env.step(response) # reward is (B, N)
        rewards.append(np.mean(reward))
        baselined_reward = reward - (np.sum(reward, axis=0) - reward) / (batch_size - 1)
        baselined_reward = torch.tensor(baselined_reward, dtype=torch.float, requires_grad=False).to(device)
        loss = -torch.sum(baselined_reward * log_prob)  # negative for gradient ascent
        loss.backward()
        optimizer.step()

    return rewards, greedy_rewards


def soft_reinforce(num_blocks, block_size, reward_index, model, alpha=0.1, batch_size=16, num_episodes=1000, lr=0.001, seed=None):
    '''
    Update rule: theta += lr * (reward - alpha * (log_prob(response) + 1)) * grad(log_prob(response))
    returns: a list of rewards obtained in each episode
    '''
    assert reward_index in range(num_blocks)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    env = miniLangShuffle(num_blocks, block_size)
    observation = env.initialize(seed=seed)
    greedy_rewards = []
    rewards = []

    for episode in tqdm(range(num_episodes)):
        optimizer.zero_grad()
        response, log_prob = generate_grpo(observation, model, batch_size) # (B, N), (B, N)
        greedy_response, greedy_log_prob = generate_greedy(observation, model)

        greedy_reward = env.inspect_reward(greedy_response)[reward_index * block_size : (reward_index + 1) * block_size]

        greedy_rewards.append(np.mean(greedy_reward))
        reward, observation = env.step(response) # reward is (B, N)

        reward[:, :reward_index * block_size] = 0.0
        reward[:, (reward_index + 1) * block_size:] = 0.0
        rewards.append(np.mean(reward[:, reward_index * block_size : (reward_index + 1) * block_size]))


        log_prob_numpy = log_prob.clone().detach().cpu().numpy() # (B, N)
        signals = reward - alpha * (log_prob_numpy) # (B, N)
        baselined_signals = signals - (np.sum(signals, axis=0) - signals) / (batch_size - 1)
        baselined_signals = torch.tensor(baselined_signals, dtype=torch.float, requires_grad=False).to(device)
        loss = -torch.sum(baselined_signals * log_prob.to(device))
        loss.backward()
        optimizer.step()

    return rewards, greedy_rewards



