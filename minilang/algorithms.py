import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import generate, generate_greedy, generate_hra, generate_greedy_hra
from minilang import miniLangShuffle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def reinforce(num_blocks, block_size, model, num_episodes=1000, lr=0.001, batch_size=32):
    '''
    Update rule: theta += lr * reward * grad(log_prob(response))
    returns: a list of rewards obtained in each episode, greedy and stochastic
    '''
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    env = miniLangShuffle(num_blocks, block_size)
    observation = env.initialize()
    greedy_rewards = []
    rewards = []

    for episode in range(num_episodes):
        optimizer.zero_grad()
        response, log_prob = generate(observation, model, batch_size) # (B, N), (B, N)
        greedy_response, greedy_log_prob = generate_greedy(observation, model)
        greedy_reward = np.mean(env.inspect_reward(greedy_response))
        greedy_rewards.append(greedy_reward)
        reward, observation = env.step(response) # reward is (B, N)
        rewards.append(np.mean(reward))
        loss = -torch.sum(torch.tensor(reward) * log_prob)  # negative for gradient ascent
        loss.backward()
        optimizer.step()

    return rewards, greedy_rewards


def grpo_reinforce(num_blocks, block_size, model, num_episodes=1000, lr=0.001, batch_size=32):
    '''
    Update rule: theta += lr * (reward - GRPO Baseline) * grad(log_prob(response))
    returns: a list of rewards obtained in each episode, greedy and stochastic
    '''
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    env = miniLangShuffle(num_blocks, block_size)
    observation = env.initialize()
    greedy_rewards = []
    rewards = []

    for episode in range(num_episodes):
        optimizer.zero_grad()
        response, log_prob = generate(observation, model, batch_size) # (B, N), (B, N)
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


def hra_reinforce(num_blocks, block_size, model, alpha=0.1, num_reward_components=2, batch_size=32, num_episodes=1000, lr=1e-4):
    '''
    Update rule: per reward component, theta += lr * (reward - GRPO Baseline) * grad(log_prob(response))
    returns: a list of rewards obtained in each episode, greedy and stochastic
    '''
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    env = miniLangShuffle(num_blocks, block_size)
    observation = env.initialize()
    greedy_rewards = []
    rewards = []

    for episode in range(num_episodes):
        optimizer.zero_grad()
        response, log_prob = generate_hra(observation, model, num_reward_components, batch_size) # (B, N), (B, num_reward_components, N)
        greedy_response, greedy_log_prob = generate_greedy_hra(observation, model, num_reward_components)
        greedy_reward = np.mean(env.inspect_reward(greedy_response))
        greedy_rewards.append(greedy_reward)
        reward, observation = env.step(response) # reward is (B, N)
        rewards.append(np.mean(reward))
        reward_by_head =  np.zeros((batch_size, num_reward_components, response.shape[1])) # (B, num_reward_components, N)

        for head in range(num_reward_components):
            reward_by_head[:, head, head * block_size : (head + 1) * block_size] = reward[:, head * block_size : (head + 1) * block_size]

        baselined_reward = reward_by_head - ((np.sum(reward_by_head, axis=0, keepdims=True) - reward_by_head) / (batch_size - 1))
        baselined_reward = torch.tensor(baselined_reward, dtype=torch.float, requires_grad=False).to(device)
        loss = -torch.sum(baselined_reward * log_prob.to(device)) # negative for gradient descent
        loss.backward()
        optimizer.step()

    return rewards, greedy_rewards
        



def soft_reinforce(num_blocks, block_size, reward_index, model, alpha=0.1, batch_size=32, num_episodes=1000, lr=0.001, seed=None):
    '''
    Update rule: theta += lr * (reward - alpha * log_prob(response)) - soft GRPO baseline) * grad(log_prob(response))
    returns: a list of rewards obtained in each episode, greedy and stochastic
    '''
    assert reward_index in range(num_blocks)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    env = miniLangShuffle(num_blocks, block_size)
    observation = env.initialize(seed=seed)
    greedy_rewards = []
    rewards = []

    for episode in range(num_episodes):
        optimizer.zero_grad()
        response, log_prob = generate(observation, model, batch_size) # (B, N), (B, N)
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



