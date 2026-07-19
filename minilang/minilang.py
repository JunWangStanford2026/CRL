import numpy as np

class miniLangShuffle():
    '''
    The token set is {0, 1, ..., num_blocks * block_size}
    The token (num_blocks * block_size) is a special token for the purpose of padding
    '''
    def __init__(self, num_blocks, block_size):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.observation = None
        self.optimal_action = None

    def initialize(self, seed=None):
        self.reset(seed)
        return self.observation

    def reset(self, seed=None):
        if seed != None:
            np.random.seed(seed)

        # first block contains the original elements
        self.observation = np.random.choice(self.num_blocks * self.block_size, self.block_size, replace=False)
        self.optimal_action = self.observation.copy()

        for j in range(1, self.num_blocks):
            # every subsequent block contains the shuffled indices of the previous block
            permutation = np.random.permutation(range(self.block_size * (j - 1),
                                                      self.block_size * j))
            self.observation = np.concatenate((self.observation, permutation))
            self.optimal_action = np.concatenate((self.optimal_action, self.optimal_action[permutation]))

        if seed != None:
            print(f"configured environment with prompt {self.observation}, optimal action {self.optimal_action}")

    def inspect_reward(self, action):
        # reward = []

        # for i in range(self.num_blocks):
        #     block = action[i * self.block_size : (i + 1) * self.block_size]

        #     observation_block = self.observation[i * self.block_size : (i + 1) * self.block_size]
        #     if i == 0:
        #         reward.append(np.mean((block == observation_block).astype(int)))
        #     else:
        #         # we still give credit if the reference predictions of the current block
        #         # are consistent with the previous block
        #         reward.append(np.mean((block == action[observation_block]).astype(int)))

        reference_action = self.optimal_action[:self.block_size]
        for j in range(1, self.num_blocks):
            observation_block = self.observation[j * self.block_size : (j + 1) * self.block_size]
            reference_action = np.concatenate((reference_action, action[observation_block]))

        reward = (action == reference_action).astype(float)

        return reward

    def step(self, action):
        if len(action.shape) > 1:
            return self.batched_step(action)

        # reward = []

        # for i in range(self.num_blocks):
        #     block = action[i * self.block_size : (i + 1) * self.block_size]

        #     observation_block = self.observation[i * self.block_size : (i + 1) * self.block_size]
        #     if i == 0:
        #         reward.append(np.mean((block == observation_block).astype(int)))
        #     else:
        #         # we still give credit if the reference predictions of the current block
        #         # are consistent with the previous block
        #         reward.append(np.mean((block == action[observation_block]).astype(int)))

        reference_action = self.optimal_action[:self.block_size]
        for j in range(1, self.num_blocks):
            observation_block = self.observation[j * self.block_size : (j + 1) * self.block_size]
            reference_action = np.concatenate((reference_action, action[observation_block]))

        reward = (action == reference_action).astype(float)

        self.reset()
        return reward, self.observation

    def batched_step(self, actions):
        B = actions.shape[0]
        # reward = np.zeros((B, self.num_blocks))

        # for i in range(self.num_blocks):
        #     block = actions[:, i * self.block_size : (i + 1) * self.block_size]

        #     observation_block = np.tile(self.observation[i * self.block_size : (i + 1) * self.block_size], (B, 1))
        #     if i == 0:
        #         reward[:, i] = np.mean((block == observation_block).astype(int), axis=1)
        #     else:
        #         # we still give credit if the reference predictions of the current block
        #         # are consistent with the previous block
        #         reward[:, i] = np.mean((block == actions[np.arange(B)[:, None], observation_block]).astype(int), axis=1)

        reference_action = np.zeros(actions.shape)
        for j in range(self.num_blocks):
            if j == 0:
                reference_action[:, :self.block_size] = np.tile(self.optimal_action[:self.block_size], (B, 1))
            else:
                observation_block = np.tile(self.observation[j * self.block_size : (j + 1) * self.block_size], (B, 1))
                reference_action[:, j * self.block_size : (j + 1) * self.block_size] = actions[np.arange(B)[:, None], observation_block]

        reward = (actions == reference_action).astype(float)

        self.reset()
        return reward, self.observation


