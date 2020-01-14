import numpy as np


class UniformExperienceReplay(object):
    def __init__(self, memory_size, input_dim):
        """
        Implement a ring buffer (FIFO)

        :param memory_size: size of the memory buffer
        :param input_dim: shape of a single observation

        NOTE : since all observation are stored in a list, we don't have to save consecutive frame as it is. We reconstruct them when calling get_batch
        """
        self.cpt = 0
        self.input_dim = input_dim
        self.memory_size = memory_size
        self.memory = {'frames': np.empty(shape=(memory_size, *self.input_dim[:-1]), dtype='uint8'),
                       'actions': np.empty(shape=memory_size, dtype='int8'),
                       'rewards': np.empty(shape=memory_size, dtype='float'),
                       'game_over': np.empty(shape=memory_size, dtype='bool')}

    def remember_experience(self, experience_list):
        """ ... """
        """ Append each time_step to memory """
        for exp in experience_list:
            state_prime, action, reward, game_over = exp

            for key, value in zip(['frames', 'actions', 'rewards', 'game_over'],
                                  [state_prime, action, reward, game_over]):
                self.memory[key][self.cpt % self.memory_size] = value
            self.cpt += 1

    def get_batch(self, batch_size, deterministic_idx=None):
        """
        Get a batch of sample [obs, action, reward, obs_prime, done]
        :param batch_size:
        :param deterministic_idx:
        :return:
        NOTE : obs are reconstructed  from input_dim (stacking consecutive frame as proposed by DeepMind)
        """
        """ Pull random samples """
        sample_idx = np.random.randint(low=0, high=self.cpt, size=batch_size) if deterministic_idx is None else deterministic_idx
        sample_mask = np.empty_like(sample_idx, dtype=bool)

        """ Error handler """
        # trick : discard idx if include end of an episode or include current cpt (2 different episodes)
        ref_modulo = min(self.cpt, self.memory_size)
        for iii in range(len(sample_idx)):
            sample_mask[iii] = not (self.memory['game_over'][(sample_idx[iii] - 1 - np.array(range(self.input_dim[-1]))) % ref_modulo].any()) \
                               and not (sample_idx[iii] - self.input_dim[-1] <= self.cpt % self.memory_size <= sample_idx[iii])
        sample_batch = sample_idx[sample_mask]

        """ Create arrays """
        states = np.empty(shape=(len(sample_batch), *self.input_dim), dtype='uint8')
        states_prime = np.empty(shape=(len(sample_batch), *self.input_dim), dtype='uint8')

        """ Build timeline states """
        for offset in range(self.input_dim[-1]):
            states[:, :, :, offset] = self.memory['frames'][(sample_batch - self.input_dim[-1] + offset) % ref_modulo]
            states_prime[:, :, :, offset] = self.memory['frames'][(sample_batch - self.input_dim[-1] + offset + 1) % ref_modulo]

        actions = self.memory['actions'][sample_batch % ref_modulo]
        rewards = self.memory['rewards'][sample_batch % ref_modulo]
        game_over = self.memory['game_over'][sample_batch % ref_modulo]

        return states, actions, rewards, states_prime, game_over


class PrioritizedExperienceReplay(UniformExperienceReplay):
    def __init__(self, memory_size, input_dim, alpha):
        super().__init__(memory_size, input_dim)

        self.alpha = alpha
        self.memory.update({'prioritization': np.empty(shape=memory_size, dtype='float')})

    def init_prioritization(self):
        return self.memory['prioritization'][:self.cpt].mean()
