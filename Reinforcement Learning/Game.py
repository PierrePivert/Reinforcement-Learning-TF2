import os
import gym
import shutil
import numpy as np
import pandas as pd
import datetime as dt
from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

from Agent import Agent_DQN

cwd = os.path.abspath(__file__)  # current repertory
path = cwd[:cwd.find('Research')] + 'Research\\MachineLearning\\Exports\\'


class Game(object):
    def __init__(self, custom_agent):
        consecutive_frames = 4
        self.step_penalty = .01
        self.max_steps = 1_000

        """ Initialize Game """
        self.env = gym.make('BreakoutDeterministic-v4')

        """ Initialize Agent """
        self.agent = custom_agent(input_dim=np.repeat(self.preprocess_img(img=self.env.observation_space.sample()), repeats=consecutive_frames, axis=-1).shape,
                                  output_dim=self.env.action_space.n)

        """ Initialize TensorBoard """
        experiment_name = 'Breakout_{}'.format(self.agent.name)

        log_dir = "{}RL\\{}-{}".format(path, dt.datetime.utcnow().strftime("%Y.%m.%d-%H.%M.%S"), experiment_name)
        tf.summary.create_file_writer(log_dir).set_as_default()

        print('=' * 30 + '\n' + 'Reinforcement learning - ' + experiment_name + '\n' + '=' * 30 + '\n')

    def preprocess_img(self, img):
        """
        Preprocess an image by :
        origial size image : 100,928
        - a single chanel (mean across 3 channels) # getsizeof : 268,928
        - as np.uint8 type because it's smallest type # getsizeof : 33,728
        - downsample by /2 height and weight # getsizeof : 128
        - we don't normalize yet since it return float type # getsizeof : 67,328
        """
        return img.mean(axis=-1, keepdims=True).astype(np.uint8)[::2, ::2, :]

    def run_episode(self, training):
        """Returns an episode reward
        (1) Play until the game is done
        (2) The agent will choose an action according to the policy
        (3) When it's done, it will train from the game play
        Args:
            env (gym.env): Gym environment
            agent (Agent): Game Playing Agent
        Returns:
            total_reward (int): total reward earned during the whole episode
        """

        done = False
        loss_list = list([])
        self.agent.reset_metrics()

        steps = 0
        total_reward = 0
        remaining_life = 5  # can set to 0 if unknown
        pre_s = self.env.reset()
        s = np.repeat(self.preprocess_img(pre_s), repeats=self.agent.input_dim[-1], axis=-1)

        """ Run experience """
        while not done and steps < self.max_steps:

            steps += 1
            a = self.agent.get_action(state=s, training=training)

            pre_s2, r, done, info = self.env.step(a)
            s2 = np.append(arr=s[:, :, 1:], values=self.preprocess_img(pre_s2), axis=-1)
            self.env.render() if self.agent.visualize else None

            # Override reward
            terminal_life_lost = True if info['ale.lives'] < remaining_life else done
            r -= self.step_penalty + terminal_life_lost
            total_reward += r

            remaining_life = info['ale.lives']
            s = s2

            """ Remember experience """
            self.agent.memory.remember_experience(experience_list=[[s2[:, :, -1], a, r, terminal_life_lost]])

            """ Experience replay """
            if self.agent.total_frames > self.agent.observe \
                    and self.agent.memory.cpt \
                    and self.agent.total_frames % self.agent.update_freq == 0 \
                    and training:
                loss_list.extend(self.agent.fit(status=done))

            """ Update target network """
            if self.agent.total_frames % self.agent.target_network_update_freq == 0:
                self.agent.target_model.set_weights(self.agent.model.get_weights())

        return total_reward, np.mean(loss_list), steps

    def train(self):

        prev_total_frame = self.agent.total_frames
        pbar = tqdm(total=self.agent.final_frame)

        """ Run experience """
        try:
            while self.agent.total_frames <= self.agent.final_frame:
                pbar.update(self.agent.total_frames - prev_total_frame)
                prev_total_frame = self.agent.total_frames

                self.agent.total_episode += 1

                if self.agent.total_episode % 50 == 0:
                    prefix = 'Test/'
                    print('=' * 100)

                else:
                    prefix = 'Train/'

                """ Episode feedback """
                feedback = dict(zip(['score', 'loss', 'nb_steps'], self.run_episode(training='Train' in prefix)))
                feedback['epsilon'] = self.agent.get_epsilon(training='Train' in prefix)

                """ Display feedback """
                print("Epoch {:03d} | Loss {:.6f} | Score {:.2f} | Epsilon {:.2f} | Frames {:03d} | Total frames {:03d} | Memory size {:03d}".
                      format(self.agent.total_episode,
                             feedback['loss'],
                             feedback['score'],
                             feedback['epsilon'],
                             feedback['nb_steps'],
                             self.agent.total_frames,
                             min(self.agent.memory.memory_size, self.agent.memory.cpt)))

                """ Write on TensorBoard """
                [tf.summary.scalar(name=prefix + k, data=v, step=self.agent.total_frames) for k, v in feedback.items()]
                [tf.summary.scalar(name=prefix + k, data=v, step=self.agent.total_frames) for k, v in self.agent.training_metrics.items()]

                """ Model weights & Actions sample """
                # if agent.total_episode % 10 == 0:
                #     file_writer.add_summary(agent.tensorboard_weights([])[0], global_step=agent.total_frames)
                #     if agent.action_batch is not None:
                #         file_writer.add_summary(agent.tensorboard_action_batch([agent.action_batch])[0], global_step=agent.total_frames)

                """ Export model and data """
                # if agent.total_episode % 1000 == 0:
                #     agent.model.save(log_dir + '\\' + experiment_name + '.h5')  # creates a HDF5 file 'my_model.h5'
                #     pd.DataFrame(tracker).to_csv(log_dir + '\\' + experiment_name + '.csv')

        finally:
            self.env.close()


if __name__ == "__main__":
    game = Game(custom_agent=Agent_DQN)
    game.train()