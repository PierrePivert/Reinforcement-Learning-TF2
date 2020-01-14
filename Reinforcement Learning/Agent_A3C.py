# TODO change optimizer for actor & critic




# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import os
import gym
import time
import shutil
import numpy as np
import pandas as pd
import datetime as dt
import threading as th
import tensorflow as tf

import Agent
import ExperienceReplay

from tqdm import tqdm
from keras.models import Model, clone_model
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten, Input, Multiply, Add, Subtract, Lambda
from keras.losses import mean_squared_error, logcosh, categorical_crossentropy
from keras.activations import softmax, relu, linear, elu
from keras.optimizers import Adam, RMSprop
from keras.initializers import VarianceScaling
from keras import backend as K
from keras import utils as np_utils

cwd = os.path.abspath(__file__)  # current repertory
path = cwd[:cwd.find('Research')] + 'Research\\MachineLearning\\Exports\\'


class Agent_A3C(Agent.Agent):

    def __init__(self, input_dim, output_dim):
        Agent.Agent.__init__(self, input_dim, output_dim)

        """ Build functions """
        self.actor_network = build_actor_network(input_dim=input_dim, output_dim=output_dim)
        self.critic_network = build_critic_network(input_dim=input_dim, output_dim=output_dim)

        self.train_fn_actor = self.__build_train_actor()
        self.train_fn_critic = self.__build_train_critic()

        self.build_tensorboard_weights({'Actor': self.actor_network,
                                        'Critic': self.critic_network})
        self.build_tensorboard_action_batch()

        """ Parameters """
        self.nb_workers = 8
        self.update_freq = 10
        self.epsilon_policy = [.01, .001, 0, 1_000_000]

        """ Epsilon slope """
        self.slope_1 = (self.epsilon_policy[1] - self.epsilon_policy[0]) / self.epsilon_policy[-1]
        self.slope_2 = (self.epsilon_policy[2] - self.epsilon_policy[1]) / (self.final_frame - self.epsilon_policy[-1])
        self.offset_2 = self.observe - self.epsilon_policy[-1]

        """ Pre sets """
        self.worker_list = None
        self.memory = ExperienceReplay.ExperienceReplay(memory_size=0, input_dim=input_dim)
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

        """ Initialize TensorBoard """
        if not os.path.isdir(log_dir): self.file_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())
        shutil.copy(src=os.path.abspath(__file__), dst=log_dir)

    def __build_train_actor(self):
        """ ~ """
        """ Placeholders """
        observations_placeholder = K.placeholder(shape=(None, *self.input_dim), name='model_inputs')
        actions_placeholder = K.placeholder(shape=(None,), dtype='uint8', name="selected_actions")
        rewards_placeholder = K.placeholder(shape=(None,), name="discounted_rewards")

        """ Internal operations """
        policy = self.actor_network(observations_placeholder)
        value = K.reshape(x=self.critic_network(observations_placeholder), shape=[-1])
        advantages = rewards_placeholder - value
        action_onehot_placeholder = K.one_hot(indices=actions_placeholder, num_classes=self.output_dim)

        """ Compute loss """
        action_prob = K.sum(policy * action_onehot_placeholder, axis=1)
        log_action_prob = K.log(action_prob + K.epsilon())
        loss = - K.mean(log_action_prob * advantages)

        entropy = K.mean(K.sum(policy * K.log(policy + K.epsilon()), axis=1))

        total_loss = loss + entropy * .1

        """ Train function """
        return K.function(inputs=[observations_placeholder, actions_placeholder, rewards_placeholder],
                          outputs=[total_loss],
                          updates=Adam(lr=self.learning_rate).get_updates(total_loss, self.actor_network.trainable_weights))

    def __build_train_critic(self):
        """ ~ """
        """ Placeholders """
        observations_placeholder = K.placeholder(shape=(None, *self.input_dim), name='model_inputs')
        rewards_placeholder = K.placeholder(shape=(None,), name="discounted_rewards")

        """ Internal operations """
        Q_value = self.critic_network(observations_placeholder)

        """ Compute loss """
        total_loss = K.mean(logcosh(rewards_placeholder, Q_value))

        """ Train function """
        return K.function(inputs=[observations_placeholder, rewards_placeholder],
                          outputs=[total_loss],
                          updates=Adam(lr=self.learning_rate).get_updates(total_loss, self.critic_network.trainable_weights))

    def train(self):
        self.worker_list = [Worker(input_dim=input_dim,
                                   output_dim=output_dim) for _ in range(self.nb_workers)]

        for worker in self.worker_list:
            time.sleep(1)
            worker.start()


class Worker(Agent.Agent, th.Thread):
    def __init__(self, input_dim, output_dim):
        Agent.Agent.__init__(self, input_dim, output_dim)
        th.Thread.__init__(self)

        """ Build functions """
        self.actor_network = build_actor_network(input_dim=input_dim, output_dim=output_dim, weights_from=agent_manager.actor_network)
        self.critic_network = build_critic_network(input_dim=input_dim, output_dim=output_dim, weights_from=agent_manager.critic_network)

        """ Pre sets """
        self.memory = ExperienceReplay.ExperienceReplay(memory_size=10_000, input_dim=input_dim)
        self.feedback_maxProb = 0
        self.feedback_maxQ = 0

    def fit(self):
        """ ~ """
        S, A, R, S_prime, game_over = self.memory.get_batch(batch_size=self.memory.cpt,
                                                            deterministic_idx=np.array(range(min(self.memory.cpt, self.memory.memory_size))))
        self.memory.cpt = 0

        if len(R):
            discounted_rewards = self.discount_rewards(R, game_over, S[-1])

            """ Train on batch """
            loss_actor = agent_manager.train_fn_actor([S, A, discounted_rewards])[0]
            loss_critic = agent_manager.train_fn_critic([S, discounted_rewards])[0]

            """ Store states batch """
            self.action_batch = self.actor_network.predict([S])

            """ Update model """
            self.actor_network.set_weights(agent_manager.actor_network.get_weights())
            self.critic_network.set_weights(agent_manager.critic_network.get_weights())

            return [loss_actor, loss_critic]
        else:
            return

    def get_action(self, state, playing):
        """ ~ """

        """ Get action """
        if np.random.rand() < agent_manager.get_epsilon(playing):
            action = np.random.randint(self.output_dim)
        else:
            action_prob = np.squeeze(self.actor_network.predict([[state]]))
            action = np.random.choice(np.arange(self.output_dim), p=action_prob)

            self.feedback_maxProb = max(action_prob.max(), self.feedback_maxProb)
            self.feedback_maxQ = max(np.squeeze(self.critic_network.predict([[state]])), self.feedback_maxQ)

        return action

    def discount_rewards(self, rewards, game_over, last_state):
        """ ~ """
        discounted_r = np.zeros_like(rewards, dtype=np.float32)
        running_add = 0 if game_over[-1] else self.critic_network.predict([[last_state]])
        for t in reversed(range(len(rewards))):
            running_add = rewards[t] + self.gamma * running_add * (1 - game_over[t])
            discounted_r[t] = running_add

        norm_discounted_r = (discounted_r - discounted_r.mean()) / (discounted_r.std() + K.epsilon())

        return norm_discounted_r

    def run(self):
        # global total_frames
        # global total_episode

        env = gym.make(env_name)

        while agent_manager.total_frames <= agent_manager.final_frame:
            # ==================================================================================================================================================================================================================================
            done = False
            experience_list = list([])
            self.feedback_maxQ = 0
            self.feedback_maxProb = 0

            steps = 0
            agent_manager.total_episode += 1
            total_reward = 0
            loss_list = []
            remaining_life = 5
            pre_s = env.reset()
            s = np.repeat(preprocess_img(pre_s), repeats=self.input_dim[-1], axis=-1)

            """ Run experience """
            while not done:
                steps += 1
                agent_manager.total_frames += 1
                a = self.get_action(state=s, playing=False)

                pre_s2, r, done, info = env.step(a)
                s2 = np.append(arr=s[:, :, 1:], values=preprocess_img(pre_s2), axis=-1)
                env.render() if self.visualize else None
                total_reward += r
                # penalty = remaining_life - info['ale.lives']
                terminal_life_lost = True if info['ale.lives'] < remaining_life else done

                """ Remember experience """
                self.memory.remember_experience(experience_list=[[s2[:, :, -1], a, r, terminal_life_lost]])

                remaining_life = info['ale.lives']
                s = s2

                """ Experience replay """
                if agent_manager.total_frames > self.observe and self.memory.cpt and (steps % agent_manager.update_freq == 0 or done):
                    buffer_loss = self.fit()
                    if buffer_loss is not None: loss_list.extend([buffer_loss])  # one last update : required for policy agent

            # ==================================================================================================================================================================================================================================
            loss_array = np.mean(loss_list, axis=0)
            print("Epoch {:03d} | Score {:.2f} | Epsilon {:.2f} % | Frames {:03d} | Total frames {:03d} | Actor loss {:03f} | Critic loss {:03f}".
                  format(agent_manager.total_episode,
                         total_reward,
                         100 * agent_manager.get_epsilon(playing=False),
                         steps,
                         agent_manager.total_frames,
                         loss_array[0],
                         loss_array[1]))

            """ Write on TensorBoard """
            agent_manager.file_writer.add_summary(summary=agent_manager.tensorboard_summary(dict_values={'loss_actor': loss_array[0],
                                                                                                         'loss_critic': loss_array[1],
                                                                                                         'score': total_reward,
                                                                                                         'steps': steps,
                                                                                                         'maxQ': self.feedback_maxQ,
                                                                                                         'maxProb': self.feedback_maxProb}),
                                                  global_step=agent_manager.total_frames)

            # Model weights & Actions sample
            if agent_manager.total_episode % 10 == 0:
                agent_manager.file_writer.add_summary(agent_manager.tensorboard_weights([])[0], global_step=agent_manager.total_frames)
                if self.action_batch is not None:
                    agent_manager.file_writer.add_summary(agent_manager.tensorboard_action_batch([self.action_batch])[0], global_step=agent_manager.total_frames)


def preprocess_img(img):
    """
    Preprocess an image by :
    origial size image : 100,928
    - a single chanel (mean across 3 channels) # getsizeof : 268,928
    - as np.uint8 type because it's smallest type # getsizeof : 33,728
    - downsample by /2 height and weight # getsizeof : 128
    - we don't normalize yet since it return float type # getsizeof : 67,328
    """
    return img.mean(axis=-1, keepdims=True).astype(np.uint8)[::2, ::2, :]


def build_actor_network(input_dim, output_dim, weights_from=None):
    frames_input = Input(shape=input_dim, name='frames')

    """ Actor network """
    normalized = Lambda(lambda x: x / 255.0, name='normalization')(frames_input)
    conv_1 = Conv2D(filters=32, kernel_size=8, strides=4, activation=relu, kernel_initializer=VarianceScaling(scale=2), name='conv_1')(normalized)
    conv_2 = Conv2D(filters=64, kernel_size=4, strides=2, activation=relu, kernel_initializer=VarianceScaling(scale=2), name='conv_2')(conv_1)
    conv_3 = Conv2D(filters=64, kernel_size=3, strides=1, activation=relu, kernel_initializer=VarianceScaling(scale=2), name='conv_3')(conv_2)
    flat = Flatten(name='flat')(conv_3)

    dense = Dense(units=1024, activation=relu, name='dense')(flat)
    output = Dense(units=output_dim, activation=softmax, name='softmax')(dense)

    actor = Model(inputs=frames_input, outputs=output)
    actor._make_predict_function()

    if weights_from is not None:
        actor.set_weights(weights_from.get_weights())
    else:
        actor.summary()

    return actor


def build_critic_network(input_dim, output_dim, weights_from=None):
    frames_input = Input(shape=input_dim, name='frames')

    """ Critic network"""
    normalized = Lambda(lambda x: x / 255.0, name='normalization')(frames_input)
    conv_1 = Conv2D(filters=32, kernel_size=8, strides=4, activation=relu, kernel_initializer=VarianceScaling(scale=2), name='conv_1')(normalized)
    conv_2 = Conv2D(filters=64, kernel_size=4, strides=2, activation=relu, kernel_initializer=VarianceScaling(scale=2), name='conv_2')(conv_1)
    conv_3 = Conv2D(filters=64, kernel_size=3, strides=1, activation=relu, kernel_initializer=VarianceScaling(scale=2), name='conv_3')(conv_2)
    flat = Flatten(name='flat')(conv_3)

    dense = Dense(units=1024, activation=relu, name='dense')(flat)
    output = Dense(units=1, activation=linear, name='value')(dense)

    critic = Model(inputs=frames_input, outputs=output)
    critic._make_predict_function()

    if weights_from is not None:
        critic.set_weights(weights_from.get_weights())
    else:
        critic.summary()

    return critic


# ==========================================================================================================================================================================================================================================
print('=' * 30 + '\n' + 'Reinforcement learning \n' + '=' * 30 + '\n')

consecutive_frames = 4
env_name = 'BreakoutDeterministic-v4'
experiment_name = 'Breakout_A3C'

env = gym.make(env_name)

log_dir = "{}{}-{}".format(path, dt.datetime.utcnow().strftime("%Y.%m.%d-%H.%M.%S"), experiment_name)

""" Initialize agent """
input_dim = np.repeat(preprocess_img(env.observation_space.sample()), repeats=consecutive_frames, axis=-1).shape
output_dim = env.action_space.n
env.close()

agent_manager = Agent_A3C(input_dim, output_dim)
agent_manager.train()

# agent.actor.save(log_dir + '\\' + experiment_name + '_a.h5')  # creates a HDF5 file 'my_model.h5'
# agent.critic.save(log_dir + '\\' + experiment_name + '_c.h5')  # creates a HDF5 file 'my_model.h5'
# pd.DataFrame(tracker).to_csv(log_dir + '\\' + experiment_name + '.csv')
