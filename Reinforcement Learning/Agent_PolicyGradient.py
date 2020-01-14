# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import os
import gym
import shutil
import numpy as np
import pandas as pd
import datetime as dt
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


class Agent_Policy(Agent.Agent):

    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim)

        """ Build functions """
        self.__build_network(input_dim, output_dim)
        self.__build_train_fn()

        self.build_tensorboard_weights({'Policy': self.model})
        self.build_tensorboard_action_batch()

        """ Parameters """
        self.memory_size = 5_000
        self.feedback_maxProb = 0

        """ Pre sets """
        self.memory = ExperienceReplay.ExperienceReplay(memory_size=self.memory_size, input_dim=input_dim)

    def __build_network(self, input_dim, output_dim):
        frames_input = Input(shape=input_dim, name='frames')

        normalized = Lambda(lambda x: x / 255.0, name='normalization')(frames_input)
        conv_1 = Conv2D(filters=32, kernel_size=8, strides=4, activation=relu, kernel_initializer=VarianceScaling(scale=2), name='conv_1')(normalized)
        conv_2 = Conv2D(filters=64, kernel_size=4, strides=2, activation=relu, kernel_initializer=VarianceScaling(scale=2), name='conv_2')(conv_1)
        conv_3 = Conv2D(filters=64, kernel_size=3, strides=1, activation=relu, kernel_initializer=VarianceScaling(scale=2), name='conv_3')(conv_2)
        flat = Flatten(name='flat')(conv_3)

        dense = Dense(units=1024, activation=relu, name='dense')(flat)
        output = Dense(units=output_dim, activation=softmax, name='logits')(dense)

        """ Network """
        self.model = Model(inputs=frames_input, outputs=output)
        self.model.compile(optimizer=Adam(lr=self.learning_rate), loss=categorical_crossentropy)
        self.model.summary()

    def __build_train_fn(self):
        """Create a train function
        It replaces `model.fit(X, y)` because we use the output of model and use it for training.
        For example, we need action placeholder
        called `action_one_hot` that stores, which action we took at state `s`.
        Hence, we can update the same action.
        This function will create
        `self.train_fn([state, action_one_hot, discount_reward])`
        which would train the model.
        """
        # """ Placeholders """
        # input_placeholder = K.placeholder(shape=(None, *self.input_dim), name='model_inputs')
        # actions_placeholder = K.placeholder(shape=(None,), dtype='uint8', name="selected_actions")
        # rewards_placeholder = K.placeholder(shape=(None,), name="discount_reward")
        #
        # """ Internal operations """
        # output_tensor = self.model(input_placeholder)
        # action_onehot_placeholder = K.one_hot(indices=actions_placeholder, num_classes=self.output_dim)
        #
        # action_prob = K.sum(output_tensor * action_onehot_placeholder, axis=1)
        # log_action_prob = K.log(action_prob + K.epsilon())
        #
        # loss = - log_action_prob * rewards_placeholder
        # total_loss = K.mean(loss)
        #
        # nn_train = K.function(inputs=[input_placeholder, actions_placeholder, rewards_placeholder],
        #                       outputs=[total_loss],
        #                       updates=Adam(lr=self.learning_rate).get_updates(total_loss, self.model.trainable_weights))
        # #                       updates=Adam(lr=5e-6, beta_1=0.5, beta_2=0.999).get_updates(total_loss, self.model.trainable_weights))
        #
        # self.train_fn = nn_train
        # ==================================================================================================================================================================================================================================
        """ Placeholders """
        observations_placeholder = K.placeholder(shape=(None, *self.input_dim), name='model_inputs')
        actions_placeholder = K.placeholder(shape=(None,), dtype='uint8', name="selected_actions")
        rewards_placeholder = K.placeholder(shape=(None,), name="discounted_rewards")

        """ Internal operations """
        Ylogits = self.model(observations_placeholder)

        cross_entropies = K.categorical_crossentropy(target=K.one_hot(indices=actions_placeholder, num_classes=self.output_dim),
                                                     output=Ylogits,
                                                     from_logits=True)  # from_logits=False
        loss = K.mean(rewards_placeholder * cross_entropies)

        nn_train = K.function(inputs=[observations_placeholder, actions_placeholder, rewards_placeholder],
                              outputs=[loss],
                              updates=Adam(lr=self.learning_rate).get_updates(loss, self.model.trainable_weights))  # RMSprop().get_updates(loss=loss, params=self.model.trainable_weights)

        self.train_fn = nn_train

    def fit(self):
        """ ~ """
        S, A, R, S_prime, game_over = self.memory.get_batch(batch_size=self.memory.cpt,
                                                            deterministic_idx=np.array(range(min(self.memory.cpt, self.memory.memory_size))))
        self.memory.cpt = 0

        discounted_rewards = self.discount_rewards(R, game_over)
        action_onehot = np_utils.to_categorical(A, num_classes=self.output_dim)

        """ Train on batch """
        # train_fn
        loss = self.train_fn([S, A, discounted_rewards])

        # fit method - one_hot rewards
        # history = self.model.fit([S], action_onehot * discounted_rewards[:, None], epochs=1, verbose=0)
        # loss = history.history['loss'][0]

        # fit method - gradients
        # probs = self.model.predict([S])
        # gradients = (action_onehot - probs) * discounted_rewards[:, None]
        # targets = probs + self.learning_rate * gradients
        #
        # history = self.model.fit([S], targets, epochs=1, verbose=0)
        # loss = history.history['loss'][0]

        """ Store states batch """
        self.action_batch = self.model.predict([S])

        return loss

    def get_action(self, state, playing):
        """ ~ """

        """ Get action """
        if np.random.rand() < self.get_epsilon(playing):
            action = np.random.randint(self.output_dim)
        else:
            action_prob = np.squeeze(self.model.predict([[state]]))
            action = np.random.choice(np.arange(self.output_dim), p=action_prob)

            self.feedback_maxProb = max(action_prob.max(), self.feedback_maxProb)

        return action

    def discount_rewards(self, rewards, game_over):
        """ ~ """
        discounted_r = np.zeros_like(rewards, dtype=np.float32)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = rewards[t] + self.gamma * running_add * (1 - game_over[t])
            discounted_r[t] = running_add

        norm_discounted_r = (discounted_r - discounted_r.mean()) / (discounted_r.std() + K.epsilon())

        return norm_discounted_r


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


def run_episode(env, agent, playing):
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
    experience_list = list([])
    agent.feedback_maxProb = 0

    steps = 0
    total_reward = 0
    remaining_life = 5
    pre_s = env.reset()
    s = np.repeat(preprocess_img(pre_s), repeats=agent.input_dim[-1], axis=-1)

    """ Run experience """
    while not done:

        steps += 1
        agent.total_frames += 1
        a = agent.get_action(state=s, playing=playing)

        pre_s2, r, done, info = env.step(a)
        s2 = np.append(arr=s[:, :, 1:], values=preprocess_img(pre_s2), axis=-1)
        env.render() if agent.visualize else None
        total_reward += r
        # penalty = remaining_life - info['ale.lives']
        terminal_life_lost = True if info['ale.lives'] < remaining_life else done

        experience_list.append([s2[:, :, -1], a, r, terminal_life_lost])  # r - penalty
        remaining_life = info['ale.lives']
        s = s2

        """ Update target network """
        if agent.total_frames % agent.target_network_update == 0:
            agent.target_model.set_weights(agent.model.get_weights())

    """ Remember experience """
    agent.memory.remember_experience(experience_list=experience_list)

    """ Experience replay """
    if agent.total_frames > agent.observe and agent.memory.cpt:
        loss_list.append(agent.fit())  # one last update : required for policy agent

    return total_reward, np.mean(loss_list), steps


# ==========================================================================================================================================================================================================================================
print('=' * 30 + '\n' + 'Reinforcement learning \n' + '=' * 30 + '\n')

consecutive_frames = 4
env = gym.make('BreakoutDeterministic-v4')
experiment_name = 'Breakout_PolicyGradient'

""" Initialize agent """
input_dim = np.repeat(preprocess_img(env.observation_space.sample()), repeats=consecutive_frames, axis=-1).shape
output_dim = env.action_space.n
agent = Agent_Policy(input_dim, output_dim)

tracker = list([])

""" Initialize TensorBoard """
log_dir = "{}{}-{}".format(path, dt.datetime.utcnow().strftime("%Y.%m.%d-%H.%M.%S"), experiment_name)
file_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())

""" Save copy of code """
shutil.copy(src=os.path.abspath(__file__), dst=log_dir)

""" Run experience """
prev_total_frame = agent.total_frames
pbar = tqdm(total=agent.final_frame)
try:
    while agent.total_frames <= agent.final_frame:
        pbar.update(agent.total_frames - prev_total_frame)
        prev_total_frame = agent.total_frames
        agent.total_episode += 1

        if agent.total_episode % 100 == 0:
            """ Testing model """
            suffix = '_test'
            print('=' * 100)

        else:
            """ Training model """
            suffix = ''

        """ Episode feedback """
        feedback = dict(zip(['score' + suffix, 'loss' + suffix, 'nb_steps' + suffix],
                            run_episode(env=env, agent=agent, playing=bool(suffix))))

        """ Display feedback """
        tracker.append(feedback)
        print("Epoch {:03d} | Loss {:.6f} | Score {:.2f} | Epsilon {:.2f} % | Frames {:03d} | Total frames {:03d} | Memory size {:03d}".
              format(agent.total_episode,
                     feedback['loss' + suffix],
                     feedback['score' + suffix],
                     100 * agent.get_epsilon(playing=bool(suffix)),
                     feedback['nb_steps' + suffix],
                     agent.total_frames,
                     min(agent.memory.memory_size, agent.memory.cpt)))

        """ Write on TensorBoard """
        if not bool(suffix):
            file_writer.add_summary(summary=agent.tensorboard_summary(dict_values={'loss': feedback['loss'],
                                                                                   'score': feedback['score'],
                                                                                   'steps': feedback['nb_steps'],
                                                                                   'maxProb': agent.feedback_maxProb}),
                                    global_step=agent.total_frames)

            # Model weights & Actions sample
            if agent.total_episode % 10 == 0:
                file_writer.add_summary(agent.tensorboard_weights([])[0], global_step=agent.total_frames)
                if agent.action_batch is not None:
                    file_writer.add_summary(agent.tensorboard_action_batch([agent.action_batch])[0], global_step=agent.total_frames)

        """ Export model and data """
        if agent.total_episode % 1000 == 0:
            agent.model.save(log_dir + '\\' + experiment_name + '.h5')  # creates a HDF5 file 'my_model.h5'
            pd.DataFrame(tracker).to_csv(log_dir + '\\' + experiment_name + '.csv')

finally:
    env.close()
