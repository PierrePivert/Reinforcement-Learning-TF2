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

from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten, Input, Multiply, Add, Subtract, Lambda
from tensorflow.keras.losses import mean_squared_error, logcosh, categorical_crossentropy
from tensorflow.keras.activations import softmax, relu, linear, elu
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras import backend as K
from tensorflow.keras import utils as np_utils

from Memory import UniformExperienceReplay


class Policy:
    def __init__(self, type, params, observe=0, final_frame=10_000):

        if type == 'linear':
            pass
        elif type == 'bi_linear':

            """ Epsilon slope """
            self.slope_1 = (params[1] - params[0]) / params[-1]
            self.slope_2 = (params[2] - params[1]) / (final_frame - params[-1])
            self.offset_2 = observe - params[-1]

            self.get_epsilon = self.bi_linear

        elif type == 'sigmoid':
            pass
            # f(n) = 1/ (1+e**(-n))

    def bi_linear(self, idx_frame, training):
        if not training:
            return .01
        elif idx_frame < self.observe:
            return 1
        elif idx_frame < self.observe + self.epsilon_policy[-1]:
            return self.epsilon_policy[0] + self.slope_1 * (self.total_frames - self.observe)
        else:
            return self.epsilon_policy[1] + self.slope_2 * (self.total_frames - self.offset_2)


class Agent:

    def __init__(self, input_dim, output_dim):
        """Gym Playing Agent
        Args:
            input_dim (int): the dimension of state.
                Same as `env.observation_space.shape[0]`
            output_dim (int): the number of discrete actions
                Same as `env.action_space.n`
            hidden_dims (list): hidden dimensions
        Methods:
            private:
                __build_train_fn -> None
                    It creates a train function
                    It's similar to defining `train_op` in Tensorflow
                __build_network -> None
                    It create a base model
                    Its output is each action probability
            public:
                get_action(state) -> action
                fit(state, action, reward) -> None
        """
        """ Parameters """
        self.final_frame = 3_000_000
        self.epsilon_policy = [1, .1, 0, 2_000_000]
        self.gamma = .99
        self.learning_rate = .000_5
        self.observe = 1_000
        self.visualize = False
        self.update_freq = 1

        """ Pre sets """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.total_frames = 0
        self.total_episode = 0
        # self.action_batch = None

        """ Epsilon slope """
        self.slope_1 = (self.epsilon_policy[1] - self.epsilon_policy[0]) / self.epsilon_policy[-1]
        self.slope_2 = (self.epsilon_policy[2] - self.epsilon_policy[1]) / (self.final_frame - self.epsilon_policy[-1])
        self.offset_2 = self.observe - self.epsilon_policy[-1]

        """ Additional parameters for other architectures """
        self.name = ''
        self.batch_size = None
        self.update_freq = None
        self.target_network_update_freq = None
        self.memory = None
        self.training_metrics = {}
        # self.feedback_maxQ = None
        # self.feedback_maxProb = None

    def get_epsilon(self, training):
        if not training:
            return .01
        elif self.total_frames < self.observe:
            return 1
        elif self.total_frames < self.observe + self.epsilon_policy[-1]:
            return self.epsilon_policy[0] + self.slope_1 * (self.total_frames - self.observe)
        else:
            return self.epsilon_policy[1] + self.slope_2 * (self.total_frames - self.offset_2)

    # def tensorboard_summary(self, dict_values):
    #     return tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value) for key, value in dict_values.items()])
    #
    # def build_tensorboard_weights(self, networks):
    #     parameters_list = []
    #     with tf.name_scope('Parameters'):
    #         for key in networks.keys():
    #             with tf.name_scope(key):
    #                 parameters_list.extend([[tf.summary.histogram(networks[key].weights[iii].name, tf.reshape(networks[key].weights[iii], shape=[-1]))]
    #                                         for iii in range(len(networks[key].weights))])
    #     tensorboard_summary = tf.summary.merge(parameters_list)
    #
    #     self.tensorboard_weights = K.function(inputs=[],
    #                                           outputs=[tensorboard_summary],
    #                                           updates=None)
    #
    # def build_tensorboard_action_batch(self):
    #     input_placeholder = K.placeholder(shape=(None, self.output_dim), name='action_batch')
    #
    #     with tf.name_scope('Actions'):
    #         actions_list = [tf.summary.histogram(str(a), tf.reshape(tensor=tf.gather(params=input_placeholder,
    #                                                                                  indices=[a],
    #                                                                                  axis=-1),
    #                                                                 shape=[-1])) for a in range(self.output_dim)]
    #     tensorboard_summary = tf.summary.merge(actions_list)
    #
    #     self.tensorboard_action_batch = K.function(inputs=[input_placeholder],
    #                                                outputs=[tensorboard_summary],
    #                                                updates=None)


class Agent_DQN(Agent):
    """
    Implemented :
    - Deep Q-learning : 3 Conv2D
    - Target network
    - Double DQN: predict action from model and compute value from target_model
    - Dueling : split value & advantage

    * replay buffer
    * huber loss (logcosh)
    * action_space : 4
    * no reward at end of episode
    * activation funcion : ReLu
    * gamma : 0.99
    * image_processing : mean(axis)
    * experience_replay_v2 : no end of episode, no current state
    * action_getter_v2 : pre_compute slope
    * tensorflow vs Keras : Keras
    * update_frequence : 4
    ... negative R at end of episode
    ... initializer
    ... batch_normalization
    ... target_network update with low LR
    ... soft Bellman equation
    ... prioritized replay
    ... curiosity model : noise of prediction for exploration
    """

    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim)

        """ Parameters """
        self.name = 'Dueling_DDQN'
        self.batch_size = 32
        self.update_freq = 1
        self.target_network_update_freq = 500

        self.training_metrics['maxQ'] = 0
        self.memory = UniformExperienceReplay(memory_size=50_000, input_dim=input_dim)

        """ Build functions """
        self.__build_network(input_dim, output_dim)
        # self.__build_train_fn()

        # self.build_tensorboard_weights({'DQN': self.model})
        # self.build_tensorboard_action_batch()

    def __build_network(self, input_dim, output_dim):
        """
        Trick : we use a action_input mask thus when computing the loss, only the loss on action choosen is returned.
        Smarter way to compute the loss (instead of computing all loss and selecting the one afterward)
        """
        frames_input = Input(shape=input_dim, name='frames')
        actions_input = Input(shape=(output_dim,), name='mask')

        normalized = Lambda(lambda x: x / 255.0, name='normalization')(frames_input)
        conv_1 = Conv2D(filters=32, kernel_size=8, strides=4, activation=relu, kernel_initializer=VarianceScaling(scale=2), name='conv_1')(normalized)
        conv_2 = Conv2D(filters=64, kernel_size=4, strides=2, activation=relu, kernel_initializer=VarianceScaling(scale=2), name='conv_2')(conv_1)
        conv_3 = Conv2D(filters=64, kernel_size=3, strides=1, activation=relu, kernel_initializer=VarianceScaling(scale=2), name='conv_3')(conv_2)
        flat = Flatten(name='flat')(conv_3)

        dense = Dense(units=1024, activation=relu, name='dense')(flat)
        value_advantage = Dense(units=output_dim + 1, activation=linear, name='value_advantage')(dense)
        output = Lambda(lambda i: K.expand_dims(i[:, 0], -1) + i[:, 1:] - K.mean(i[:, 1:], keepdims=True),
                        output_shape=(output_dim,),
                        name='output')(value_advantage)

        # value = Dense(units=1, activation=linear, name='value')(dense)
        # advantage = Dense(units=output_dim, activation=linear, name='advantage')(dense)
        # average_advantage = Lambda(lambda i: K.mean(i, axis=1, keepdims=True), name='average_advantage')(advantage)
        # net_advantage = Subtract(name='net_advantage')([advantage, average_advantage])
        # output = Add(name='output')([value, net_advantage])

        filtered_output = Multiply(name='Qvalue')([output, actions_input])

        """ Base network """
        self.model = Model(inputs=[frames_input, actions_input], outputs=filtered_output)
        self.model.compile(optimizer=Adam(lr=self.learning_rate), loss=logcosh)
        self.model.summary()

        """ Target network """
        self.target_model = clone_model(model=self.model)
        self.target_model.set_weights(self.model.get_weights())

    def fit(self, status=None):
        """Train a network
        Args:
            S (2-D Array): `state` array of shape (n_samples, state_dimension)
            A (1-D Array): `action` array of shape (n_samples,)
                It's simply a list of int that stores which actions the agent chose
            R (1-D Array): `reward` array of shape (n_samples,)
                A reward is given after each action.
        """

        """ Pull random samples """
        S, A, R, S_prime, game_over = self.memory.get_batch(batch_size=self.batch_size)

        """ Convert actions to one_hot """
        action_onehot = np_utils.to_categorical(A, num_classes=self.output_dim, dtype='uint8')

        """ Compute target Q_values """
        values = self.model.predict([S_prime, np.ones_like(action_onehot)])
        selected_action = np_utils.to_categorical(values.argmax(axis=1), num_classes=self.output_dim)

        target_values = self.target_model.predict([S_prime, np.ones_like(action_onehot)])
        value_action = (target_values * selected_action).sum(axis=1)  # Double DQN :select action from model, update value from target
        # value_action = target_values.max(axis=1)  # Target DQN : action & value from target (less variance)
        # value_action = values.max(axis=1)  # DQN : action & value from model

        Q_values = R + self.gamma * value_action * (1 - game_over)
        # Q_values [game_over] = -1

        """ Train on batch """
        history = self.model.fit([S, action_onehot], action_onehot * Q_values[:, None], epochs=1, verbose=0)
        loss = history.history['loss'][0]

        """ Store action batch """
        # self.action_batch = self.model.predict([S, np.ones_like(action_onehot)])

        return [loss]

    def get_action(self, state, training):
        """Returns an action at given `state`
        Args:
            state (1-D or 2-D Array): It can be either 1-D array of shape (state_dimension, )
                or 2-D array shape of (n_samples, state_dimension)
        Returns:
            action: an integer action value ranging from 0 to (n_actions - 1)
        """
        self.total_frames += 1

        """ Get action """
        if np.random.rand() < self.get_epsilon(training):
            action = np.random.randint(self.output_dim)
        else:
            # action_prob = np.squeeze(self.model.predict([[state]]))
            q_value = np.squeeze(self.model.predict([np.array([state]), np.array([np.ones(self.output_dim)])]))
            self.training_metrics['maxQ'] = max(q_value.max(), self.training_metrics['maxQ'])

            # action = np.random.choice(np.arange(self.output_dim), p=action_prob)
            action = q_value.argmax()

        return action

    def reset_metrics(self):
        self.training_metrics['maxQ'] = 0


class Agent_Policy(Agent):

    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim)

        """ Parameters """
        self.name = 'PolicyGradient'

        self.training_metrics['maxProb'] = 0
        self.memory = UniformExperienceReplay(memory_size=5_000, input_dim=input_dim)

        """ Build functions """
        self.__build_network(input_dim, output_dim)
        self.__build_train_fn()

        # self.build_tensorboard_weights({'Policy': self.model})
        # self.build_tensorboard_action_batch()

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

    def fit(self, status=False):

        if status:
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

            return [loss]
        else:
            return []

    def get_action(self, state, playing):

        self.total_frames += 1

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

    def reset_metrics(self):
        self.training_metrics['maxProb'] = 0
