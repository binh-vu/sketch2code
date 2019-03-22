#!/usr/bin/python
# -*- coding: utf-8 -*-
import random
from collections import deque
from typing import Tuple

import numpy as np
import tensorflow as tf


class ExperienceBuffer:

    def __init__(self, buffer_size: int = 5000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:len(experience) + len(self.buffer) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])


class DQNAgent(object):
    def __init__(self, state_size: int, action_size: int, image_shape: Tuple[int, ...]):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=2000)
        self.gamma = 1.0

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        self.model = self._build_model(image_shape)

    def _build_model(self, image_shape: Tuple[int, ...]):
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(5e-4, self.global_step, 500, 0.96, staircase=True)
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        self.gstate = tf.placeholder(tf.float32, [None] + list(image_shape))
        self.pstate = tf.placeholder(tf.float32, [None] + list(image_shape))

        with tf.variable_scope('gconv1'):
            self.gconv1 = conv2d(self.gstate, 7, 1, 32)
            self.gconv1_norm = tf.layers.batch_normalization(self.gconv1, momentum=0.9, training=self.is_training)
            self.grelu1 = tf.nn.selu(self.gconv1_norm)
            self.gpool1 = max_pool(self.grelu1, 3, 2)

        with tf.variable_scope('pconv1'):
            self.pconv1 = conv2d(self.pstate, 7, 1, 32)
            self.pconv1_norm = tf.layers.batch_normalization(self.pconv1, momentum=0.9, training=self.is_training)
            self.prelu1 = tf.nn.selu(self.pconv1_norm)
            self.ppool1 = max_pool(self.prelu1, 3, 2)

        self.flat = tf.concat([flatten(self.gpool1), flatten(self.ppool1)], axis=1)
        print('flat layer', self.flat.get_shape())

        with tf.variable_scope('fc1'):
            self.fc1 = tf.contrib.slim.fully_connected(self.flat, 300, activation_fn=None)
            self.relu1 = tf.nn.selu(self.fc1)
            self.fc1_out = tf.layers.dropout(self.relu1, rate=0.7, training=self.is_training)

        with tf.variable_scope('fc2'):
            self.fc2 = tf.contrib.slim.fully_connected(self.fc1_out, self.action_size)

        self.q = self.fc2
        self.next_q = tf.placeholder(shape=self.fc2.get_shape(), dtype=tf.float32, name='next_q')
        self.loss_op = tf.reduce_mean(tf.square(self.next_q - self.fc2))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
                self.loss_op, global_step=self.global_step)
        return None

    def memory(self, state, action, reward, next_state, done):
        # done is whether the episode is ended or not
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # explore
            return random.randrange(self.action_size)

        # exploit
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def compute_loss(self, session, current_state):
        next_q = session.run([self.q], feed_dict={
            self.gstate: current_state.goal,
            self.pstate: current_state.curr
        })

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target

            self.model.fit(state, target_f, epoches=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, session, env_creator, n_episodes):
        done = False
        for e in range(n_episodes):
            # get example from environment
            env = env_creator.create()
            state = env.init()

            t = 0
            # each eps have max 500 time steps
            while t < 500:
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)



    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def conv2d(input, kernel_size, stride, num_filter, padding='SAME'):
    stride_shape = [1, stride, stride, 1]
    filter_shape = [kernel_size, kernel_size, input.get_shape()[3], num_filter]

    W = tf.get_variable('w', filter_shape, tf.float32, tf.random_normal_initializer(0.0, 0.02))
    b = tf.get_variable('b', [1, 1, 1, num_filter], initializer=tf.constant_initializer(0.0))

    return tf.nn.conv2d(input, W, stride_shape, padding=padding) + b


def max_pool(input, kernel_size, stride, padding='SAME'):
    ksize = [1, kernel_size, kernel_size, 1]
    strides = [1, stride, stride, 1]

    return tf.nn.max_pool(input, ksize=ksize, strides=strides, padding=padding)


def flatten(tensor):
    return tf.reshape(tensor, [-1] + [np.prod(tensor.get_shape()[1:])])
