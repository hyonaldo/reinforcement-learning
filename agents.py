# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import os
import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import datetime
from dateutil.tz import tzlocal


class BaseModel(object):
    """Inherit from this class when implementing new models."""

    def __init__(self, learning_rate, discount_factor,
                 layers,
                 reward_standardize=False
                 ):
        self.scope = self.__class__.__name__
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.reward_standardize = reward_standardize
        
        print("scope:",self.scope, "learning_rate:",self.learning_rate,
              "discount_factor:",self.discount_factor, "reward_standardize:",self.reward_standardize)

    ### util functions ###
    def get_integer_time(self):
        # Get the current date/time with the timezone.
        now = datetime.datetime.now(tzlocal())
        str_min = now.strftime('%Y%m%d%H%M')
        int_min = int(str_min)
        return int_min

    # discount factor for reward
    def gamma_discount(self, r, gamma=0.99, standardize=False):
        discounted_r = np.zeros_like(r, dtype=np.float32)
        running_add = 0
        for t in reversed(xrange(0, len(r))):
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        if standardize == True:
            # size the rewards to be unit normal (helps control the gradient estimator variance)
            try:
                if np.std(discounted_r) > 0:
                    discounted_r -= np.mean(discounted_r)
                    discounted_r /= np.std(discounted_r)
            except:
                pass

        return discounted_r


class MultilayerPolicyGradient(BaseModel):
    def __init__(self, learning_rate, discount_factor,
                 layers,
                 reward_standardize=False
                 ):
        BaseModel.__init__(self,
                           learning_rate=learning_rate,
                           discount_factor=discount_factor,
                           layers=layers,
                           reward_standardize=reward_standardize)

        self.build_network(layers)
        self.build_objectives(layers[-1])

        target_variables = [variable for variable in tf.global_variables() if variable.name.startswith(self.scope)]
        self.saver = tf.train.Saver(target_variables)

    def weight_variable(self, shape, name):
        #         initial = tf.truncated_normal(shape, stddev=0.1)
        #         return tf.Variable(initial)
        return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

    def bias_variable(self, shape, name):
        return tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(0.1))

    #         return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

    def build_network(self, layers):
        state_space_size = layers[0]
        with tf.variable_scope(self.scope):
            self.is_training = tf.placeholder(dtype=tf.bool)
            self.state_in = tf.placeholder(shape=[None, state_space_size], dtype=tf.float32)

            # build shallow graph such as layers parameter
            self.weights = []
            self.biases = []
            self.nodes = []
            ### Input & Hidden Layer ###
            for i in range(len(layers) - 2):
                layer_num = i + 1
                layer_in = layers[i]
                layer_out = layers[i + 1]
                if (layer_num == 1):
                    layer_str = "input_layer"

                    self.weights.append(self.weight_variable([layer_in, layer_out], name="W" + str(i)))
                    self.biases.append(self.bias_variable([layer_out], name="B" + str(i)))
                    self.nodes.append(tf.nn.relu(tf.matmul(self.state_in, self.weights[i]) + self.biases[i]))
                    print("1 relu layer is added")

                else:
                    layer_str = "layer" + str(layer_num)

                    self.weights.append(self.weight_variable([layer_in, layer_out], name="W" + str(i)))
                    self.biases.append(self.bias_variable([layer_out], name="B" + str(i)))
                    self.nodes.append(tf.nn.relu(tf.matmul(self.nodes[i - 1], self.weights[i]) + self.biases[i]))
                    print("1 relu layer is added")

            ### Output Layer ###
            i = i + 1
            layer_in = layers[i]
            layer_out = layers[i + 1]

            self.weights.append(self.weight_variable([layer_in, layer_out], name="W" + str(i)))
            self.biases.append(self.bias_variable([layer_out], name="B" + str(i)))

            self.tvars = tf.trainable_variables()
            self.gradient_holders = []
            for idx, var in enumerate(self.tvars):
                placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
                self.gradient_holders.append(placeholder)

            ####################################
            self.z = tf.matmul(self.nodes[i - 1], self.weights[i]) + self.biases[i]
            #             self.output = tf.nn.softmax( self.z )
            self.output = tf.clip_by_value(tf.nn.softmax(self.z), 1e-19, 1.0)
            print("1 softmax layer is added")
            ####################################

    def build_objectives(self, action_space_size):
        with tf.variable_scope(self.scope):
            # action들과 그들 각각에 대한 reward를 받아서 loss를 구하고 network을 update함
            self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
            self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

            # action값에 해당하는 output노드만 추려낸다. (masking) 해당 노드 확률값에 대해서만 gradient ascending 하기 위해서다.
            self.masking = tf.one_hot(self.action_holder, action_space_size)
            self.selected_outputs = tf.reduce_sum(self.output * self.masking, axis=1)

            self.loss = -tf.reduce_mean(tf.log(self.selected_outputs) * self.reward_holder)

            self.gradients = tf.gradients(self.loss, self.tvars)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            self.update_batch = self.optimizer.apply_gradients(zip(self.gradient_holders, self.tvars))

    # self.no_batch_just_train = self.optimizer.minimize(self.loss)

    # Before training, any initialization code
    def before(self, sess):
        # store observations, actions and rewards in an episode
        self.episode_observations, self.episode_actions, self.episode_rewards = [], [], []

        # Reset the gradient placeholder.
        gradBuffer = sess.run(tf.trainable_variables())
        for ix, grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0
        self.gradBuffer = gradBuffer

    # Return an action
    def get_action(self, sess, observation):
        # Probabilistically pick an action given our network outputs.
        a_dist = sess.run(self.output, feed_dict={self.state_in: [observation], self.is_training: True})
        action = np.random.choice(a_dist[0], p=a_dist[0])
        action = np.argmax(a_dist == action) 

        self.episode_observations.append(observation)
        self.episode_actions.append(action)

        return action

    # Return an action by [just exploit!]
    def exploit(self, sess, observation):
        # Probabilistically pick an action given our network outputs.
        action_pd = sess.run(self.output, feed_dict={self.state_in: [observation],
                                                     self.is_training: False})
        dist = action_pd[0]
        action = np.argmax(dist)

        return action

    # After action has been processed by env, what to do with reward
    def after_action(self, sess, reward):
        # just store the reward for later
        self.episode_rewards.append(reward)

    # After each episode
    def after_episode(self, sess):
        discounted_rewards = self.gamma_discount(r=self.episode_rewards,
                                                 gamma=self.discount_factor,
                                                 standardize=self.reward_standardize)

        feed_dict = {self.reward_holder: discounted_rewards,
                     self.action_holder: self.episode_actions,
                     self.state_in: np.vstack(self.episode_observations),
                     self.is_training: True
                     }

        #         sess.run(myAgent.no_batch_just_train, feed_dict=feed_dict)

        # store gradients in grad buffer
        loss, grads = sess.run([self.loss, self.gradients], feed_dict=feed_dict)
        if np.isnan(loss):
            print("=======================")
            print("loss is nan!!!!! skip apply grads...")
            print("=======================")
        else:
            for idx, grad in enumerate(grads):
                if np.isnan(grad).any():
                    print("=======================")
                    print("nan coming in grads!!!!! skip apply grads...")
                    print("=======================")
                else:
                    self.gradBuffer[idx] += grad
                    #             self.gradBuffer[idx] += grad

        # clear episode variables
        self.episode_observations, self.episode_actions, self.episode_rewards = [], [], []

        # return to debug or tensorboard ... etc
        return loss, grads, discounted_rewards

    # After each (bach size) episodes
    def after_batch(self, sess):
        # Update the network.
        feed_dict = dictionary = dict(zip(self.gradient_holders, self.gradBuffer))
        _ = sess.run(self.update_batch, feed_dict=feed_dict)

        # clear the gradient buffer
        for ix, grad in enumerate(self.gradBuffer):
            self.gradBuffer[ix] = grad * 0

    def read_ckpt(self, dir_path):
        ckpt_path = tf.train.latest_checkpoint(dir_path)
        reader = tf.train.NewCheckpointReader(ckpt_path)
        saved_variables = [var_name for var_name in reader.get_variable_to_shape_map() if
                           var_name.startswith(self.scope)]
        return ckpt_path, saved_variables

    def save_model(self, sess, dir_path):
        ckpt_path = dir_path + "/model.ckpt"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        now = self.get_integer_time()
        self.saver.save(sess, ckpt_path, global_step=now)

        print("Model saved at", dir_path)

    def load_model(self, sess, dir_path):
        #         ckpt_path = dir_path+"/model.ckpt"
        ckpt_path = tf.train.latest_checkpoint(dir_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        try:
            self.saver.restore(sess, ckpt_path)
            print("Model restored successfully from", dir_path)
        except:
            print("Loading model failed from", dir_path)
            raise

    def load_specific_model(self, sess, ckpt_path):
        #         ckpt_path = dir_path+"/model.ckpt"
        ckpt_path = tf.train.latest_checkpoint(dir_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        try:
            self.saver.restore(sess, ckpt_path)
            print("Model restored successfully from", dir_path)
        except:
            print("Loading model failed from", dir_path)
            raise



class VanillaPolicyGradient(MultilayerPolicyGradient):
    def __init__(self, learning_rate, discount_factor,
                 layers,
                 reward_standardize=False
                 ):

        MultilayerPolicyGradient.__init__(self,
                                          learning_rate=learning_rate,
                                          discount_factor=discount_factor,
                                          layers=layers,
                                          reward_standardize=reward_standardize)

    def build_network(self, layers):
        valid_layers = 3
        if len(layers) != valid_layers:
            err = self.scope + "must have [input,hidden,output] layers. i.e ("
            err = err + str(valid_layers) + ",)!\n"
            err = err + "But your input shape was" + str(np.array(layers).shape)
            raise TypeError(err)

        state_space_size = layers[0]
        hidden_size = layers[1]
        action_space_size = layers[2]
        with tf.variable_scope(self.scope):
            self.is_training = tf.placeholder(dtype=tf.bool)
            self.state_in = tf.placeholder(shape=[None, state_space_size], dtype=tf.float32)

            # build shallow graph using tensorflow.contrib.slim
            hidden = slim.fully_connected(self.state_in, hidden_size,
                                          biases_initializer=None,
                                          activation_fn=tf.nn.relu)
            ####################################
            self.output = tf.clip_by_value(
                slim.fully_connected(hidden,
                                     action_space_size,
                                     activation_fn=tf.nn.softmax,
                                     biases_initializer=None)
                , 1e-19, 1.0
            )
            ####################################
            
            self.tvars = tf.trainable_variables()
            self.gradient_holders = []
            for idx, var in enumerate(self.tvars):
                placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
                self.gradient_holders.append(placeholder)


class MultilayerPolicyGradientCNN(MultilayerPolicyGradient):
    def __init__(self, learning_rate, discount_factor,
                 layers,
                 dropout_rate,
                 reward_standardize=False
                 ):

        self.dropout_rate = dropout_rate
        print("dropout_rate:",self.dropout_rate)
        
        MultilayerPolicyGradient.__init__(self,
                                          learning_rate=learning_rate,
                                          discount_factor=discount_factor,
                                          layers=layers,
                                          reward_standardize=reward_standardize)

    def weight_variable(self, shape, name):
        return tf.get_variable(name=name, shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.1))

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def decide_virtual_image_size(self, input_size):
        width_height = 2  # only take even
        max_wh = 1000

        while width_height < max_wh:
            width_height = width_height + 2  # only take even
            expand_size = pow(width_height, 2)
            diff = abs(input_size - expand_size)
            next_diff = abs(input_size - pow(width_height + 1, 2))
            if expand_size >= input_size and diff < next_diff:
                break
                
        return width_height, expand_size

    # return 2d image array to plt.imshow() or plt.imsave()
    def get_real_image(self, sess, observation):

        # Probabilistically pick an action given our network outputs.
        expanded = sess.run(self.x, feed_dict={self.state_in: [observation],
                                                     self.is_training: False})[0]
        width_height = int(np.sqrt( len(expanded) ) )
        return np.reshape(expanded, [width_height, width_height])
                       
    def build_network(self, layers):
        valid_layers = 2
        if len(layers) != valid_layers:
            err = self.scope + "must have [input,output] layers. i.e (" + str(valid_layers) + ",)!\n"
            err = err + "But your input shape was" + str(np.array(layers).shape)
            raise TypeError(err)

        state_space_size = layers[0]
        action_space_size = layers[1]
        width_height, expand_size = self.decide_virtual_image_size(state_space_size)
        print("original input size:", state_space_size)
        print("virtual_image_size:", width_height, "X", width_height, "=", expand_size)
        
#         layers = [state_space_size, expand_size, action_space_size]
#         MultilayerPolicyGradient.build_network(self, layers)
        
        with tf.variable_scope(self.scope):
            self.is_training = tf.placeholder(dtype=tf.bool)
            self.state_in = tf.placeholder(shape=[None, state_space_size], dtype=tf.float32)

            # expand original input dim to expand_size dim
            self.w_in = self.weight_variable([state_space_size, expand_size], name="w_in")
            self.b_in = self.bias_variable([expand_size], name="b_in")
            self.x = tf.matmul(self.state_in, self.w_in) + self.b_in

            ### First Convolutional Layer ###
            self.W_conv1 = self.weight_variable([5, 5, 1, 32], name="W_conv1")  # compute 32 features for each 5x5 patch
            self.b_conv1 = self.bias_variable([32], name="b_conv1")

            # reshape x to a 4d tensor, with the second and third dimensions corresponding to image
            # width and height, and the final dimension corresponding to the number of color channels
            self.x_image = tf.reshape(self.x, [-1, width_height, width_height, 1])

            self.h_conv1 = tf.nn.relu(self.conv2d(self.x_image, self.W_conv1) + self.b_conv1)
            self.h_pool1 = self.max_pool_2x2(self.h_conv1)  # max_pool_2x2 method will reduce the image size to (width_height* 1/2)

            ### Second Convolutional Layer ###
            self.W_conv2 = self.weight_variable([5, 5, 32, 64], name="W_conv2")
            self.b_conv2 = self.bias_variable([64], name="b_conv2")

            self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
            self.h_pool2 = self.max_pool_2x2(self.h_conv2)  # max_pool_2x2 method will reduce the image size to (width_height* 1/2 *1/2)

            ### Densely Connected Layer ###
            reduced_size = int(width_height * 0.5 * 0.5)  # the image size reduced by max_pool_2x2
            self.W_fc1 = self.weight_variable([reduced_size * reduced_size * 64, 1024], name="W_fc1")
            self.b_fc1 = self.bias_variable([1024], name="b_fc1")

            self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, reduced_size * reduced_size * 64])
            self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)

            ### Dropout & Readout Layer ###
            self.h_fc1_drop = tf.layers.dropout(self.h_fc1, rate=self.dropout_rate, training=self.is_training)

            self.W_fc2 = self.weight_variable([1024, action_space_size], name="W_fc2")
            self.b_fc2 = self.bias_variable([action_space_size], name="b_fc2")

            self.z = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2

            ####################################
            self.output = tf.clip_by_value(tf.nn.softmax(self.z), 1e-19, 1.0)
            print("simple CNN is constructed")
            ####################################

            self.tvars = tf.trainable_variables()
            self.gradient_holders = []
            for idx, var in enumerate(self.tvars):
                placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
                self.gradient_holders.append(placeholder)


class MultiActionsPolicyGradient(MultilayerPolicyGradient):
    def __init__(self, learning_rate, discount_factor,
                 layers,
                 action_top_n,
                 reverse=False
                 ):

        self.action_top_n = action_top_n
        self.reverse = reverse
        MultilayerPolicyGradient.__init__(self,
                                          learning_rate=learning_rate,
                                          discount_factor=discount_factor,
                                          layers=layers,
                                          reward_standardize=False)

    def build_objectives(self, action_space_size):
        with tf.variable_scope(self.scope):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            ### for pre-training ###
            self.past_click = tf.placeholder(shape=[None, action_space_size], dtype=tf.float32)
#             self.pre_loss = tf.losses.mean_squared_error(predictions=self.z, labels=self.past_click)
            self.pre_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.z, labels=self.past_click)
            )
            self.pre_train = self.optimizer.minimize(self.pre_loss)
            ####################################

            # action_holder 가 필요 없는 이유는, reward_holder가 이미 action한 element에 대해서만 값을 가지고 있다고 가정하기 때문.
            self.reward_holder = tf.placeholder(shape=[None, action_space_size], dtype=tf.float32)

            # 따라서 모든 action element에 대한 self.output을 reward_holder와 곱하면 원하는 trajectory에 대한 계산이 가능.

            #         self.loss = -tf.reduce_mean(tf.clip_by_value(tf.log(self.output), -100, +100) * self.reward_holder)
            self.loss = -tf.reduce_mean(tf.log(self.output) * self.reward_holder)

            self.tvars = tf.trainable_variables()
            self.gradient_holders = []
            for idx, var in enumerate(self.tvars):
                placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
                self.gradient_holders.append(placeholder)

            self.gradients = tf.gradients(self.loss, self.tvars)

            self.update_batch = self.optimizer.apply_gradients(zip(self.gradient_holders, self.tvars))

    # size the rewards to be unit normal (helps control the gradient estimator variance)
    def standardize_for_nonzero(self, r):
        rewards_list = np.array(r, dtype=np.float32)
        nonzero_values = []
        for i in range(len(rewards_list)):
            rewards = rewards_list[i]
            for j in range(len(rewards)):
                reward = rewards[j]
                if (reward < 0) or (reward > 0):
                    nonzero_values.append(reward)

        mean = np.mean(nonzero_values)
        std = np.std(nonzero_values)
        if std > 0:
            for i in range(len(rewards_list)):
                rewards = rewards_list[i]
                for j in range(len(rewards)):
                    reward = rewards[j]
                    if (reward < 0) or (reward > 0):
                        rewards_list[i][j] = (rewards_list[i][j] - mean) / std

        return rewards_list

    def get_ranked_list(self, top_indexes, list_size, reverse=False):
        ranked_list = np.zeros(list_size, dtype=np.float32)
        n = len(top_indexes)
        for i in range(n):
            if reverse == False:
                rank = i + 1
            else:
                rank = n - i
            ranked_list[top_indexes[i]] = rank
        return ranked_list

    def get_top_n_actions(self, dist, action_top_n, random=True):
        dist = np.array(dist)
        top_n_actions = []

        if random == False: # when exploit !!
            n_nonzeros = min(len([x for x in dist if x > 0]), action_top_n)
            top_n_actions = sorted(range(len(dist)),
                                   key=lambda i: dist[i],
                                   reverse=True)[:n_nonzeros]
            dist = np.delete(dist, top_n_actions)

        while (len(top_n_actions) < action_top_n):
            sum_probability = sum(dist)
            if(sum_probability > 0):
                dist = dist / sum_probability
                act = np.random.choice(len(dist) , 1, p=dist)[0]
            else:
                act = np.random.choice(len(dist) , 1)[0]

            if act not in top_n_actions:
                top_n_actions.append(act)
                dist[act] = .0

        return top_n_actions
    
#     def get_top_n_actions(self, dist, action_top_n):

#         top_n_actions = []
#         try:
#             top_n_actions = np.random.choice(len(dist), action_top_n, p=dist, replace=False)

#         except ValueError:
#             # case "Fewer non-zero entries in p than size"
#             n_nonzeros = len([x for x in dist if x > 0])
#             top_n_actions = np.random.choice(len(dist), n_nonzeros, p=dist, replace=False)

#             n_shortage = action_top_n - n_nonzeros
#             while n_shortage > 0:
#                 candidate = np.random.choice(len(dist), 1)
#                 if candidate not in top_n_actions:
#                     top_n_actions = np.append(top_n_actions, candidate)
#                     n_shortage = n_shortage - 1

#         return top_n_actions

    # Return an action by [explore and exploit]
    def get_action(self, sess, observation):
        # Probabilistically pick an action given our network outputs.
        action_pd = sess.run(self.output, feed_dict={self.state_in: [observation],
                                                     self.is_training: True})
        dist = action_pd[0]

        top_n_actions = []
        top_n_actions = self.get_top_n_actions(dist, self.action_top_n)

        action = self.get_ranked_list(top_n_actions, len(dist), reverse=self.reverse)

        self.episode_observations.append(observation)
        self.episode_actions.append(action)

        return action

    # Return an action by [just exploit!]
    def exploit(self, sess, observation):
        # Probabilistically pick an action given our network outputs.
        action_pd = sess.run(self.output, feed_dict={self.state_in: [observation],
                                                     self.is_training: False})
        dist = action_pd[0]
        
        top_n_actions = []
        top_n_actions = self.get_top_n_actions(dist, self.action_top_n, random=False)
        
        action = self.get_ranked_list(top_n_actions, len(dist), reverse=self.reverse)

        self.episode_observations.append(observation)
        self.episode_actions.append(action)

        return action

    # After each episode
    def after_episode(self, sess):
        self.episode_observations = np.vstack(self.episode_observations)
        self.episode_rewards = np.vstack(self.episode_rewards)

        discounted_rewards = self.gamma_discount(r=self.episode_rewards,
                                                 gamma=self.discount_factor)
        # discounted_rewards = self.standardize_for_nonzero(self.episode_rewards)

        feed_dict = {self.reward_holder: discounted_rewards,
                     self.state_in: np.vstack(self.episode_observations),
                     self.is_training: True
                     }

        # store gradients in grad buffer
        tvars, output, log_output, reward_holder, loss, grads = sess.run([self.tvars,
                                                                          self.output,
                                                                          tf.log(self.output),
                                                                          self.reward_holder,
                                                                          self.loss,
                                                                          self.gradients],
                                                                         feed_dict=feed_dict)
        # grads = sess.run(self.gradients,feed_dict=feed_dict)
        has_grad_nan = False
        for idx, grad in enumerate(grads):
            if np.isnan(grad).any():
                has_grad_nan = True
                print("=======================")
                print("tvar idx", idx, "nan coming!!!!!")
                print("loss ===========")
                print(loss)
                print("idx, tvars min, max, avg ===========")
                print(idx, np.min(tvars[idx]), np.max(tvars[idx]), np.mean(tvars[idx]))

            else:
                self.gradBuffer[idx] += grad

        # clear episode variables
        self.episode_observations, self.episode_actions, self.episode_rewards = [], [], []

        return loss, tvars, has_grad_nan


class MultiActionsPolicyGradientCNN(MultiActionsPolicyGradient, MultilayerPolicyGradientCNN):
    def __init__(self, learning_rate, discount_factor,
                 layers,
                 action_top_n,
                 dropout_rate,
                 reverse=False,
                 reward_standardize=False
                 ):
        self.dropout_rate = dropout_rate

        MultiActionsPolicyGradient.__init__(self,
                                            learning_rate=learning_rate,
                                            discount_factor=discount_factor,
                                            layers=layers,
                                            action_top_n=action_top_n,
                                            reverse=reverse
                                            )

    def build_network(self, layers):
        MultilayerPolicyGradientCNN.build_network(self, layers)

class MultiActionsPolicyGradientVanilla(MultiActionsPolicyGradient, VanillaPolicyGradient):
    def __init__(self, learning_rate, discount_factor,
                 layers,
                 action_top_n,
                 reverse=False,
                 reward_standardize=False
                 ):

        MultiActionsPolicyGradient.__init__(self,
                                            learning_rate=learning_rate,
                                            discount_factor=discount_factor,
                                            layers=layers,
                                            action_top_n=action_top_n,
                                            reverse=reverse
                                            )

    def build_network(self, layers):
        VanillaPolicyGradient.build_network(self, layers)

    def build_objectives(self, action_space_size):
        with tf.variable_scope(self.scope):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            # action_holder 가 필요 없는 이유는, reward_holder가 이미 action한 element에 대해서만 값을 가지고 있다고 가정하기 때문.
            self.reward_holder = tf.placeholder(shape=[None, action_space_size], dtype=tf.float32)

            # 따라서 모든 action element에 대한 self.output을 reward_holder와 곱하면 원하는 trajectory에 대한 계산이 가능.

            #         self.loss = -tf.reduce_mean(tf.clip_by_value(tf.log(self.output), -100, +100) * self.reward_holder)
            self.loss = -tf.reduce_mean(tf.log(self.output) * self.reward_holder)

            self.tvars = tf.trainable_variables()
            self.gradient_holders = []
            for idx, var in enumerate(self.tvars):
                placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
                self.gradient_holders.append(placeholder)

            self.gradients = tf.gradients(self.loss, self.tvars)

            self.update_batch = self.optimizer.apply_gradients(zip(self.gradient_holders, self.tvars))
