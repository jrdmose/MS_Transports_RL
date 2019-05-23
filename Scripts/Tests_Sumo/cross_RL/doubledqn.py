
# DOUBLE DQN
################################

import tools
import os, sys
import random
import numpy as np
import tensorflow as tf
import copy
import keras.backend as K

SAVE_AFTER = 11000 # Save model checkpoint
STORE_LOGS_AFTER = 100 # Store tensorflow logs after STORE_LOGS_AFTER iterations


class DoubleDQN:
    """The DQN agent. Handles the updating of q-networks, takes action, and gets environment response.

    Attributes
    ----------
    q_network : keras model instance to predict q-values for current state
    target_q_network : keras model instance to predict q-values for state after action
    memory : memory instance - needs to be instantiated first # should this be instantiated here?
    gamma : (int) discount factor for rewards
    target_update_freq : (int) defines after how many steps the q-network should be re-trained
    train_freq: (int) How often you actually update your Q-Network. Sometimes stability is improved
        if you collect a couple samples for your replay memory, for every Q-network update that you run.
    num_burn_in : (int) defines the size of the replay memory to be filled before, using a specified policy
    batch_size : (int) size of batches to be used to train models
    trained_episodes : (int) episode counter
    max_ep_len : (int) stops simulation after specified number of episodes
    output_dir : (str) directory to write tensorboard log and model checkpoints
    experiment_id : (str) ID of simulation
    summary_writer : tensorboard summary stat writer instance
    itr : (int) counts global training steps in all run episodes

    Methods
    -------
    __compile()
        Initialisation method, using the keras instance compile method.

    fill_replay()
        Helper method for train. Fills the memory before model training begins.

    save()
        Calls keras save method using the keras model instance.

    update_network(env, policy)
        Helper method for train. Computes keras neural network updates using samples from memory.

    train(env, num_episodes, policy, **kwargs)
        Main method for the agent. Trains the keras neural network instances, calls all other helper methods.

    evaluate(env)
        Use trained agent to run a simulation without training.
    """

    def __init__(self,
                 q_network,
                 target_q_network,
                 memory,
                 gamma,
                 target_update_freq,
                 train_freq,
                 num_burn_in,
                 batch_size,
                 optimizer,
                 loss_func,
                 max_ep_length,
                 env_name,
                 output_dir,
                 monitoring,
                 experiment_id,
                 summary_writer,
                 model_checkpoint = True,
                 opt_metric = None # Used to judge the performance of the ANN
                 ):
        """
        # TODO: specify defaults for required arguments

        Parameters
        ----------
        """
        self.q_network = q_network
        self.target_q_network = target_q_network
        self.target_q_network.set_weights(self.q_network.get_weights())
        self.__compile(optimizer, loss_func, opt_metric)
        self.memory = memory
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.num_burn_in = num_burn_in
        self.batch_size = batch_size
        self.trained_episodes = 0
        self.max_ep_len = max_ep_length
        self.output_dir = output_dir
        self.monitoring = monitoring
        self.experiment_id = experiment_id
        self.summary_writer = summary_writer
        self.train_freq = train_freq
        self.itr = 0


    def __compile(self, optimizer, loss_func, opt_metric):
        """Initialisation method, using the keras instance compile method. """

        self.q_network.compile(optimizer, loss_func, opt_metric)
        self.target_q_network.compile(optimizer, loss_func, opt_metric)



    def fill_replay(self, env):
        """Helper method for train. Fills the memory before model training begins
        choosing random actions.

        Parameters
        ----------
        env :  environment instance
        policy : (str) policy to be used to fill memory
        """

        # print("Filling experience replay memory...")

        env.start_simulation(self.output_dir)


        for i in range(self.num_burn_in):
            action = env.action.select_action('rand')
            state, reward, nextstate, done = env.step(action)
            self.memory.append(state, action, reward, nextstate, done)
            # If episode finished, continue with another episode
            if done:
                print("Episode finished during memory replay fill. Starting new episode...")
                env.start_simulation(self.output_dir)

        env.stop_simulation()
        # print("...done filling replay memory")

    def update_network(self):
        """Helper method for train. Computes keras neural network updates using samples from memory.

        Notice that we want to incur in loss in the actions that we have selected.
        Q_target and Q are set equal for not relevant actions so the loss is 0.
        (weights not being updated due to these actions)
        """
        # Sample mini batch
        states_m, actions_m, rewards_m, states_m_p, done_m = self.memory.sample(self.batch_size)

        # randomly swap the target and active networks
        # if np.random.uniform() < 0.5:
        #     import pdb; pdb.set_trace()
        #     temp = self.q_network
        #     self.q_network = self.target_q_network
        #     self.target_q_network = temp

        #import pdb; pdb.set_trace()
        # attach q-values to states
        target_batch = self.q_network.predict(states_m)
        next_state_q = self.target_q_network.predict(states_m_p)

        # Predict q values for updated network on s' and choose action according to online network (q_network)
        next_q_online_network = self.q_network.predict(states_m_p)
        selected_actions = np.argmax(next_q_online_network, axis=1)

        # update q-values
        for i, action in enumerate(selected_actions):
            if done_m[i]:
                target_batch[i,action] = rewards_m[i]
            else:
                target_batch[i, action] =  rewards_m[i] + self.gamma * next_state_q[i, action]

        # keras method to train on batch that returns loss
        fit = self.q_network.fit(states_m, target_batch, batch_size = self.batch_size, verbose = 0)

        # Update weights every target_update_freq steps
        if self.itr % self.target_update_freq == 0:
            # get weights
            weights = self.q_network.get_weights()
            self.target_q_network.set_weights(weights)

        # Save network every save_after iterations if monitoring allowed
        if self.monitoring and self.itr % SAVE_AFTER == 0:
            self.save()

        return fit.history["loss"][0]

    def train(self, env, num_episodes, policy, connection_label,**kwargs):
        """Main method for the agent. Trains the keras neural network instances, calls all other helper methods.

        Parameters
        ----------
        env: (str) name of environment instance
        num_episodes: (int) number of training episodes
        policy: (str) name of policy to use to fill memory initially
        """

        all_stats = []
        all_rewards = []
        start_train_ep = self.trained_episodes

        for i in range(num_episodes):
            # print progress of training
            if self.trained_episodes % 1 == 0:
                print('Run {} -- running episode {} / {}'.format(connection_label,
                                                            self.trained_episodes+1,
                                                            start_train_ep + num_episodes))

            # Each time an episode is run need to create a new random routing
            env.start_simulation(self.output_dir)

            nextstate = env.state.get()
            done = False

            stats = {
                'ep_id' : self.trained_episodes,
                'total_reward': 0,
                'episode_length': 0,
                'max_q_value': 0,
                'mean_delay': 0
            }

            while not done and stats["episode_length"] < self.max_ep_len:

                if policy == "linDecEpsGreedy" or policy == "epsgreedy_decay":
                    kwargs["itr"] = self.itr

                q_values = self.q_network.predict(nextstate)
                action = env.action.select_action(policy, q_values = q_values, **kwargs)
                state, reward, nextstate, done = env.step(action)
                self.memory.append(state, action, reward, nextstate, done)

                # Update network weights and record loss for Tensorboard

                if self.itr % self.train_freq == 0:
                    loss = self.update_network()


                if self.monitoring and self.itr % STORE_LOGS_AFTER == 0:
                    # create list of stats for Tensorboard, add scalars

                    training_data = [tf.Summary.Value(tag = 'TD - loss',
                                                      simple_value = loss)]
                                    #  tf.Summary.Value(tag = 'learning rate',
                                    #                  simple_value = K.eval(self.q_network.optimizer.lr))]

                    # add histogram of weights to list of stats for Tensorboard
                    for index, layer in enumerate(self.q_network.layers):

                        if index != len(self.q_network.layers) - 1:
                            training_data.append(tf.Summary.Value(tag = str(layer.name) + " weights" ,
                                                            histo = self.histo_summary(layer.get_weights()[0])))
                            if len(layer.get_weights()) > 1:
                                training_data.append(tf.Summary.Value(tag = str(layer.name) + " relu" ,
                                                            histo = self.histo_summary(layer.get_weights()[1])))

                        else:
                            training_data.append(tf.Summary.Value(tag = str(layer.name) + " output weights" ,
                                                            histo = self.histo_summary(layer.get_weights()[0])))
                            training_data.append(tf.Summary.Value(tag = "output values",
                                                            histo = self.histo_summary(layer.get_weights()[1])))

                    # write the list of stats to the logdd
                    self.summary_writer.add_summary(tf.Summary(value = training_data), global_step=self.itr)

                self.itr += 1

                stats["ep_id"] = self.trained_episodes
                stats["episode_length"] += 1
                stats['total_reward'] += reward
                stats['max_q_value'] += max(q_values)

            env.stop_simulation()

            # Static policy evaluation for comparison during training
            #_,static_dur = self.evaluate(env,"fixed", v_row_t = 40, h_row_t = 40)

            if self.monitoring:
                mean_delay = tools.compute_mean_duration(self.output_dir)

                episode_summary = [tf.Summary.Value(tag = 'Reward',
                                                  simple_value = stats['total_reward']),
                                   tf.Summary.Value(tag = 'Average vehicle delay',
                                                  simple_value = mean_delay)]

                               #tf.Summary.Value(tag = 'Average vehicle delay static',
                               #                  simple_value = static_dur)]
                stats["mean_delay"] = mean_delay
                self.summary_writer.add_summary(tf.Summary(value = episode_summary), global_step=self.trained_episodes)
            all_stats.append(stats)
            self.trained_episodes += 1
        return all_stats


    def evaluate(self, env, policy, **kwargs):
        """Use trained agent to run a simulation.

        Parameters
        ----------
        env : environment instance
        """

        env.start_simulation(self.output_dir)
        nextstate = env.state.get()
        done = False
        it = 0

        transition = {
            "it" : it,
            "state" : nextstate,
            "q_values" : np.zeros(2),
            "action" : 0,
            "reward" : 0,
            "next_state" : nextstate,
        }

        all_trans = []

        if policy == 'fixed':
            kwargs["env"] = env

        if policy == "linDecEpsGreedy":
            kwargs["itr"] = self.itr


        while not done and it < self.max_ep_len:
            #import pdb; pdb.set_trace()
            transition["q_values"] = self.q_network.predict(transition["next_state"])
            transition["action"] = env.action.select_action(policy, q_values = transition["q_values"], **kwargs)
            transition["state"], transition["reward"], transition["next_state"],done = env.step(transition["action"])
            transition["it"] += 1

            all_trans.append(copy.deepcopy(transition))

        env.stop_simulation()
        mean_delay = tools.compute_mean_duration(self.output_dir)

        return all_trans, mean_delay

    def histo_summary(self, values, bins=100):
        """Helper function in train method. Log a histogram of the tensor of values for tensorboard.

        Creates a HistogramProto instance that can be fed into Tensorboard.

        Parameters
        ---------
        values :  (np.array) histogram values
        bins : (int) how coarse the histogram is supposed to be
        """

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins = bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        return hist

    def save(self):
        """Calls keras save function using the keras model instance"""

        filename =  "{}/model_checkpoints/run{}_iter{}.h5" .format(self.output_dir,
                                               self.experiment_id,
                                               self.itr)
        self.q_network.save(filename)

    def load(self, filename):
        self.q_network.load_weights(filename)

    def named_logs(self, q_network, logs):
        """create logs"""

        result = {}
        for l in zip(q_network.metrics_names, logs):
            result[l[0]] = l[1]

        return result
