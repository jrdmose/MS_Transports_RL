
# DOUBLE DQN
################################

import tools
import os, sys
import random
import numpy as np
import tensorflow as tf
import copy

SAVE_AFTER = 11000 # Save model checkpoint
STORE_LOGS_AFTER = 100 # Store tensorflow logs after STORE_LOGS_AFTER iterations
WARM_UP_NET = 20 # Number of simu steps to warm up the network

class DoubleDQN:
    """The DQN agent. Handles the updating of q-networks, takes action, and gets environment response.

    Attributes
    ----------
    q_network : keras model instance to predict q-values for current state
    target_q_network : keras model instance to predict q-values for state after action
    memory : memory instance - needs to be instantiated first # should this be instantiated here?
    gamma : (int) discount factor for rewards
    target_update_freq : (int) defines after how many steps the q-network should be re-trained
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
                 num_burn_in,
                 batch_size,
                 optimizer,
                 loss_func,
                 max_ep_length,
                 env_name,
                 output_dir,
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
        self.__compile(optimizer, loss_func,opt_metric)
        self.memory = memory
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.num_burn_in = num_burn_in
        self.batch_size = batch_size
        self.trained_episodes = 0
        self.max_ep_len = max_ep_length
        self.output_dir = output_dir
        self.experiment_id = experiment_id
        self.summary_writer=summary_writer
        self.itr = 0

        #self.learning_type = learning_type


    def __compile(self, optimizer, loss_func, opt_metric):
        """Initialisation method, using the keras instance compile method. """

        self.q_network.compile(optimizer, loss_func, opt_metric)
        self.target_q_network.compile(optimizer, loss_func, opt_metric)

    def warm_up_net(self, env, num_it):
        """ Runs the environment for some iterations to fill the network.
        The network is filled with a static policy

        Parameters
        ----------

        num_it =  number of simulation steps to run
        """

        for i in range(num_it):
            action = env.action.select_action("fixed", env=env, v_row_t = 15, h_row_t = 40)
            env.step(action)

    def fill_replay(self, env):
        """Helper method for train. Fills the memory before model training begins
        choosing random actions.

        Parameters
        ----------
        env :  environment instance
        policy : (str) policy to be used to fill memory
        """

        print("Filling experience replay memory...")

        env.start_simulation(self.output_dir)
        self.warm_up_net( env, WARM_UP_NET)

        for i in range(self.num_burn_in):
            action = env.action.select_action('rand')
            state, reward, nextstate, done = env.step(action)
            self.memory.append(state, action, reward, nextstate)

        env.stop_simulation()
        print("...Done")

    def update_network(self):
        """Helper method for train. Computes keras neural network updates using samples from memory.

        Notice that we want to incur in loss in the actions that we have selected.
        Q_target and Q are set equal for not relevant actions so the loss is 0.
        (weights not being updated due to these actions)
        """
        # Sample mini batch
        states_m, actions_m, rewards_m, states_m_p = self.memory.sample(self.batch_size)

        # attach q-values to states
        target_batch = self.q_network.predict(states_m)
        target_q = self.target_q_network.predict(states_m_p)

        # choose action
        selected_actions = np.argmax(target_q, axis=1)

        # update q-values
        for i, action in enumerate(selected_actions):
            target_batch[i, action] =  rewards_m[i] + self.gamma * target_q[i, action]

        # keras method to train on batch that returns loss
        loss = self.q_network.train_on_batch(states_m, target_batch)

        # get weights
        weights = self.q_network.get_weights()

        # Update weights every target_update_freq steps
        if self.itr % self.target_update_freq == 0:
            self.target_q_network.set_weights(weights)

        # Save network every save_after iterations if monitoring allowed
        if self.output_dir and self.itr % SAVE_AFTER == 0:
            self.save()

        return loss

    def train(self, env, num_episodes, policy, **kwargs):
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
            sys.stdout.write("\r"+'Running episode {} / {}'.format(self.trained_episodes+1,
                                                       start_train_ep + num_episodes))

            # Each time an episode is run need to create a new random routing
            tools.generate_routefile()
            env.start_simulation(self.output_dir)
            self.warm_up_net( env, WARM_UP_NET)

            nextstate = env.state.get()
            done = False

            stats = {
                'ep_id' : self.trained_episodes,
                'total_reward': 0,
                'episode_length': 0,
                'max_q_value': 0,
            }

            while not done and stats["episode_length"] < self.max_ep_len:


                q_values = self.q_network.predict(nextstate)
                action = env.action.select_action(policy, q_values = q_values, **kwargs)
                state, reward, nextstate, done = env.step(action)
                self.memory.append(state, action, reward, nextstate)

                # Update network weights and record loss for Tensorboard
                loss = self.update_network()


                if self.output_dir and self.itr % STORE_LOGS_AFTER == 0:
                    # create list of stats for Tensorboard, add scalars
                    training_data = [tf.Summary.Value(tag = 'loss',
                                                      simple_value = loss)]
                                    #                   ,
                                    # tf.Summary.Value(tag = 'Action 1',
                                    #                   simple_value = self.q_network.layers[-1].get_weights()[1][0]),
                                    # tf.Summary.Value(tag = 'Action 2',
                                    #                   simple_value = self.q_network.layers[-1].get_weights()[1][1]),
                                    # tf.Summary.Value(tag = 'Episode Length',
                                    #                   simple_value = stats["episode_length"])]

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

            mean_delay = tools.compute_mean_duration(self.output_dir)

            episode_summary = [tf.Summary.Value(tag = 'reward',
                                              simple_value = stats['total_reward']),
                               tf.Summary.Value(tag = 'Average vehicle delay',
                                              simple_value = mean_delay)]
            self.summary_writer.add_summary(tf.Summary(value = episode_summary), global_step=self.trained_episodes)

            self.trained_episodes += 1


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

        self.warm_up_net( env, WARM_UP_NET)

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

        while not done and it < self.max_ep_len:
            #import pdb; pdb.set_trace()
            transition["q_values"] = self.q_network.predict(transition["next_state"])
            transition["action"] = env.action.select_action(policy, q_values = transition["q_values"], **kwargs)
            transition["state"], transition["reward"], transition["next_state"],done = env.step(transition["action"])
            transition["it"] +=1

            all_trans.append(copy.deepcopy(transition))

        env.stop_simulation()

        mean_duration = tools.compute_mean_duration(self.output_dir)

        return all_trans, mean_duration

    def histo_summary(self, values, bins=1000):
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

    def named_logs(self, q_network, logs):
        """create logs"""

        result = {}
        for l in zip(q_network.metrics_names, logs):
            result[l[0]] = l[1]

        return result
