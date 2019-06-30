
import agent
import environment
import doubledqn
import tools
import memory
import os
import inspect
import json

import numpy as np

class simulator:
    """Wrapper that handles all objects needed for a simulation:
    The neural network (agent.py), the decision function (doubledqn.py),
    the environment (environment.py), and the memory (memory.py).

    Algorithm:
    Double DQN algorithm with experience replay and frozen targets.
    Environment:
    SUMO junction

    ** more information in the report folder

    Parameters
    ----------

    *DDQN parameters*
    q_network : (str) keras model instance to predict q-values for current state ('simple' or 'linear')
    target_q_network :  (str) keras model instance to predict q-values for state after action ('simple' or 'linear')
    gamma : (int) discount factor for rewards
    target_update_freq : (int) defines after how many steps the q-network should be re-trained
    train_freq: (int) How often you actually update your Q-Network. Sometimes stability is improved
        if you collect a couple samples for your replay memory, for every Q-network update that you run.
    num_burn_in : (int) defines the size of the replay memory to be filled before, using a specified policy
    batch_size : (int) size of batches to be used to train models
    optimizer : (str) keras optimizer identifier ('adam')
    loss_func : (str) keras loss func identifier ('mse')
    max_ep_len : (int) stops simulation after specified number of episodes
    experiment_id : (str) ID of simulation
    model_checkpoint : (bool) store keras model checkpoints during training
     policy : (str) policy to choose actions ('epsGredy', 'linDecEpsGreedy', 'greedy' 'randUni')
     eps : (float) exploration factor
        if policy = 'linDecEpsGreedy' -> The epsilon will decay from 1 to eps
        if policy = 'epsGredy' -> eps to evaluate eps policy

    *Environment parameters*
     network : (str) network complexity ('simple' or 'complex')
     net_file : (str) SUMO .net.xml file
     route_file : (str) SUMO .rou.xml file
     network_dir : (str) path to network folder
     demand : (str) demand scenario ('rush' or 'nominal')
     state_shape, : (tup) state shape
     num_actions : (int) number of actions (traffic signal phases)
     use_gui : (bool) wether to use user interface
     delta_time : (int) simulation time between actions
     reward : type of reward. ('balanced' or 'negative')

     *Memory buffer parameters*
     max_size : (int) memory capacity required

     *Additional parameters*
      num_episodes : (int) number of episodes to train the algorithm. THis can also be changed in train method.
      eval_fixed = (bool) Evaluate fixed policy during training. Used for plotting
      monitoring : (bool) store episode logs in tensorboard
      episode_recording : (bool) store intra episode logs in tensorboard
      seed = (int)

    Methods
    -------
    __compile()
        Initialises all objects

    train(env, num_episodes)
        Main method for the agent. Trains the keras neural network instances,
        calls all other helper methods.

    evaluate(env)
        Use trained agent to run a simulation without training.
    """

    def __init__(self,
                # ddqn parameters
                 connection_label = "lonely_worker",
                 q_network_type = 'simple',
                 target_q_network_type = 'simple',
                 gamma = 0.99,
                 target_update_freq = 10000,
                 train_freq = 3,
                 num_burn_in = 300,
                 batch_size = 32,
                 optimizer = 'adam',
                 loss_func = "mse",
                 max_ep_length = 1000,
                 experiment_id = "Exp_1",
                 model_checkpoint = True,
                 # environment parameters
                 network = "simple", #"complex"
                 route_file = "cross.rou.xml",
                 network_dir = "./network",
                 demand = "nominal",
                 use_gui = False,
                 delta_time = 10,
                 reward = "balanced",
                 # memory parameters
                 max_size = 100000,
                 # additional parameters
                 policy = "linDecEpsGreedy",
                 eps = 0.1,
                 num_episodes = 2,
                 eval_fixed = False,
                 monitoring = True,
                 episode_recording = False,
                 seed = 1,
                 hparams = None):

        np.random.seed(seed)
        import tensorflow as tf
        from keras import optimizers

        if hparams:
            args_description = locals()
            args_description = str({ key : args_description[key] for key in set(("eps","policy","target_update_freq","reward"))})
        else:
            args_description = "single_worker"

        self.connection_label = connection_label
        self.q_network_type = q_network_type
        self.target_q_network_type = q_network_type
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.train_freq = train_freq
        self.num_burn_in = num_burn_in
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.max_ep_length = max_ep_length
        self.experiment_id = experiment_id
        self.model_checkpoint = model_checkpoint

        # additional parameters
        self.policy = policy
        self.eps = eps
        self.num_episodes = num_episodes
        self.monitoring = monitoring
        self.episode_recording = episode_recording
        self.eval_fixed = eval_fixed
        self.output_dir, self.summary_writer_folder = tools.get_output_folder("./logs", self.experiment_id, args_description)
        self.summary_writer = tf.summary.FileWriter(logdir = self.summary_writer_folder)

        # environment parameters

        if network == "simple":
            self.network = network
            self.net_file = os.path.join(network_dir, "simple_cross.net.xml")
            self.state_shape = (1,15)
            self.num_actions = 2

        elif network == "complex":
            self.network = network
            self.net_file = os.path.join(network_dir, "complex_cross.net.xml")
            self.state_shape = (1,41)
            self.num_actions = 4

        else:
            print("Network doesn't exist")

        self.route_file = os.path.join(self.output_dir, route_file)
        self.demand = demand
        self.use_gui = use_gui
        self.delta_time = delta_time
        self.reward = reward

        # memory parameters
        self.max_size = max_size


        # Initialize Q-networks (value and target)
        self.q_network = agent.get_model(model_name = self.q_network_type,
                                input_shape = (self.state_shape[1],),
                                num_actions = self.num_actions)

        self.target_q_network = agent.get_model(model_name = self.target_q_network_type,
                                input_shape = (self.state_shape[1],),
                                num_actions = self.num_actions)

        # Initialize environment
        self.env =  environment.Env(connection_label = self.connection_label,
                                network = self.network,
                                net_file = self.net_file,
                                route_file = self.route_file,
                                demand =  self.demand,
                                state_shape = self.state_shape,
                                num_actions = self.num_actions,
                                max_ep_len = self.max_ep_length,
                                policy = self.policy,
                                use_gui = self.use_gui,
                                eps = self.eps,
                                reward = self.reward)

        # Initialize replay memory
        self.memory = memory.ReplayMemory(max_size = self.max_size,
                                state_shape = self.state_shape,
                                num_actions = self.num_actions)

        # Initialize Double DQN algorithm
        self.ddqn = doubledqn.DoubleDQN(q_network = self.q_network,
                                target_q_network = self.target_q_network,
                                memory = self.memory,
                                gamma = self.gamma,
                                target_update_freq = self.target_update_freq,
                                train_freq = self.train_freq,
                                num_burn_in = self.num_burn_in,
                                batch_size = self.batch_size,
                                optimizer = self.optimizer,
                                loss_func = self.loss_func,
                                max_ep_length = self.max_ep_length,
                                output_dir = self.output_dir,
                                monitoring = self.monitoring,
                                episode_recording = self.episode_recording,
                                experiment_id = self.experiment_id,
                                summary_writer = self.summary_writer)

        # Store initialization prameters
        self.store_init(locals())


    def train(self, num_episodes = None):
        """
        Train the DDQN algorithm

        returns training data logs.
        """
        if self.memory.cur_size < self.num_burn_in:
            self.ddqn.fill_replay(self.env)

        if num_episodes: self.num_episodes = num_episodes

        train_data = self.ddqn.train(env = self.env,
                        num_episodes = self.num_episodes,
                        policy = self.policy,
                        eval_fixed = self.eval_fixed,
                        connection_label = self.connection_label)

        return train_data
        #print(self.ddqn.q_network.get_weights())

    def load(self,checkpoint_dir):
        with open(os.path.join(os.path.dirname(os.path.dirname(checkpoint_dir)), 'parameters.json'), 'r') as fp:
            arguments = json.load(fp)
        self.__init__(**arguments)
        self.ddqn.load(checkpoint_dir)

    def store_init(self,init_arguments):
        """
        Store arguments of the function
        """
        arguments = {arg:init_arguments[arg] for arg in inspect.getfullargspec(self.__init__).args[1:]}
        arguments.pop('hparams',None)
        with open(os.path.join(self.output_dir, 'parameters.json'), 'w') as fp:
            json.dump(arguments, fp , indent=4)

    def evaluate(self, runs = 5, use_gui = False):
        """Tests the performance of the agent

            Returns evaluation logs
        """
        self.env.render(use_gui)

        evaluation_results = {
            "runs" : runs,
            "unfinished_runs" : 0,
            "average_delay" : [],
            "episode_mean_delays" : [],
            "episode_mean_delays_fixed" : []
            #"episode_delay_lists" : []
        }

        for i in range(runs):

            print('Evaluate {} -- running episode {} / {}'.format(self.connection_label,
                                                        i+1,
                                                        runs))


            all_trans, mean_delay, fixed_mean_delays = self.ddqn.evaluate(env = self.env, policy = "epsGreedy", eval_label = str(i), **{'eps':0.01})


            #evaluation_results["episode_delay_lists"].append(vehicle_delays)
            evaluation_results["episode_mean_delays"].append(mean_delay)
            evaluation_results["episode_mean_delays_fixed"].append(fixed_mean_delays)

            if mean_delay != -1:
                evaluation_results["average_delay"].append(mean_delay)
            else:
                evaluation_results["unfinished_runs"] += 1

        runs -= evaluation_results["unfinished_runs"]

        if runs == 0:
            evaluation_results["average_delay"].append(-1)
        else:
            evaluation_results["average_delay"] = sum(evaluation_results["average_delay"])/runs

        # print(self.ddqn.q_network.get_weights())

        return evaluation_results
