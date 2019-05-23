import agent
import environment
import doubledqn
import tools
import memory
import os

import tensorflow as tf
from keras import optimizers

class simulator:
    """Wrapper that handles all objects needed for a simulation:
    The neural network (agent.py), the decision function (doubledqn.py),
    the environment (environment.py), and the memory (memory.py).

    Attributes
    ----------
    *DDQN attributes*
    q_network : keras model instance to predict q-values for current state
    target_q_network : keras model instance to predict q-values for state after action
    memory : memory instance - needs to be instantiated first
    gamma : (int) discount factor for rewards
    target_update_freq : (int) defines after how many steps the q-network should be re-trained
    num_burn_in : (int) defines the size of the replay memory, using a specified policy
    batch_size : (int) size of batches to be used to train models
    trained_episodes : (int) episode counter
    max_ep_len : (int) stops simulation after specified number of episodes
    output_dir : (str) directory to write tensorboard log and model checkpoints
    experiment_id : (str) ID of simulation
    summary_writer : tensorboard summary stat writer instance
    itr : (int) counts global training steps in all run episodes

    *Environment attributes*
    net_file : (str) SUMO .net.xml file
    route_file : (str) SUMO .rou.xml file
    state_shape : (np.array) 2-dimensional array specifying state space dimensions
    num_actions : (int) specifying the number of actions available
    use_gui : (bool) Whether to run SUMO simulation with GUI visualisation
    delta_time : (int) Simulation seconds between actions

    *Memory attributes*
    memory: (list) list of class SingleSample containing memory of max_size of transitions
    max_size : (int) memory capacity required
    itr : (int) current index
    cur_size : (int) current size of memory

    *Additional parameters*
    policy: (str) name of the policy the agent should use
    eps: (int) exploration rate

    Methods
    -------
    __compile()
        Initialises all objects

    train(env, num_episodes, policy, **kwargs)
        Main method for the agent. Trains the keras neural network instances,
        calls all other helper methods.

    evaluate(env)
        Use trained agent to run a simulation without training.

    gridsearch(params)
        search the best parameters
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
                 optimizer = optimizers.RMSprop(lr=0.001, rho=0.95),
                 loss_func = "mse",
                 max_ep_length = 1000,
                 experiment_id = "Exp_1",
                 model_checkpoint = True,
                 opt_metric = None,
                 # environment parameters
                 net_file = "cross.net.xml",
                 route_file = "cross.rou.xml",
                 network_dir = "./network",
                 demand = "nominal",
                 state_shape = (1,11),
                 num_actions = 2,
                 use_gui = False,
                 delta_time = 10,
                 # memory parameters
                 max_size = 100000,
                 # additional parameters
                 policy = "linDecEpsGreedy",
                 eps = 0.1,
                 num_episodes = 2,
                 monitoring = False):

        # ddqn parameters
        self.connection_label = connection_label
        self.q_network_type = q_network_type
        self.target_q_network_type = target_q_network_type
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
        self.opt_metric = opt_metric

        # additional parameters
        self.policy = policy
        self.eps = eps
        self.num_episodes = num_episodes
        self.monitoring = monitoring
        self.output_dir = tools.get_output_folder("./logs", self.experiment_id)
        self.summary_writer = tf.summary.FileWriter(logdir = self.output_dir)

        # environment parameters
        self.net_file = os.path.join(network_dir, net_file)
        self.route_file = os.path.join(self.output_dir, route_file)
        self.demand = demand
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.use_gui = use_gui
        self.delta_time = delta_time

        # memory parameters
        self.max_size = max_size
        self.state_shape = state_shape

        # Initialize Q-networks (value and target)
        self.q_network = agent.get_model(model_name = self.q_network_type,
                                input_shape = (self.state_shape[1],),
                                num_actions = self.num_actions)

        self.target_q_network = agent.get_model(model_name = self.target_q_network_type,
                                input_shape = (self.state_shape[1],),
                                num_actions = self.num_actions)

        # Initialize environment
        self.env =  environment.Env(connection_label = self.connection_label,
                                net_file = self.net_file,
                                route_file = self.route_file,
                                demand =  self.demand,
                                state_shape = self.state_shape,
                                num_actions = self.num_actions,
                                use_gui = self.use_gui,
                                eps = self.eps)

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
                                env_name = self.env,
                                output_dir = self.output_dir,
                                monitoring = self.monitoring,
                                experiment_id = self.experiment_id,
                                summary_writer = self.summary_writer)


    def train(self):
        self.ddqn.fill_replay(self.env)
        self.ddqn.train(env = self.env,
                        num_episodes = self.num_episodes,
                        policy = self.policy,
                        connection_label = self.connection_label)
        # print(self.ddqn.q_network.get_weights())

    def load(self,checkpoint_dir):
        self.ddqn.load(checkpoint_dir)

    def evaluate(self,
                 runs = 5,
                 use_gui = False):
        """Tests the performance of the agent"""

        self.env.render(use_gui)

        mean_delays = []
        for i in range(runs):
            all_trans, mean_delay = self.ddqn.evaluate(env = self.env,
                                                        policy = "greedy")
            mean_delays.append(mean_delay)

        # print(self.ddqn.q_network.get_weights())

        return sum(mean_delays)/runs
