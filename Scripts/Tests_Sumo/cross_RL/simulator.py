import agent
import environment
import doubledqn
import tools
import memory

import numpy as np
import matplotlib.pyplot as plt
import time
import itertools


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
                 q_network_type = 'simple',
                 target_q_network_type = 'simple',
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 batch_size,
                 optimizer,
                 loss_func,
                 max_ep_length,
                 output_dir,
                 experiment_id,
                 summary_writer,
                 model_checkpoint = True,
                 opt_metric = None,
                 # environment parameters
                 net_file,
                 route_file,
                 state_shape,
                 num_actions,
                 use_gui = False,
                 delta_time=10
                 # memory parameters
                 self,
                 max_size,
                 state_shape,
                 num_actions
                 # additional parameters
                 policy = "epsGreedy",
                 eps = eps,
                 num_episodes = 20):

        # Initialize Q-networks (value and target)
        self.q_network = agent.get_model(model_name = self.q_network_type,
                                input_shape = (self.state_shape[1],),
                                num_actions = self.num_actions)

        self.target_q_network = agent.get_model(model_name = self.target_q_network_type,
                                input_shape = (self.state_shape[1],),
                                num_actions = self.num_actions)

        # Initialize environment
        self.env =  environment.Env(net_file = self.net_file,
                                route_file =self.route_file,
                                state_shape = self.state_shape,
                                num_actions = self.num_actions,
                                use_gui = self.use_gui)

        # Initialize replay memory
        self.memory = memory.ReplayMemory(memory_size = self.memory_size,
                                state_shape = self.state_shape,
                                num_actions = self.num_actions)

        # Initialize Double DQN algorithm
        self.ddqn = doubledqn.DoubleDQN(     q_network = self.q_network,
                                target_q_network = self.target_q_network,
                                memory = self.memory,
                                gamma = self.gamma,
                                target_update_freq = self.target_update_freq,
                                num_burn_in = self.num_burn_in,
                                batch_size = self.batch_size,
                                optimizer = self.optimizer,
                                loss_func = self.loss,
                                max_ep_length = self.max_ep_length,
                                env_name = self.env,
                                output_dir = self.output_dir,
                                experiment_id = self.experiment_id,
                                summary_writer = self.summary_writer)

        # Fill memory
        self.ddqn.fill_replay(self.env)

        def train(num_episodes = self.num_episodes, eps = self.eps):
        """Trains the agent"""
            self.ddqn.train(env = self.env,
                            num_episodes = num_episodes,
                            policy = self.policy,
                            eps = eps)
            print("Agent trained")


        def evaluate(sumo_env = self.env, policy = self.policy, cv = 5):
        """Tests the performance of the agent"""

            mean_duration_cv = []

            for i in cv:
                all_trans, mean_duration = self.ddqn.evaluate(
                    sumo_env = self.env,
                    policy = self.policy)
                mean_duration_cv.append(mean_duration)

            return mean_duration_cv


        def _get_chunks(iterable, chunks=1):
        """Split parameter grid into chunks"""

            lst = list(iterable)
            return [lst[i::chunks] for i in range(chunks)]


        def take(n, iterable):
        """Return first n items of the iterable as a list"""

            return list(itertools.islice(iterable, n))


        def _worker(chunked_param_list):
        """Runs through a chunk of the grid"""

            worker_results = []
            worker_params = []
            for params in chunked_param_list:
                self.__init__()
                self.train()
                worker_results.append(self.evaluate())
                worker_params.append(params)

            return worker_results, worker_params


        def gridsearch(param_grid):
        """Runs a parallelised gridsearch"""

            jobs = []
            chunked_param_list = _get_chunks(param_grid, chunks = multiprocessing.cpu_count())
            pool = multiprocessing.Pool()
            results, parameters = pool.map(_worker, chunked_param_list)
            pool.close()
            pool.join()
            # Now combine the results
            print(results, parameters)
            max = max(results)
            max_params = parameters[results.index(max)]
            return max, max_params  # Winner
