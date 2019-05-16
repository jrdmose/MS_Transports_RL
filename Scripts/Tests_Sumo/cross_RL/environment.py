# IMPORTS
##########################

import copy
import numpy as np

import time
import os, sys

from keras.models import Sequential
from keras.layers import InputLayer, Dense
import tensorflow as tf

# Making sure path to SUMO bins correctly specified
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import sumolib
import traci

# ENVIRONMENT CLASS
##################################


class Env:
    """Main class to manage environment. Supplies the environment responses to actions taken.

    Sends commands to sumo to take an action, receives state information,
    computes rewards. Does so step by step to allow q-network to train batch-wise.

    Attributes
    ----------
    net : (str) points to the SUMO .net.xml file which specifies network arquitechture
    route : (str) points to the SUMO .rou.xml file which specified traffic demand
    use_gui : (bool) Whether to run SUMO simulation with GUI visualisation
    time_step : (int) Simulation seconds between actions
    sumo_binary : (str) Points to the binary to run sumo
    num_actions : (int) number of actions (traffic signal phases)

    Methods
    -------
    start_simulation()
        Opens call to sumo

    take_action(action)
        Sets the traffic lights according to the action fed as argument

    compute_reward(state, next_state)
        Takes the current state and next state (from object state) and computes reward

    step(action)
        Combines take_action, compute_reward and update_state (from observation object)

    done()
        checks whether all links are empty, and hence the simulation done

    stop_simulation()
        closes the sumo/traci call
    """

    def __init__(self,
                 net_file,
                 route_file,
                 state_shape,
                 num_actions,
                 use_gui = False,
                 delta_time=10):
        """Initialises object instance.

        Parameters
        ----------
        net_file : (str) SUMO .net.xml file
        route_file : (str) SUMO .rou.xml file
        state_shape : (np.array) 2-dimensional array specifying state space dimensions
        num_actions : (int) specifying the number of actions available
        use_gui : (bool) Whether to run SUMO simulation with GUI visualisation
        delta_time : (int) Simulation seconds between actions
        """
        self.net = net_file
        self.route = route_file
        self.use_gui = use_gui
        self.time_step = delta_time
        self.input_lanes = ["4i_0","2i_0","3i_0","1i_0"]

        if self.use_gui:
            self.sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self.sumo_binary = sumolib.checkBinary('sumo')

        self.state = Observation(state_shape, self.input_lanes)

        self.action = Action(num_actions)
        self.counter = np.zeros((1,2))


    def start_simulation(self, parent_dir = None ):
        """Opens a connection to sumo/traci [with or without GUI] and
        updates obs atribute  (the current state of the environment).
        """

        sumo_cmd = [self.sumo_binary,
                    '-n', self.net,
                    '-r' ,self.route]

        if parent_dir:
            sumo_cmd.append('--tripinfo-output')
            sumo_cmd.append(parent_dir + '/tripinfo.xml')

        if self.use_gui:
            sumo_cmd.append('--start')
        traci.start(sumo_cmd)
        self.state.update_state()


    def take_action(self,action):
        """Sets the action variable in sumo/traci to a new value.

        Parameters
        ----------
        action : (int) index of the one-hot encoded vector of action to be taken
        """

        #action = 0 -> row vertically
        if action == 0 and traci.trafficlight.getPhase("0") != 0:
            traci.trafficlight.setPhase("0",3)
            self.counter += self.state.get()[:,-2:]
            self.state.get()[:,-2:] = 0
        #action = 1 -> row horizontally
        elif action == 1 and traci.trafficlight.getPhase("0") != 2:
            traci.trafficlight.setPhase("0",1)
            self.counter += self.state.get()[:,-2:]
            self.state.get()[:,-2:] = 0

    def compute_reward(self, state, next_state):
        """ Computes reward from state and next_state.

        Parameters
        ----------
        state : (np.array) vector of current state
        next_state: (np.array) vector of next state
        """
        # Here is whre reward is specified
        diff = next_state - state
        # b = np.round(state,decimals=1)
        # aux = np.divide(a, b, out=np.zeros_like(a), where=b!=0)

        return -np.sum(diff) # delta waiting time in the network

    def compute_waiting_time(self):
        aux= []
        for lane in self.input_lanes:
            aux.append(traci.lane.getWaitingTime(lane))
        return np.array(aux)

    def step(self, action):
        """ Runs one step of the simulation.

        Makes a deep copy of the current state,
        runs the simulation with the currently set action,
        gets the state after the action,
        and computes the reward associated to action taken.

        Parameters
        ----------
        action : (int) index of the one-hot encoded vector of action to be taken
        """

        state = copy.deepcopy(self.state.get())
        wt = self.compute_waiting_time()

        self.take_action(action)
        traci.simulationStep(traci.simulation.getTime() + self.time_step) # Run the simulation time_step (s)
        self.state.update_state()
        next_state = self.state.get()

        wt_next = self.compute_waiting_time()
        reward = self.compute_reward(wt,wt_next)

        return state, reward, next_state, self.done()

    def done(self):
        """Calls sumo/traci to check whether there are still cars in the network"""
        return traci.simulation.getMinExpectedNumber() == 0

    def stop_simulation(self):
        """Closes the sumo/traci connection"""
        traci.close()

class Observation:
    """
    Helper class for environment. Handles the updating of the state. Ports the state from sumo/traci to python.

    Attributes
    ----------
    obs : (np.array) holds the state variables

    Methods
    -------
    update_state()
        reads the state through a call to sumo/traci
            --> modify here to include additional state variables,
            but make sure to also modify state_shape when running the simulation!

    get()
        returns state

    get_reward()
        returns reward, computed from current state
    """

    def __init__(self, shape,lanes):
        """
        Parameters
        ----------
        shape : (tuple) specifying dimensionality of state vector
        """

        self.obs = np.zeros(shape)
        self.lanes = lanes

    def update_state(self):
        """
        Parameters
        ----------
        lanes : (list) hardcoded list of lanes in network
        # TODO: read this from network file
        """

        for i,lane in enumerate(self.lanes):

            self.obs[:,i] = traci.lane.getLastStepHaltingNumber(lane) # Occupancy
            self.obs[:,i+4] = traci.lane.getLastStepMeanSpeed(lane) # Average speed

        self.obs[:,8] = traci.trafficlight.getPhase("0") # Traffic light phase
        if self.obs[:,8] == 0:
            # Amount of time phase 0 (vertical row) has been on since last phase change
            self.obs[:,9] = traci.trafficlight.getPhaseDuration("0") - (traci.trafficlight.getNextSwitch("0") - traci.simulation.getTime())
        elif self.obs[:,8] == 2:
            # Amount of time phase 2 (horizontal row) has been on since last phase change
            self.obs[:,10] = traci.trafficlight.getPhaseDuration("0") - (traci.trafficlight.getNextSwitch("0") - traci.simulation.getTime())

    def get(self):
        """Returns state vector"""
        return self.obs

class Action:
    """
    Helper class for observation. One-hot encoding of the phase of the traffic signal.
    The methods for this class handle the selection of the best action.

    Methods
    -------
    select_action(policy, **kwargs)
        Takes policy as argument, and then calls the corresponding method.
        # TODO: assert that the kwargs being fed correspond to the policy selected,
        and handle errors

    select_rand()
        Select one of the actions randomly.

    select_greedy(q-values)
        Check which action corresponds to the highest predicted reward/q-value.

    select_epsgreedy(eps, q_values)
        Choose whether to explore or exploit.
    """

    def __init__( self, num_actions):

        self.num_actions = num_actions
        self.action_space = np.identity(num_actions)

    def select_action(self, policy, q_values = None, **kwargs):
        """Takes policy as argument, and then calls the corresponding helper method.

        Parameters
        ----------
        policy : (string) indicating which policy to consider.
            Currently implemented:
                - Pick action randomly ("rand")
                - Pick action greedely ("greedy")
                - Pick action in a eps - greedy fashion ("epsGreedy")
                - Pick action statically.

        **kwargs : arguments for helper methods
        """

        if policy == "randUni":
            return self.select_rand(q_values) #q values not used
        elif policy == "greedy":
            return self.select_greedy(q_values, **kwargs)
        elif policy == "epsGreedy":
            return self.select_epsgreedy(q_values, **kwargs)
        elif policy == "fixed":
            return self.select_fixed(q_values, **kwargs) #q values not used

    def select_rand(self, q_values):
        """Feeds into select_greedy or directly into select_action method.
        Selects one of the actions randomly."""

        return np.random.randint(0, self.num_actions)

    def select_greedy(self, q_values):
        """Feeds into select_epsgreedy or directly into select_action method.
        Checks which action corresponds to the highest predicted reward/q-value.

        Parameters
        ----------
        q_values : (np.array) predicted q-values
        """

        return np.argmax(q_values)

    def select_epsgreedy(self, q_values, eps):
        """Feeds into select_action method.
        If explore, select action randomly,
        if exploit, select action greedily using the predicted q values

        Parameters
        ----------
        eps : (int) exploration paramter
        q_values : (np.array) predicted q-values
        """

        if np.random.uniform() < eps:
            return self.select_rand(q_values)
        else:
            return self.select_greedy(q_values)

    def select_fixed(self, q_values,env, v_row_t, h_row_t):
        """ Feeds into select_action method.
        It gives right of way horizontally h_row_t seconds.
        It gives right of way vertically v_row_t seconds.
        """
        #Vertical row
        if env.state.get()[:,9] > v_row_t:
            return 1
        #Horizontal row
        elif env.state.get()[:,10] > h_row_t:
            return 0

    def select_discepsgreedy(self, q_values, eps, itr):
        pass
