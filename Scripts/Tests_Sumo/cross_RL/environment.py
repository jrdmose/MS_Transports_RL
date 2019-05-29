# IMPORTS
##########################

import copy
import numpy as np
import tools

import time
import os, sys

from keras.models import Sequential
from keras.layers import InputLayer, Dense
import tensorflow as tf

# Making sure path to SUMO bins correctly specified
if 'SUMO_HOME' in os.environ:
    tools_sumo = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools_sumo)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import sumolib
import traci

# ENVIRONMENT CLASS
##################################
WARM_UP_NET = 20 # Number of simu steps to warm up the network

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
                 demand,
                 state_shape,
                 num_actions,
                 eps,
                 use_gui = False,
                 delta_time = 10,
                 connection_label = "lonely_worker",
                 reward = "balanced"):
        """Initialises object instance.

        Parameters
        ----------
        connection: (str) name of sumo connection
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
        self.demand = demand
        self.time_step = delta_time
        self.input_lanes = ["4i_0","2i_0","3i_0","1i_0"]
        self.connection_label = connection_label
        self.reward = reward

        self.render(self.use_gui)

        self.state = Observation(state_shape, self.input_lanes)
        self.action = Action(num_actions, eps)
        self.counter = np.zeros((1,2))

    def warm_up_net(self, num_it):
        """ Runs the environment for some iterations to fill the network.
        The network is filled with a static policy

        Parameters
        ----------

        num_it =  number of simulation steps to run
        """
        for i in range(num_it):
            action = self.action.select_action("fixed", state = self.state, v_row_t = 20, h_row_t = 40)
            self.step(action)


    def start_simulation(self, parent_dir = None ):
        """Opens a connection to sumo/traci [with or without GUI] and
        updates obs atribute  (the current state of the environment).
        """
        tools.generate_routefile(route_file_dir = self.route, demand = self.demand)

        sumo_cmd = [self.sumo_binary,
                    '-n', self.net,
                    '-r' ,self.route,
                    '--time-to-teleport', '-1']

        if parent_dir:
            sumo_cmd.append('--tripinfo-output')
            sumo_cmd.append(parent_dir + '/tripinfo.xml')


        traci.start(sumo_cmd, label = self.connection_label)
        # print('Started connection for worker #', self.connection_label)
        self.connection = traci.getConnection(self.connection_label)
        self.state.update_state(connection = self.connection)
        self.warm_up_net(WARM_UP_NET)
        self.state.update_state(connection = self.connection)

    def render(self,use_gui):
        if use_gui:
            self.sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self.sumo_binary = sumolib.checkBinary('sumo')

    def take_action(self, action):
        """Sets the action variable in sumo/traci to a new value.

        Parameters
        ----------
        action : (int) index of the one-hot encoded vector of action to be taken
        """

        #action = 0 -> row vertically
        if action == 0 and self.connection.trafficlight.getPhase("0") != 0:
            self.connection.trafficlight.setPhase("0",3)
            self.counter += self.state.get()[:,-2:]
            self.state.get()[:,-2:] = 0
        #action = 1 -> row horizontally
        elif action == 1 and self.connection.trafficlight.getPhase("0") != 2:
            self.connection.trafficlight.setPhase("0",1)
            self.counter += self.state.get()[:,-2:]
            self.state.get()[:,-2:] = 0
        else:
            try:
                action == -1 # Do nothing (for fixed policy)
            except:
                raise ValueError("Action ''{}'' not a valid action".format(action))

    def compute_reward(self, state, next_state):
        """ Computes reward from state and next_state.

        Parameters
        ----------
        state : (np.array) vector of current state
        next_state: (np.array) vector of next state
        """
        # Here is whre reward is specified
        diff = next_state - state
        difference = -np.sum(diff[:,15:])
        # b = np.round(state,decimals=1)
        # aux = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
        if self.reward == "negative":
            if difference < 0:
                token = difference
            else:
                token = 0
        else:
            token = difference
        return token # delta waiting time in the network

    # def compute_waiting_time(self):
    #     aux= []
    #     for lane in self.input_lanes:
    #         aux.append(self.connection.lane.getWaitingTime(lane))
    #     return np.array(aux)

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
        # wt = self.compute_waiting_time()

        self.take_action(action)

        self.connection.simulationStep(self.connection.simulation.getTime() + self.time_step) # Run the simulation time_step (s)
        self.state.update_state(connection = self.connection)
        next_state = self.state.get()

        # wt_next = self.compute_waiting_time()

        reward = self.compute_reward(state,next_state)

        # print("state", state, "next_state", next_state, "reward", reward)

        return state, reward, next_state, self.done()

    def done(self):
        """Calls sumo/traci to check whether there are still cars in the network"""
        return self.connection.simulation.getMinExpectedNumber() == 0

    def stop_simulation(self):
        """Closes the sumo/traci connection"""
        self.connection.close()
        # print('Stopped connection for worker #', self.connection_label)

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

    def __init__(self, shape, lanes):
        """
        Parameters
        ----------
        shape : (tuple) specifying dimensionality of state vector
        """

        self.obs = np.zeros(shape)
        self.lanes = lanes
        self.veh_time_in_lane = {lane:{} for lane in lanes}
        self.veh_waiting_time = {lane:{} for lane in lanes}

    def update_state(self, connection):
        """
        Parameters
        ----------
        lanes : (list) hardcoded list of lanes in network
        # TODO: read this from network file
        """

        for i,lane in enumerate(self.lanes):
            self.obs[:,i] = connection.lane.getLastStepOccupancy(lane) # Occupancy
            self.obs[:,i+4] = connection.lane.getLastStepMeanSpeed(lane)/19.44 # Average speed
            time, wt = self.compute_time_in_lane(connection, lane)
            # print ("time", time, "wt", wt)
            self.obs[:,i+11] = time
            self.obs[:,i+15] = wt

        self.obs[:,8] = connection.trafficlight.getPhase("0") # Traffic light phase
        if self.obs[:,8] == 0:
            # Amount of time phase 0 (vertical row) has been on since last phase change
            self.obs[:,9] = connection.trafficlight.getPhaseDuration("0") - (connection.trafficlight.getNextSwitch("0") - connection.simulation.getTime())
        elif self.obs[:,8] == 2:
            # Amount of time phase 2 (horizontal row) has been on since last phase change
            self.obs[:,10] = connection.trafficlight.getPhaseDuration("0") - (connection.trafficlight.getNextSwitch("0") - connection.simulation.getTime())

    def compute_time_in_lane(self, connection, lane):

        previous_veh_time_in_lane = copy.deepcopy(self.veh_time_in_lane[lane])
        veh_in_previous_state = set(previous_veh_time_in_lane.keys())
        veh_in_state = set(connection.lane.getLastStepVehicleIDs(lane))

        time = 0
        wt = 0

        intersection = veh_in_state & veh_in_previous_state
        for veh in intersection:
            self.veh_time_in_lane[lane][veh] += 10
            if connection.vehicle.getSpeed(veh) == 0:
                self.veh_waiting_time[lane][veh] += 10
                # print(veh, "is stopped")
            time += self.veh_time_in_lane[lane][veh]
            wt += self.veh_waiting_time[lane][veh]

        in_state = veh_in_state - veh_in_previous_state
        for veh in in_state:
            self.veh_time_in_lane[lane][veh] = 0
            self.veh_waiting_time[lane][veh] = 0

        in_previous_state = veh_in_previous_state - veh_in_state
        for veh in in_previous_state:
            self.veh_time_in_lane[lane].pop(veh)
            self.veh_waiting_time[lane].pop(veh)

        return time, wt

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

    def __init__( self, num_actions, eps):

        self.num_actions = num_actions
        self.action_space = np.identity(num_actions)
        self.eps = eps

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
        elif policy == "linDecEpsGreedy":
            return self.select_discepsgreedy(q_values, **kwargs)
        elif policy == "epsgreedy_decay":
            return self.select_epsgreedy_decay(q_values, **kwargs)
        else:
            raise ValueError("Policy {} not found".format(policy))

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

    def select_epsgreedy(self, q_values):
        """Feeds into select_action method.
        If explore, select action randomly,
        if exploit, select action greedily using the predicted q values

        Parameters
        ----------
        eps : (int) exploration paramter
        q_values : (np.array) predicted q-values
        """

        if np.random.uniform() < self.eps:
            return self.select_rand(q_values)
        else:
            return self.select_greedy(q_values)

    def select_fixed(self, q_values, state, v_row_t, h_row_t):
        """ Feeds into select_action method.
        It gives right of way horizontally h_row_t seconds.
        It gives right of way vertically v_row_t seconds.
        """
        #Vertical row
        if state.get()[:,9] > v_row_t:
            return 1
        #Horizontal row
        elif state.get()[:,10] > h_row_t:
            return 0
        else:
            return -1 # Do nothing

    def select_discepsgreedy(self, q_values, itr, final_eps = 0.1, total_it = 100000):
        """ eps-greedy policy with the eps decreasing linearly from start_eps to
            final_eps over total_it steps.
        """
        if itr < total_it:
            self.eps = (total_it - itr) / total_it * (self.eps - final_eps) + final_eps
        else :
            self.eps = final_eps
        return self.select_epsgreedy(q_values)

    def select_epsgreedy_decay(self, q_values, itr, omega = 1-1e-9):
        """ eps-greedy policy with the eps decreasing exponentially.
        """
        if self.eps < 0.2:
            self.eps = 0.2
        else :
            self.eps *= omega ** itr
        return self.select_epsgreedy(q_values)
