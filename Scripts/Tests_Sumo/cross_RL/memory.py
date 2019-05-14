# MEMORY CLASS
##################################################
import numpy as np


class ReplayMemory():
    """
    Keeps a memory of transitions the environment has revealed as response to actions taken.

    Keeps a memory of size max_size, implemented through lists of class SingleSample: state, action, reward and nextstate.
    An index counter is pushed forward, through the list as the memory is filled. Once memory is full
    the index starts from 0, and overwrites existing memory. The index counter is also used to indicate
    which one the most recent memory is, to allow prioritising more recent memory over older memory.

    Attributes
    ----------
    memory: (list) list of class SingleSample containing memory of max_size of transitions
    max_size : (int) memory capacity required
    itr : (int) current index
    cur_size : (int) current size of memory

    Methods
    ------
    append(state, action, reward, nextstate)
        adds new elements to the four lists, handles index counter

    sample(batch_size)
        Randomly draws a sample of batch_size of transitions. It returns 2 arrays and two lists.
        The arrays correspond to state and next state of the transitions with dimensions
        (batch_size, space_shape).The lists correspond to actions and rewards of those transitions.

    print_obs(obs)
        prints a specific transition

    get_size
        returns the current size of the memory
    """

    def __init__(self, max_size, state_shape, num_actions):
        """Initialize the whole memory at once.

        Parameters
        ----------
        max_size : (int) memory capacity required
        state_shape : (tuple) tuple specifying the shape of the array in which state variables are stored.
        num_actions : (int) number of actions (traffic signal phases)
        """

        self.memory = [SingleSample(state_shape,num_actions) for _ in range(max_size)]
        self.max_size = max_size
        self.itr = 0  # insert the next element here
        self.cur_size = 0


    def append(self, state, action, reward, nextstate):
        """Adds new elements to the four lists, handles index counter.

        Parameters
        ----------
        state : (np.array) all state variables for one observation
        action : (int) index of the one-hot encoded vector indicating action
        reward : (float) reward after action taken
        nextstate : (np.array) all state variables for one observation, after action taken
        """

        self.memory[self.itr].assign(state, action, reward, nextstate)
        self.itr += 1
        self.cur_size = min(self.cur_size + 1, self.max_size)
        self.itr %= self.max_size


    def sample(self, batch_size):
        """Uniform sampling, later prioritized experience replay can be implemented.

        Parameters
        ----------
        batch_size : (int) size of the batch to be sampled
        """

        states, actions, rewards, next_states = [],[],[],[]
        for i, idx in enumerate(np.random.randint(0, self.cur_size, size=batch_size)):
            transition = self.memory[idx]
            states.append(transition.state)
            actions.append(transition.action)
            rewards.append(transition.reward)
            next_states.append(transition.nextstate)
        return np.vstack(states), actions, rewards, np.vstack(next_states)


    def print_obs(self,obs):
        """Selects a specific transition to view.

        Parameters
        ----------
        obs : (int) index of the specific transition to view
        """

        self.memory[obs].print_obs() # This calls a SingleSample method called also print_obs


    def get_size(self):
        """Shows current size of memory"""

        return self.cur_size



class SingleSample():
    """A helper for the memory class. It stores single transition objects.

    Attributes
    ----------
    state : (np.array) all state variables for one observation
    action : (int) index of the one-hot encoded vector indicating action
    reward : (int) reward after action taken
    nextstate : (np.array) all state variables for one observation, after action taken

    Methods
    -------
    assign(self, state, action, reward, nextstate)
        Assigns new values to attributes.

    print_obs()
        Prints current observation/ transition.
    """

    def __init__(self, state_shape, num_actions): # Num actions not used up to now
        """Initialises object instance.
        Parameters
        ----------
        state_shape : (tuple) (tuple) tuple specifying the shape of the array in which state variables are stored.
        num_actions : (int) number of actions (traffic signal phases)
        """

        self.state = np.zeros(state_shape)
        self.action = 0
        self.reward = 0
        self.nextstate = np.zeros(state_shape)


    def assign(self, state, action, reward, nextstate):
        """Assigns new values to attributes.

        Parameters
        ----------
        state : (np.array) all state variables for one observation
        action : (np.array) index of the one-hot encoded vector indicating action
        reward : (int) reward after action taken
        nextstate : (np.array) all state variables for one observation, after action taken
        """

        self.state[:] = state
        self.action = action
        self.reward = reward
        self.nextstate[:] = nextstate


    def print_obs(self):
        """Prints current observation"""

        print( "State: \n\n",self.state,
               "\n\nAction:\n\n",self.action,
               "\n\nReward:\n\n",self.reward,
               "\n\nNext State:\n\n",self.nextstate)
