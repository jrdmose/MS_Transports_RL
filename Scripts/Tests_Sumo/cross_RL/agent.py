# Q NETWORKS
################################
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Flatten


# More ANN arquitectures to be specified here
def get_model(model_name, *args):
    """Helper function to instantiate q-networks.

    Parameters
    ----------
    model_name : (str) Name of the network architecture to be used to instantiate q-network
    *args : arguments to be passed onto helper functions
    """
    if model_name == 'linear':
        return linear(*args)
    elif model_name == 'simple':
        return simple(*args)


def linear(input_shape, num_actions):
    """Feeds into get_model. Sets up a linear keras model instance.

    Parameters
    ----------
    input_shape : (np.array) shape of the state vector to be fed into the model as input layer
    num_actions : (int) number of nodes of the output layer
    """
    model = Sequential
    model.add(Flatten(input_shape=input_shape, name="Layer_1")) # If a vector the flattening does not have any effect. (only matrices)
    model.add(Dense(num_actions,activation=None, name = "Output"))
    return model


def simple(input_shape, num_actions):
    '''Feeds into get_model. Sets up a neural network keras model instance.

    Parameters
    ----------
    input_shape : (np.array) shape of the state vector to be fed into the model as input layer
    num_actions : (int) number of nodes of the output layer
    '''
    model = Sequential()
    model.add(Dense(9, input_shape = input_shape, activation='relu',name = "Layer_1"))
    model.add(Dense(9, activation='relu', name= "Layer_2"))
    model.add(Dense(num_actions, activation=None, name= "Output"))
    return model
