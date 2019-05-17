# Tools
####################################
import os, sys
import random
import numpy as np
import xml.etree.ElementTree as ET
import simulation
import time
import itertools
import multiprocessing


def get_output_folder(parent_dir, exp_id):
    """Return save folder parent_dir/Results/exp_id

    If this directory already exists it creates parent_dir/Results/exp_id_{i},
    being i the next smallest free number.

    Inside this directory it also creates a sub-directory called model_checkpoints to
    store intermediate training steps.

    This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir : str
        Path of the directory where results will be stored.

    Returns
    -------
    parent_dir/Results/exp_id
        Path to this run's save directory.
    """
    try:
        # Returns an error if parent_dir already exists
        os.makedirs(parent_dir)
    except:
        pass

    if exp_id in os.listdir(parent_dir):

        experiment_id = 1
        new_folder = os.path.join(parent_dir,exp_id+"_"+str(experiment_id))

        while os.path.exists(new_folder):
            experiment_id +=1
            new_folder = os.path.join(parent_dir,exp_id+"_"+str(experiment_id))

        parent_dir = new_folder
        os.makedirs(parent_dir)
        os.mkdir(os.path.join(parent_dir,"model_checkpoints"))
    else:
        parent_dir = os.path.join(parent_dir,exp_id)
        os.makedirs(parent_dir)
        os.mkdir(os.path.join(parent_dir,"model_checkpoints"))

    return parent_dir


### TO DO
# Create a class to specify different types of demand

def generate_routefile():
    """Returns XML file specifying network layout for sumo simulation"""

    N = 3600  # number of time steps
    # demand per second from different directions

    pEW = 1 / 20
    pNS = 1 / 80
    pWE = 1 / 20
    pSN = 1 / 80

    with open("cross.rou.xml", "w") as routes:
        print("""<routes>
        <vType id="car" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>
        <route id="right" edges="51o 1i 2o 52i" />
        <route id="left" edges="52o 2i 1o 51i" />
        <route id="down" edges="54o 4i 3o 53i" />
        <route id="up" edges="53o 3i 4o 54i" />""", file=routes)
        vehNr = 0
        for i in range(N):
            if random.uniform(0, 1) < pWE:
                print('    <vehicle id="right_%i" type="car" route="right" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pEW:
                print('    <vehicle id="left_%i" type="car" route="left" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pNS:
                print('    <vehicle id="down_%i" type="car" route="up" depart="%i" color="1,0,0"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pSN:
                print('    <vehicle id="UP_%i" type="car" route="down" depart="%i" color="1,0,0"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1
        print("</routes>", file=routes)

def compute_mean_duration(parent_dir):

    tree = ET.parse(os.path.join(parent_dir,'tripinfo.xml'))
    root = tree.getroot()

    mean_duration = []
    for veh in root:
        mean_duration.append(float(veh.get("duration")))

    return np.mean(mean_duration)

def take(n, iterable):
    """Return first n items of the iterable as a list"""

    return list(itertools.islice(iterable, n))


def _worker(input, output):
    """Runs through a chunk of the grid"""

    for position, args in iter(input.get, 'STOP'):
        print('Started with position', position + 1, 'and parameters', args)
        result = _worker_task(position, args)
        output.put(result)


def _worker_task(position, args):
    """Tells the worker what to do with grid chunk"""
    print(position + 1, 'reached first assignment')
    # initialise all objects
    agent = simulation.simulator(connection_label = position)
    print('Initialised objects for position', position + 1, 'and parameters', args)
    # train ddqn
    agent.ddqn.train(env = agent.env,
                batch_size = args[0],
                target_update_frequency = args[1],
                gamma = args[2],
                eps = args[3])

    print('Trained agent for position', position + 1, 'and parameters', args)

    result = agent.evaluate()

    return '%s evaluated at position %s with parameters %s gives result %s' % \
        (multiprocessing.current_process().name, position + 1, args, result)


def gridsearch(param_grid):
    """Runs a parallelised gridsearch"""

    number_of_processes = 1 #multiprocessing.cpu_count()

    # Set up task list
    tasks = [(idx, val) for idx, val in enumerate(param_grid)]

    # Create queues
    task_queue = multiprocessing.Queue()
    done_queue = multiprocessing.Queue()

    # Submit tasks
    for task in tasks:
        task_queue.put(task)

    # Start worker processes
    for i in range(number_of_processes):
        print('Started process #', i + 1)
        multiprocessing.Process(target = _worker,
                                args = (task_queue, done_queue)).start()

    # Get and print results
    print('Unordered results:')
    for i in range(len(tasks)):
        print('\t', done_queue.get())

    # Tell child processes to stop
    for i in range(number_of_processes):
        task_queue.put('STOP')

    # Now combine the results
    # max = max(done_queue)
    # max_params = parameters[results.index(max)]
    return done_queue # Winner
#
#
# if __name__ == '__main__':
#     multiprocessing.freeze_support()
