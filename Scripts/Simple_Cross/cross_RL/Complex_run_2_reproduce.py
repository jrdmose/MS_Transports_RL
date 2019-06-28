# IMPORTS
##########################

import agent
import environment
import doubledqn
import tools
import memory
import simulation
import multiprocessing
import pandas as pd
import os
import json

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import numpy as np
import matplotlib.pyplot as plt
import time
import itertools
from keras import optimizers

def iter_params(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

# REPRODUCEº
###############################################################

experiment_id = "Complex_run_2_reproduce"


param = {

    "batch_size" : [30], #,50],#,50],# 30,50,60,[70]
    "target_update_freq" : [5000],#,40000],# 100000],  #10000], # 10000,20000,30000,[40000]
    "gamma" : [0.99],#,0.95],# # 0.98,0.99,0.995,[0.999]
    "train_freq" : [1],#,4], # 2,3,4,[5]
    "max_size" : [10000],#,70000], # 20000,50000,70000,[100000]
    "max_ep_length" : [1000],
    "policy" : ["linDecEpsGreedy"],# "linDecEpsGreedy"],#, "epsGreedy"],
    "eps" : [0.3],#, 0.3, 0.1],
    "delta_time" : [10],#,10,10,10,10,10,10,10],
    "reward" : ["negative"],#,"negative"]
    "seed" : [1,5,8,12,20,24,50,61]
    #"optimizer": [optimizers.RMSprop(lr= 0.001), optimizers.Adagrad(), optimizers.Adam()]
}

param_grid = iter_params(**param)



def worker(input, output):
    """Runs through a chunk of the grid"""

    for position, args in iter(input.get, 'STOP'):
        result = worker_task(position, args)
        output.put(result)


def worker_task(position, args):
    """Tells the worker what to do with grid chunk"""
    # initialise all obje

    # print('Run', position + 1, '-- parameters', args)

    sumo_RL = simulation.simulator(
                    connection_label = position + 1,
                    q_network_type = 'simple',
                    target_q_network_type = 'simple',
                    gamma = args["gamma"],
                    target_update_freq = args["target_update_freq"],
                    train_freq = args["train_freq"],
                    num_burn_in = 1000,
                    batch_size = args["batch_size"],
                    optimizer = 'adam',
                    loss_func = "mse",
                    max_ep_length = args["max_ep_length"],
                    experiment_id = experiment_id,
                    model_checkpoint = True,
                    opt_metric = None,

                   # environment parameters
                    network = "complex", # complex
                    net_file = "complex_cross.net.xml", # "complex_cross.net.xml",
                    route_file = "cross.rou.xml",
                    demand = "rush",
                    state_shape = (1,41),#(1,41)
                    num_actions = 4, #4
                    use_gui = False,
                    delta_time = args["delta_time"],
                    reward = args["reward"],

                   # memory parameters
                    max_size = args["max_size"],

                   # additional parameters

                    policy = args["policy"],
                    eps = args["eps"],
                    num_episodes = 100,
                    monitoring = True,
                    episode_recording = False,
                    eval_fixed = True,
                    seed = args["seed"],
                    hparams = args.keys())


    # print("training agent", position + 1)
    train_data = sumo_RL.train()
    # print("evaluating agent", position + 1)
    evaluation_results = sumo_RL.evaluate(runs = 5)

    return ({"run" : position + 1,
             "args" : args,
             "eval_delay" : evaluation_results,
             "eval_mean_delay" : evaluation_results["average_delay"],
             "train_data": train_data})


def gridsearch(param_grid):
    """Runs a parallelised gridsearch"""

    number_of_processes = multiprocessing.cpu_count()

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
        multiprocessing.Process(target = worker,
                                args = (task_queue, done_queue)).start()

    with open(os.path.join("./logs",experiment_id,"GS_results.json"), "a") as file:
            file.write('{ "results": [')

    # Get and print results
    results = []
    for i in range(len(tasks)):
        results.append(done_queue.get())

        with open(os.path.join("./logs",experiment_id,"GS_results.json"), "a") as file:
            json.dump(results[-1], file , indent=4)
            if i != len(tasks)-1:
                file.write(",\n")

    with open(os.path.join("./logs",experiment_id,"GS_results.json"), "a") as file:
        file.write("]}")

        #print('%s -- [RESULTS]: Run %s -- Parameters %s -- Mean duration %6.0f' % results[-1])

    # Tell child processes to stop
    for i in range(number_of_processes):
        task_queue.put('STOP')

    # Now combine the results
#     scores = [result[-1] for result in results]
#     lowest = min(scores)
#     winner = results[scores.index(lowest)]
#    return winner, results

multiprocessing.freeze_support()
gridsearch(param_grid)
