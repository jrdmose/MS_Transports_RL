# Tools
####################################
import os, sys
import random
import numpy as np
import xml.etree.ElementTree as ET
import simulation

import multiprocessing
import itertools
import re
import glob
import json
import pandas as pd

def get_output_folder(output_dir, exp_id, args_description):
    """Return save folder output_dir/logs/exp_id

    If this directory already exists it creates parent_dir/logs/exp_id_{i},
    being i the next smallest free number.

    Inside this directory it also creates two sub-directories:
        - model_checkpoints to store model intermediate training steps.
        - tensorboard logs with name the args of the model

    This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    output_dir : str
        Path of the directory where results will be stored.

    Returns
    -------
    outputidir/logs/exp_id
        Path to this run's save directory.
    """
    try:
        # Returns an error if parent_dir already exists
        os.makedirs(output_dir)
    except:
        pass

    exp_run = 1
    if exp_id in os.listdir(output_dir):

        new_folder = os.path.join(output_dir,exp_id,"run"+"_"+str(exp_run))

        while os.path.exists(new_folder):
            exp_run +=1
            new_folder = os.path.join(output_dir,exp_id,"run"+"_"+str(exp_run))

        output_dir = new_folder
        os.makedirs(output_dir)
        os.mkdir(os.path.join(output_dir,"model_checkpoints"))
        summary_writer_folder = os.path.join(output_dir,args_description)
        os.mkdir(summary_writer_folder)
    else:
        output_dir = os.path.join(output_dir,exp_id,"run"+"_"+str(exp_run))
        os.makedirs(output_dir)
        os.mkdir(os.path.join(output_dir,"model_checkpoints"))
        summary_writer_folder = os.path.join(output_dir,args_description)
        os.mkdir(summary_writer_folder)


    return output_dir , summary_writer_folder



def get_veh_sec(x, demand,high, nominal,total_time):
    """
    Helper class for tunning the probabilities of a car entering in each lane
    in rush hour demand setup
    """


    factor = 10
    if demand == "rush":
        part = total_time/5
        if x < part:
            return np.random.normal(nominal, nominal/factor)
        if x < 2*part:
            aux = (nominal-high)/(-part)*x + nominal + (nominal-high)
            return np.random.normal(aux, aux/factor)
        if x < 3*part:
            return np.random.normal(high, high/factor)
        if x < 4*part:
            aux = -(high-nominal)/(part)*x + high+(high-nominal)*3
            return np.random.normal(aux, aux/factor)
        else:
            return np.random.normal(nominal, nominal/factor)
    else:
        return 1


def generate_routefile(route_file_dir, demand, network):
    """Returns XML file specifying demand file for sumo simulation"""

    N = 3600  # number of time steps

    nominal = 1
    high =3 # At rush hour two times more cars

    # demand per second from different directions

    pEW,pWE = tuple(np.repeat(1/15,2))
    pNS,pSN= tuple(np.repeat(1/60,2))
    if network == "complex":
        pEN,pES,pWN,pWS = tuple(np.repeat(1/30,4))
        pNE,pSE,pNW,pSW = tuple(np.repeat(1/120,4))
    else:
        pEN,pES,pWN,pWS,pNE,pSE,pNW,pSW = tuple(np.repeat(0,8))

    with open(route_file_dir, "w") as routes:
        print("""<routes>
        <vType id="car" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>
        <route id="WE" edges="-e78 -e07 e03 e34" />
        <route id="WS" edges="-e78 -e07 e05 e56" />
        <route id="WN" edges="-e78 -e07 e01 e12" />
        <route id="EW" edges="-e34 -e03 e07 e78" />
        <route id="ES" edges="-e34 -e03 e05 e56" />
        <route id="EN" edges="-e34 -e03 e01 e12" />
        <route id="SW" edges="-e56 -e05 e07 e78" />
        <route id="SE" edges="-e56 -e05 e03 e34" />
        <route id="SN" edges="-e56 -e05 e01 e12" />
        <route id="NW" edges="-e12 -e01 e07 e78" />
        <route id="NE" edges="-e12 -e01 e03 e34" />
        <route id="NS" edges="-e12 -e01 e05 e56" />""", file=routes)
        vehNr = 0

        for i in range(N):
            if random.uniform(0, 1) < pWE*get_veh_sec(i,demand,high,nominal,N):
                print('    <vehicle id="WE_%i" type="car" color="red" route="WE" depart="%i" departSpeed="max" departPos="last" departLane="best" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pWS*get_veh_sec(i,demand,high,nominal,N):
                print('    <vehicle id="WS_%i" type="car" route="WS" depart="%i" departSpeed="max" departPos="last" departLane="best" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pWN*get_veh_sec(i,demand,high,nominal,N):
                print('    <vehicle id="WN_%i" type="car" route="WN" depart="%i" departSpeed="max" departPos="last" departLane="best" />' % (
                    vehNr, i), file=routes)
                vehNr += 1


            if random.uniform(0, 1) < pEW*get_veh_sec(i,demand,high,nominal,N):
                print('    <vehicle id="EW_%i" type="car" color="red" route="EW" depart="%i" departSpeed="max" departPos="last" departLane="best" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pES*get_veh_sec(i,demand,high,nominal,N):
                print('    <vehicle id="ES_%i" type="car" route="ES" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pEN*get_veh_sec(i,demand,high,nominal,N):
                print('    <vehicle id="EN_%i" type="car" route="EN" depart="%i" departSpeed="max" departPos="last" departLane="best" />' % (
                    vehNr, i), file=routes)
                vehNr += 1


            if random.uniform(0, 1) < pSW*get_veh_sec(i,demand,high,nominal,N):
                print('    <vehicle id="SW_%i" type="car" route="SW" depart="%i" departSpeed="max" departPos="last" departLane="best" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pSE*get_veh_sec(i,demand,high,nominal,N):
                print('    <vehicle id="SE_%i" type="car" route="SE" depart="%i" departSpeed="max" departPos="last" departLane="best" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pSN*get_veh_sec(i,demand,high,nominal,N):
                print('    <vehicle id="SN_%i" type="car" color="red" route="SN" depart="%i" departSpeed="max" departPos="last" departLane="best" />' % (
                    vehNr, i), file=routes)
                vehNr += 1


            if random.uniform(0, 1) < pNW*get_veh_sec(i,demand,high,nominal,N):
                print('    <vehicle id="NW_%i" type="car" route="NW" depart="%i" departSpeed="max" departPos="last" departLane="best" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pNE*get_veh_sec(i,demand,high,nominal,N):
                print('    <vehicle id="NE_%i" type="car" route="NE" depart="%i" departSpeed="max" departPos="last" departLane="best" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pNS*get_veh_sec(i,demand,high,nominal,N):
                print('    <vehicle id="NS_%i" type="car" color="red" route="NS" depart="%i" departSpeed="max" departPos="last" departLane="best" />' % (
                    vehNr, i), file=routes)
                vehNr += 1

        print("</routes>", file=routes)


def get_vehicle_delay(output_dir, eval_label = 'tripinfo.xml'):
    """
    Once the simulation is done, it returns the total mean delay of the vehicles
    """

    tree = ET.parse(os.path.join(output_dir,eval_label))
    root = tree.getroot()

    vehicle_delay = []

    for veh in root:
        vehicle_delay.append(float(veh.get("duration")))

    return vehicle_delay


# functions for the gridsearch
# -worker
# -worker_task
# -gridsearch
# -iter_params


def worker(input, output):
    """Runs through a chunk of the grid"""

    for position, args in iter(input.get, 'STOP'):
        result = worker_task(position, args)
        output.put(result)


def worker_task(position, args):
    """Tells the worker what to do with grid chunk"""
    # print('Run', position + 1, '-- parameters', args)

    sumo_RL = simulation.simulator(connection_label = position +1, **args)

    # print("training agent", position + 1)
    train_data = sumo_RL.train()
    # print("evaluating agent", position + 1)
    evaluation_results = sumo_RL.evaluate(runs = 5)

    return ({"run" : position + 1,
             "args" : args,
             "eval_delay" : evaluation_results,
             "eval_mean_delay" : evaluation_results["average_delay"],
             "train_data": train_data})


def gridsearch(param_grid, log_path):
    """Runs a parallelised gridsearch"""

    multiprocessing.freeze_support()

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

    with open(os.path.join(log_path, "GS_results.json"), "w") as file:
            file.write('{ "results": [')

    # Get and print results
    results = []
    for i in range(len(tasks)):
        results.append(done_queue.get())

        with open(os.path.join(log_path, "GS_results.json"), "a") as file:
            json.dump(results[-1], file , indent=4)
            if i != len(tasks)-1:
                file.write(",\n")

    with open(os.path.join(log_path, "GS_results.json"), "a") as file:
        file.write("]}")

        #print('%s -- [RESULTS]: Run %s -- Parameters %s -- Mean duration %6.0f' % results[-1])

    # Tell child processes to stop
    for i in range(number_of_processes):
        task_queue.put('STOP')


def iter_params(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def get_grid_search_results(path):

    path = os.path.join(path,'GS_results.json')

    gs_results= {
        "run_id" : [],
        "unfinished_runs" : [],
        "RL_mean_delay" : [],
        "fixed_mean_delay" : [],
        "reward" : [],
        "policy" : [],
        "eps" : [],
        "update_freq" : []

    }

    with open(path) as file:
        data = json.load(file)

        for run in data['results']:

            gs_results["run_id"].append(run["run"])
            gs_results["unfinished_runs"].append(run["eval_delay"]["unfinished_runs"])
            gs_results["RL_mean_delay"].append(run["eval_delay"]["average_delay"])
            gs_results["fixed_mean_delay"].append(np.mean(run["eval_delay"]["episode_mean_delays_fixed"]))
            gs_results["reward"].append(run["args"]["reward"])
            gs_results["policy"].append(run["args"]["policy"])
            gs_results["eps"].append(run["args"]["eps"])
            gs_results["update_freq"].append(run["args"]["target_update_freq"])
    return pd.DataFrame(gs_results)


def load_last_model_checkpoint(logs_path,run):
    """
    Loads last model checkpoint of the training
    returns a simulation object
    """
    model_folder = os.path.join(logs_path, 'run_'+str(run), "model_checkpoints/")
    model_path = max(glob.iglob(model_folder+"/*.h5"), key = os.path.getmtime)
    sumo_RL = simulation.simulator()
    sumo_RL.load(model_path)

    return(sumo_RL)
