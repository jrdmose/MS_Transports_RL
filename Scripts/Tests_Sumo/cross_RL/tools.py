# Tools
####################################
import os, sys
import random
import numpy as np
import xml.etree.ElementTree as ET

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

    experiment_id = 1
    if exp_id in os.listdir(parent_dir):

        new_folder = os.path.join(parent_dir,exp_id,"run"+"_"+str(experiment_id))

        while os.path.exists(new_folder):
            experiment_id +=1
            new_folder = os.path.join(parent_dir,exp_id,"run"+"_"+str(experiment_id))

        parent_dir = new_folder
        os.makedirs(parent_dir)
        os.mkdir(os.path.join(parent_dir,"model_checkpoints"))
    else:
        parent_dir = os.path.join(parent_dir,exp_id,"run"+"_"+str(experiment_id))
        os.makedirs(parent_dir)
        os.mkdir(os.path.join(parent_dir,"model_checkpoints"))

    return parent_dir


### TO DO
# Create a class to specify different types of demand

def generate_routefile(parent_dir):
    """Returns XML file specifying network layout for sumo simulation"""

    N = 3600  # number of time steps
    # demand per second from different directions

    pEW = 1 / 20
    pNS = 1 / 80
    pWE = 1 / 20
    pSN = 1 / 80

    if parent_dir == None:
        route_file_dir = "cross.rou.xml"
    else:
        route_file_dir = os.path.join(parent_dir,"cross.rou.xml")

    with open(route_file_dir, "w") as routes:
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