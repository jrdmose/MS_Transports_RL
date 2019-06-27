# Tools
####################################
import os, sys
import random
import numpy as np
import xml.etree.ElementTree as ET

def get_output_folder(output_dir, exp_id, args_description):
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


### TO DO
# Create a class to specify different types of demand


def get_veh_sec(x, demand,high, nominal,total_time):

    if demand == "rush":
        part = total_time/5
        if x < part:
            return np.random.normal(nominal, nominal/10)
        if x < 2*part:
            aux = (nominal-high)/(-part)*x + nominal + (nominal-high)
            return np.random.normal(aux, aux/10)
        if x < 3*part:
            return high
        if x < 4*part:
            aux = -(high-nominal)/(part)*x + high+(high-nominal)*3
            return np.random.normal(aux, aux/10)
        else:
            return np.random.normal(nominal, nominal/10)
    else:
        return 1

def get_veh_sec_eval(x, demand,high, nominal,total_time):

    if demand == "rush":
        part = total_time/5
        if x < part:
            aux = (nominal-high)/(-part)*x + nominal + (nominal-high)
            return np.random.normal(aux, aux/10)


            return np.random.normal(nominal, nominal/10)
        if x < 2*part:
            return high
        if x < 3*part:
            aux = -(high-nominal)/(part)*x + high+(high-nominal)*3
            return np.random.normal(aux, aux/10)
        if x < total_time:
            return np.random.normal(nominal, nominal/10)
    else:
        return 1

def generate_routefile(route_file_dir, demand, network):
    """Returns XML file specifying network layout for sumo simulation"""

    N = 3600  # number of time steps

    nominal = 1
    high =2 # At rush hour two times more cars

    # demand per second from different directions

    pEW,pWE = tuple(np.repeat(1/10,2))
    pNS,pSN= tuple(np.repeat(1/40,2))
    if network == "complex":
        pEN,pES,pWN,pWS = tuple(np.repeat(1/20,4))
        pNE,pSE,pNW,pSW = tuple(np.repeat(1/80,4))
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
                print('    <vehicle id="WE_%i" type="car" color="red" route="WE" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pWS*get_veh_sec(i,demand,high,nominal,N):
                print('    <vehicle id="WS_%i" type="car" route="WS" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pWN*get_veh_sec(i,demand,high,nominal,N):
                print('    <vehicle id="WN_%i" type="car" route="WN" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1


            if random.uniform(0, 1) < pEW*get_veh_sec(i,demand,high,nominal,N):
                print('    <vehicle id="EW_%i" type="car" color="red" route="EW" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pES*get_veh_sec(i,demand,high,nominal,N):
                print('    <vehicle id="ES_%i" type="car" route="ES" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pEN*get_veh_sec(i,demand,high,nominal,N):
                print('    <vehicle id="EN_%i" type="car" route="EN" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1


            if random.uniform(0, 1) < pSW*get_veh_sec(i,demand,high,nominal,N):
                print('    <vehicle id="SW_%i" type="car" route="SW" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pSE*get_veh_sec(i,demand,high,nominal,N):
                print('    <vehicle id="SE_%i" type="car" route="SE" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pSN*get_veh_sec(i,demand,high,nominal,N):
                print('    <vehicle id="SN_%i" type="car" color="red" route="SN" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1


            if random.uniform(0, 1) < pNW*get_veh_sec(i,demand,high,nominal,N):
                print('    <vehicle id="NW_%i" type="car" route="NW" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pNE*get_veh_sec(i,demand,high,nominal,N):
                print('    <vehicle id="NE_%i" type="car" route="NE" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pNS*get_veh_sec(i,demand,high,nominal,N):
                print('    <vehicle id="NS_%i" type="car" color="red" route="NS" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1

        print("</routes>", file=routes)


def get_vehicle_delay(output_dir, eval_label = 'tripinfo.xml'):

    tree = ET.parse(os.path.join(output_dir,eval_label))
    root = tree.getroot()

    vehicle_delay = []

    for veh in root:
        vehicle_delay.append(float(veh.get("duration")))

    return vehicle_delay
