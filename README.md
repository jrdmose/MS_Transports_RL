# Deep Reinforcement learning applied to transport network

Master thesis of Data Science BGSE masters program

Authors: Monika Matyja, Sebastian Wolf and Jordi Morera

Supervisors: Hrvoje Stojic and Anestis Papanikolaou

### Thesis description

### Guide for folders:

- MoM : Minutes of meeting of the weekly talks
- References : Important references

- Data: Network and demand files needed for SUMO simulations
- Scripts: Any script is needed to run the simulation.
- Report

### Installation guide

**Option 1**

Run setup.sh 

```
bash setup.sh
```

**Option 2**

Install sumo binaries

```
sudo apt-get install sumo sumo-tools sumo-doc 
```

Set SUMO_HOME variable (default sumo installation path is /usr/share/sumo)

```
echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
source ~/.bashrc
```
Installing required packages

```
pip3 install numpy tensorflow keras
```
**Tensorboard set-up**

Define the path of the logs

```
tensorboard --logdir ./logs
```
