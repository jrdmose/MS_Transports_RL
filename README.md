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

**Option 3**

 Pull Docker image
```
docker pull sebxwolf/sumo
```

 Run Docker container
```
docker run -d -p 5901:5901 -p 6901:6901 -p 8888:8888 sebxwolf/sumo
```
In your browser, open:
http://localhost:6901/?password=vncpassword

 In the Desktop that opens, go to Applications and open Terminal Emulator

 In the Terminal Emulator that opens, run
```
jupyter notebook --NotebookApp.token=admin --ip 0.0.0.0 --allow-root
```

 In your browser, open
localhost:8888/tree

 The password is 'admin' and you can run all your notebooks and tutorials

 (based on lucasfischerberkeley/flowdesktop)

### Running tensorboard
In terminal, run:
```
tensorboard --logdir ./Scripts/Tests_Sumo/cross_RL/Logs/[name of experiment]
```
Then open a browser and enter:
```
http://localhost:6006/
```
