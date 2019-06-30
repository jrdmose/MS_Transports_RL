# Deep reinforcement learning for the optimization of traffic light control with real-time data 

Master thesis of Data Science BGSE masters program

Authors: Monika Matyja, Sebastian Wolf and Jordi Morera

Supervisors: Hrvoje Stojic and Anestis Papanikolaou

### Thesis description

We develop a traffic light control agent that can manage traffic lights with the objective to reduce traffic jams, trip time and other traffic metrics in a given network using reinforcement learning. To this end, we implement a Double Deep Q-Network algorithm and test its performance in controlling traffic lights on a ’small’ and a ’large’ traffic junction. We find that this algorithm beats a fixed traffic light phase program when traffic demand fluctuates, as it is capable of reacting to real-time traffic situations. The algorithm can be scaled up and holds promise to also perform well in controlling larger transport networks. For more information check the report.

### Guide for folders:

- Scripts: Any script is needed to run the simulation. Also a couple of examples of how to run the code
- Report
- Videos: Recordings of an evaluation episode in split screen: DDQN agent vs benchmark policy

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
```
