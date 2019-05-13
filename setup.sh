echo "Installing system dependencies for SUMO"
sudo apt-get update
sudo apt-get install -y python3.6 python3-pip
pip3 install numpy tensorflow keras jupyter


echo "Installing sumo binaries"
sudo apt-get install -y sumo sumo-tools sumo-doc
echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
source ~/.bashrc
