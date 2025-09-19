# Anomalous-living-crystals

This repository is currently a work in progress where I will gradually upload the files used to simulate active Brownian particles in a heterogenous environment. For now, I include a brief description of the files and the contents:

## Simulator.cpp
This file contains the main Simulator class. This handles simulation of the particles and generally can be run in time or event driven fashion.

## bindings.py
This file sets up the possibility of communication from of the Simulator.cpp with a python script, using a simple import statement. The main aim of this approach is to enable streamlined reinforcement learning framework to run in Python with the more efficient c++ code (Simulator) doing the simulation heavy lifting. The communication is set up using Pybind (documentation and the relevant commmand arguments to enable compilation can be found here: https://pybind11.readthedocs.io/en/stable/index.html). 

## RL_actor_critic.py
Reinforcement learning framework which handles interaction with the Simulator and training of the model. 

## environment.py
Calculating rewards for the simulation.

Since the above is pending publication, for now, only the Simulator class is available. The rest of the repository will be filled out and committed once available. Establishing this repository has been a continual learning process in all aspects. Any comments and suggestions as to the implementation details are welcome and appreciated. 
