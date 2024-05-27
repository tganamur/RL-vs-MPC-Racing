# Learning to Race using RL PPO and MPC
## Project Overview 
F1Tenth Reinforcement Learning vs Model Predictive Control
This project compares the performance of a Reinforcement Learning (RL) agent to a Model Predictive Control (MPC) approach for autonomous racing on the F1Tenth simulated environment.
Overview
The goal was to develop an autonomous racing agent that can complete laps around the F1Tenth example track as quickly as possible. Two main approaches were explored in parallel:

__Reinforcement Learning (RL)__: An agent was trained using stable_baselines3's Proximal Policy Optimization (PPO) on the F1Tenth gym environment. The final agent achieved a lap time of 16.58 seconds on the example track.  

__Model Predictive Control (MPC)__: Two different vehicle dynamics models were implemented - a unicycle model and a kinematic bicycle model. The MPC solutions tracked the center line waypoints while optimizing for minimum lap time under constraints like steering limits. The unicycle model achieved 8.7 seconds and the kinematic bicycle model 10.81 seconds on the example track.

The project involved setting up the simulation environments, defining reward functions, tuning hyperparameters, and addressing challenges related to dynamics modeling and constraint handling. This project was a collaboration between myself, @EdwardShiBerkeley, @FahimChoudhury007, and @yashopadmin, 

## How to Run (Reinforcement Learning)
First clone the repo. 
In a terminal window, create a new conda envirnoment, activate it, and install the required packages.
``` sh
conda create -n <name_of_env> python=3.8
conda activate <name_of_env>
cd ME292B_FinalProject
pip install -r requirements.txt
```
### Training (Train using PPO)
```sh
python rl_train_test.py --run train 
```
### Testing (Test best saved model from training)
```sh 
python rl_train_test.py --run test --model_path <path/to/model>
```
### Quick Run Best Model (Best results)
```sh
python rl_best_model.py 
```
## How to Run (Model Predictive Control)
```sh
MPC_Laptime_Racing.ipynb
```
Run all the cells in order:\
-For the 4th cell ensure that the file paths are correct to not have any errors\
-For the 6th cell choose the track and dynamics model\
-The 7th cell has the results of the MPC plotted out 
# Results
The results from RL and MPC are shown below. The RL agent finishes the track in 16.58 seconds, around the example track. The bicycle kinematic dynamic based MPC finished the example track in 10.81 seconds. 
## RL Result

<img src="https://github.com/tganamur/ME292B_FinalProject/blob/main/rl_training_clip.gif" width="500" height="406">\
RL Training Result on Example Track (250,000 training steps)

<img src="https://github.com/tganamur/ME292B_FinalProject/blob/main/rl_result.gif" width="500" height="365">\
RL Result on Example Track (5,000,000 training steps, following raceline)

## MPC Result
<img src="https://github.com/tganamur/ME292B_FinalProject/blob/main/mpc_result_exampleTrack.png" width="500" height="390">\
<img src="https://github.com/tganamur/ME292B_FinalProject/blob/main/mpc_result_BrandsHatch.png" width="500" height="390">\
<img src="https://github.com/tganamur/ME292B_FinalProject/blob/main/MPC_result_IMS_1.png" width="400" height="800">\
Results for MPC in three tracks: Example Track, Brands Hatch and IMS

## Acknowledgements
1. F1Tenth Gym Environment and documentation [F1Tenth](https://github.com/f1tenth/f1tenth_gym)
2. Stable Baselines3 documentation and examples of PPO RL method usage.

** All other citations are in the project report under "References". 

