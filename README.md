# ME292B Final Project (Group 9)
## Project Overview 
The project is centered around comparing the performance of a Reinforcement Learning (RL) based method to a Model Predictive Control (MPC) Method around the same track, the example track as provided by F1Tenth. The RL portion used stable_baselines3's PPO algorithm to train an agent to race around the track, using the F1Tenth gym envirnoment to aid in training and testing. The MPC part was implemented in Google colab using jupyter notebooks. 
## How to Run
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
## Results
### RL Result

<img src="https://github.com/tganamur/ME292B_FinalProject/blob/main/rl_training_clip.gif" width="500" height="406">\
RL Training Result on Example Track (250,000 training steps)

<img src="https://github.com/tganamur/ME292B_FinalProject/blob/main/rl_result.gif" width="500" height="365">\
RL Result on Example Track (5,000,000 training steps, following raceline)

### MPC Result

<img src="https://github.com/tganamur/ME292B_FinalProject/blob/main/mpc_result_exampleTrack.png" width="500" height="390">\
<img src="https://github.com/tganamur/ME292B_FinalProject/blob/main/mpc_result_IMS.png" width="500" height="390">\
<img src="https://github.com/tganamur/ME292B_FinalProject/blob/main/mpc_result_BrandsHatch.png" width="500" height="390">\
Results for MPC in three tracks: Example Track, IMS and Brands Hatch

## Acknowledgements
1. F1Tenth Gym Environment and documentation [F1Tenth](https://github.com/f1tenth/f1tenth_gym)
2. Stable Baselines3 documentation and examples of PPO RL method usage. 
## Citations
1. [F1Tenth](https://github.com/f1tenth/f1tenth_gym)
2. [Stable_baseline3](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#example)

### Instructions for final project submission
Please use this public repository as the template and name it "Group_X". For example, Group 2 should create a private repository in the ME292B organization and name it "Group_2".

Each group should submit the following files through GitHub:
 - Final project report: format, PDF.
 - Code, if available: format, ZIP.
 - Slides: format, PPT/PPTX/PDF.

Please use this template for final project report, [Final project report template](https://www.overleaf.com/read/ynbtcmrnwnkp#5cb5c7). It should be no longer than 7 pages, excluding references.

Note that in the final report and slides, each group should specify the parts each member has completed and highlight the contributions of each member. We grade each student based on the group project and individual contributions within the group. The code should include instructions on how to run it and reproduce the results.

