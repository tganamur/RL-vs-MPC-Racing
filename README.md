## Instructions for final project submission
Please use this public repository as the template and name it "Group_X". For example, Group 2 should create a private repository in the ME292B organization and name it "Group_2".

Each group should submit the following files through GitHub:
 - Final project report: format, PDF.
 - Code, if available: format, ZIP.
 - Slides: format, PPT/PPTX/PDF.

Please use this template for final project report, [Final project report template](https://www.overleaf.com/read/ynbtcmrnwnkp#5cb5c7). It should be no longer than 7 pages, excluding references.

Note that in the final report and slides, each group should specify the parts each member has completed and highlight the contributions of each member. We grade each student based on the group project and individual contributions within the group. The code should include instructions on how to run it and reproduce the results.
## Project Overview 
## How to Run
### Training 
python rl_train_test.py --run train 
### Testing 
python rl_train_test.py --run test --model_path <path/to/model>
### Quick Run Best Model
python rl_best_model.py 

## Acknowledgements
1. F1Tenth Gym Environment and documentation [F1Tenth](https://github.com/f1tenth/f1tenth_gym)
2. Stable Baselines3 documentation and examples of PPO RL method usage. 
## Citations
1. [F1Tenth](https://github.com/f1tenth/f1tenth_gym)
2. [Stable_baseline3](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#example)
