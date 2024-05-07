import yaml
from argparse import Namespace
from stable_baselines3 import PPO, SAC, TD3
from env_wrappers import make_env
from planner import PurePursuitPlanner
import argparse
import time
import gym
from pathlib import Path
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from tqdm import tqdm
import os

with open('f1tenth_racetracks\example\config_example_map.yaml') as file:
    conf_dict = yaml.load(file, Loader=yaml.FullLoader)
conf = Namespace(**conf_dict)

# Create the planner instance
planner = PurePursuitPlanner(conf, (0.17145 + 0.15875))
MODEL_DIR = "models"
LOG_DIR = "logs"


def train(args):
    # Create the environment
    env = make_env('f110_gym:f110-v0', planner, max_steps=5000)

    # Create the PPO model
    model = PPO('MlpPolicy', env, verbose=1, learning_rate=0.0005, tensorboard_log="./logs/")
    eval_callback = EvalCallback(env, best_model_save_path='./train_test/',
                             log_path='./train_test/', eval_freq=5000,
                             deterministic=True, render=False)
    # Train the model
    model.learn(total_timesteps=500_000, callback= eval_callback, progress_bar=True)

    # Save the trained model
    train_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    model.save(f"./{MODEL_DIR}/f1tenth_ppo_model-{train_time}")

def test(args):
    model_path = Path(args.model_path)
    num_episodes = args.num_test_episodes
    total_reward = 0
    # create evaluation environment (same as train environment in this case)
    
    env = make_env('f110_gym:f110-v0', planner, max_steps=5000)
    # Load the trained model
    model = PPO.load(path=model_path)
    # Evaluate the model
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render()

    print(f"Total reward: {total_reward}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True, choices=["train", "test"])
    parser.add_argument(
        "--num_test_episodes",
        type=int,
        default=100,
        help="Number of episodes to test the model",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the model (.zip). If passed for training, the model is used as the starting point for training. If passed for testing, the model is used for inference.",
    )
    args = parser.parse_args()
    if args.run == "train":
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        train(args)
    elif args.run == "test":
        if args.model_path is None:
            raise ValueError("--model_path is required for testing")
        test(args)