from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
import os

env_name = 'BipedalWalker-v3'
env = gym.make(env_name, hardcore=False, render_mode = 'human')

episodes = 5

# print("Before Traning")
# for episode in range(episodes):
#     state = env.reset()
#     done = False
#     score = 0

#     while not done:
#         env.render()
#         action = env.action_space.sample()
#         n_steps, reward, truncated, terminated, info = env.step(action)
#         done = terminated or truncated
#         score += reward
#     print(f"Episode: {episode}, Score: {score}")

env = DummyVecEnv([lambda: env])

log_path = os.path.join("Training", "Logs")
model_path = os.path.join("Training", "Model")

# model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

# stop_callback = StopTrainingOnRewardThreshold(reward_threshold=300, verbose=1)
# eval_callback = EvalCallback(env, callback_on_new_best=stop_callback, eval_freq=5000, best_model_save_path=model_path, verbose=1)

# model.learn(total_timesteps=100000, callback=eval_callback)

model = PPO.load(os.path.join(model_path, "best_model.zip"), env)
print(evaluate_policy(model, env, n_eval_episodes=10))