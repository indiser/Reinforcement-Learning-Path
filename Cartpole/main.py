from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import gymnasium as gym
import os

env_name = 'CartPole-v1'
env = gym.make(env_name, render_mode = "human")

# episodes = 5

# for episode in range(episodes):
#   state = env.reset()
#   done = False
#   score = 0

#   while not done:
#     env.render()
#     action = env.action_space.sample()
#     n_state, reward, terminated, truncated, info = env.step(action)
#     done = terminated or truncated
#     score += reward
#   print(f'Episode: {episode} Score: {reward}')

# # env.close()

# log_path = os.path.join("Traning", "Logs")
# env = DummyVecEnv([lambda: env])

# model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

# model.learn(total_timesteps=20000)

model_path = os.path.join("Training", "Saved Models", "PPO_model")

# model.save(model_path)

model = PPO.load(model_path, env)

# episodes = 5

# for episode in range(episodes):
#   obs, info = env.reset()
#   done = False
#   score = 0

#   while not done:
#     env.render()
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(int(action))
#     done = terminated or truncated
#     score += reward
#   print(f'Episode: {episode} Score: {score}')

# print(evaluate_policy(model, env, n_eval_episodes=10, render=True))

# env.close()

stop_callback = StopTrainingOnRewardThreshold(reward_threshold=500, verbose=1)
eval_callback = EvalCallback(env, callback_on_new_best=stop_callback, eval_freq=10000, best_model_save_path=model_path, verbose=1)

model.learn(total_timesteps=100000, callback=eval_callback)