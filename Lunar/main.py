from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
import os

env_name = 'LunarLander-v3'
env = gym.make(env_name, render_mode = "rgb_array")

episodes = 5
for episode in range(episodes):
    state = env.reset()
    done = False
    score = 0

    while not done:
        action = env.action_space.sample()
        n_steps, reward, truncated, terminated, info = env.step(action)
        done = terminated or truncated
        score += reward
    print(f"Episode: {episode}, Score: {score}")

env = DummyVecEnv([lambda: env])

log_path = os.path.join("Training", "Logs")
model_path = os.path.join("Training", "Model")

# model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

# stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)
# eval_callback = EvalCallback(env, callback_on_new_best=stop_callback, eval_freq=10000, best_model_save_path=model_path)

# model.learn(total_timesteps=100000, callback=eval_callback)

model = PPO.load(os.path.join(model_path, "best_model.zip"), env)
# print(evaluate_policy(model, env, n_eval_episodes=10))

print("After Training")
for episode in range(episodes):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, dones, infos = env.step(action)
        done = dones[0]
        score += reward[0]
    print(f"Episode: {episode}, Score: {score}")