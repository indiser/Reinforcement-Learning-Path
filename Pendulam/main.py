from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
import os

env_name = 'Pendulum-v1'
env = gym.make(env_name, render_mode = 'human', g = 9.81)

episodes = 5

print("Before Traning")
for episode in range(episodes):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        n_steps, reward, truncated, terminated, info = env.step(action)
        done = terminated or truncated
        score += reward
    print(f"Episode: {episode}, Score: {score}")


env = DummyVecEnv([lambda: env])

log_path = os.path.join("Training", "Logs")
model_path = os.path.join("Training", "Model")

# model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

# stop_callback = StopTrainingOnRewardThreshold(reward_threshold=-100, verbose=1)
# eval_callback = EvalCallback(env, callback_on_new_best=stop_callback, eval_freq=5000, best_model_save_path=model_path, verbose=1)

# model.learn(total_timesteps=500000, callback=eval_callback)

model = A2C.load(os.path.join(model_path, "best_model.zip"), env)
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