from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os
import gymnasium as gym

enviourment_name = 'Acrobot-v1'
env = gym.make(enviourment_name, render_mode = 'rgb_array')

print("Before Training")
episodes = 5
for episode in range (1, episodes + 1):
    done = False
    state = env.reset()
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        score += reward
    print(f"Episode: {episode}, Score: {score}")

# env.close()

env = DummyVecEnv([lambda: env])

# log_path = os.path.join("Traning", "Logs", "Acrobat_Logs")
# model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

model_path = os.path.join("Traning", "Models", "Acrobat_PPO", "best_model")
# stop_callback = StopTrainingOnRewardThreshold(reward_threshold=-100, verbose=1)
# eval_callback = EvalCallback(env, callback_on_new_best=stop_callback, eval_freq=5000, best_model_save_path=model_path, verbose=1)

# model.learn(total_timesteps=300000, callback=eval_callback)

model = PPO.load(model_path, env)
# print(evaluate_policy(model, env, n_eval_episodes=10, deterministic=False))

print("After Traning")
episodes = 5
for episode in range (1, episodes + 1):
    done = False
    obs = env.reset()
    score = 0

    while not done:
        env.render()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = env.step(action)
        done = dones[0]
        score += reward[0]
    print(f"Episode: {episode}, Score: {score}")

env.close()