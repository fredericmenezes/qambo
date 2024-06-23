import wandb
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from environment import Env

def evaluate():
    # Carregar o modelo treinado
    model = DQN.load("dqn_ambulance_dispatch")

    # Criar o ambiente com renderização
    env = Monitor(Env(render_mode="human"))

    obs, info = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        env.render()

if __name__ == "__main__":
    evaluate()
