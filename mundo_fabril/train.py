import wandb
import gym
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from utils import WandbCallback
from environment import Env

def train():
    # Inicializar o wandb
    wandb.init(project="ambulance-dispatch")

    # Configurações do experimento
    config = wandb.config

    # Criar o ambiente
    env = DummyVecEnv([lambda: Monitor(Env())])

    # Criar o modelo
    model = DQN(
        config.policy_type,
        env,
        learning_rate=config.learning_rate,
        buffer_size=config.buffer_size,
        batch_size=config.batch_size,
        gamma=config.gamma,
        exploration_fraction=config.exploration_fraction,
        exploration_final_eps=config.exploration_final_eps,
        train_freq=config.train_freq,
        target_update_interval=config.target_update_interval,
        verbose=1
    )

    # Configurar callbacks
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./models/', name_prefix='dqn_model')
    wandb_callback = WandbCallback()

    # Treinar o modelo
    model.learn(total_timesteps=config.total_timesteps, callback=[checkpoint_callback, wandb_callback])

    # Salvar o modelo final
    model.save("dqn_ambulance_dispatch")

    # Finalizar o wandb
    wandb.finish()

if __name__ == "__main__":
    train()
