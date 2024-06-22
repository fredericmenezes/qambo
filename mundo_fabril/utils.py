import wandb
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log custom metrics from the environment
        episode_rewards = self.locals['rewards']
        infos = self.locals['infos']

        if len(infos) > 0:
            # Accumulate custom metrics
            mean_call_to_arrival = np.mean([info['call_to_arrival'] for info in infos])
            mean_assignment_to_arrival = np.mean([info['assignment_to_arrival'] for info in infos])
            demand_met = np.mean([info['fraction_demand_met'] for info in infos])
            
            # Log metrics to wandb
            wandb.log({
                'mean_call_to_arrival': mean_call_to_arrival,
                'mean_assignment_to_arrival': mean_assignment_to_arrival,
                'demand_met': demand_met
            })

        return True



def get_distance(x1, y1, x2, y2):
    """
    Pythagorean distance between two points
    """

    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    return distance
