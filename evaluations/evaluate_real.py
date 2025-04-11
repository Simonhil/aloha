from typing import Callable, Optional
import functools
import os

from evaluations.wrappers.vlp_eval_wrapper import VLPEvalWrapper
from utils.keyboard import KeyManager
import wandb
from typing import Callable
from hydra import compose, initialize
from flower_vla.agents.utils.diffuser_ema import EMAModel

from flower_vla.agents.lang_encoders.florence_tokens import TokenVLM
from flower_vla.dataset.oxe.transforms import generate_policy_prompt, get_action_space_index
from flower_vla.agents.utils.action_index import ActionIndex
from flower_vla.dataset.utils.frequency_mapping import DATASET_FREQUENCY_MAP

import imageio
import numpy as np
import torch
import tensorflow as tf
#import gymnasium as gym


# from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.utils import set_random_seed


import json

import itertools
import tqdm
from tqdm import trange

import einops
from copy import deepcopy


#for real aloha
from aloha_scripts import real_env
from data_collection.config import BaseConfig as bc
from data_collection import teleop_helper












# def make_env(env_id: str, rank: int, seed: int = 0, max_episode_steps: int = 400) -> Callable:
#     """
#     Utility function for multiprocessed env.

#     :param env_id: the environment ID
#     :param num_env: the number of environments you wish to have in subprocesses
#     :param seed: the initial seed for RNG
#     :param rank: index of the subprocess
#     """
#     def _init():
#         env = gym.make(env_id,
#                        obs_type="pixels_agent_pos",
#                        render_mode="rgb_array",
#                        max_episode_steps=max_episode_steps)
#         env.reset(seed=seed + rank)
#         print(f"environment with seed: {seed + rank} and max_episode_steps: {max_episode_steps} created")
#         return env
#     set_random_seed(seed)
#     return _init



    
def rollout(
    env ,
    policy: VLPEvalWrapper,
    task_description: Optional[str] = None,
    max_episode_steps: int = 400,
) -> dict:
    """Run a batched policy rollout once through a batch of environments.

    Note that all environments in the batch are run until the last environment is done. This means some
    data will probably need to be discarded (for environments that aren't the first one to be done).

    The return dictionary contains:
        (optional) "observation": A a dictionary of (batch, sequence + 1, *) tensors mapped to observation
            keys. NOTE the that this has an extra sequence element relative to the other keys in the
            dictionary. This is because an extra observation is included for after the environment is
            terminated or truncated.
        "action": A (batch, sequence, action_dim) tensor of actions applied based on the observations (not
            including the last observations).
        "reward": A (batch, sequence) tensor of rewards received for applying the actions.
        "success": A (batch, sequence) tensor of success conditions (the only time this can be True is upon
            environment termination/truncation).
        "done": A (batch, sequence) tensor of **cumulative** done conditions. For any given batch element,
            the first True is followed by True's all the way till the end. This can be used for masking
            extraneous elements from the sequences above.

    Args:
        env: The batch of environments.
        policy: The policy. Must be a PyTorch nn module.
        seeds: The environments are seeded once at the start of the rollout. If provided, this argument
            specifies the seeds for each of the environments.
        return_observations: Whether to include all observations in the returned rollout data. Observations
            are returned optionally because they typically take more memory to cache. Defaults to False.

    Returns:
        The dictionary described above.
    """
    # Reset the policy and environments.
    policy.reset(task_description)

    image_recorder = env.image_recorder

    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []

    step = 0
    done = False
    
    #generated progress bar
    progbar = trange(
        max_episode_steps,
        desc=f"Running rollout with at most {max_episode_steps} steps",
        disable=True,
        leave=False,
    )
    observation = teleop_helper.get_observations(env,image_recorder)
    km = KeyManager()
    while not done and step < max_episode_steps:
        # Numpy array to tensor and changing dictionary keys to LeRobot policy format.



        #TODO WHY? Do i need this?
        #observation["pixels"] = np.stack([obs['top'] for obs in observation['pixels']])


        with torch.inference_mode():
            action = policy.step(observation, task_description)

        assert action.ndim == 2, "Action dimensions should be (batch, action_dim)"

        # Apply the next action.
        observation, reward, new_done= teleop_helper.step(action, env, image_recorder)


        # VectorEnv stores is_success in `info["final_info"][env_index]["is_success"]`. "final_info" isn't
        # available of none of the envs finished.
        # successes = []
        # for info_dict in info:
        #     truncated.append(info_dict["TimeLimit.truncated"])
        #     successes.append(info_dict["is_success"])
        # successes = np.array(successes)
        # truncated = np.array(truncated)

        # Keep track of which environments are done so far.

        is_success = False

        if km.key == "s":
            is_success = True
            new_done = True

        if km.key == "a":
            return

        done = done | new_done

        # all_actions.append(torch.from_numpy([action]))
        # all_rewards.append(torch.from_numpy([reward]))
        # all_dones.append(torch.tensor([new_done]))
        # all_successes.append(torch.tensor([is_success]))

        step += 1
        progbar.update()

    # Stack the sequence along the first dimension so that we have (batch, sequence, *) tensors.
    ret = {
        # "action": torch.stack(all_actions, dim=1),
        # "reward": torch.stack(all_rewards, dim=1),
        # "success": torch.stack(all_successes, dim=1),
        # "done": torch.stack(all_dones, dim=1),
    }

    return ret


def main():

    os.environ['MUJOCO_GL'] = 'egl'

    with initialize(config_path="../config/vlp_eval"):
        cfg = compose(config_name="vlp_aloha")

    if cfg.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)

    n_parallel_envs = cfg.evaluation.num_parallel_envs
    max_episode_steps = cfg.evaluation.max_episode_steps
    replan_after_nsteps_list = cfg.evaluation.replan_after_nsteps
    ensemble_strategy_list = cfg.evaluation.ensemble_strategy

    combinations = list(itertools.product(replan_after_nsteps_list, ensemble_strategy_list))
    
    combinations =  [{"task_id": idx + 1,
          "replan_after_nsteps": combo[0],
          "ensemble_strategies": combo[1]}
         for idx, combo in enumerate(combinations)]
    print(combinations)

    for combi in combinations:


        #TODO generalise
        # if cfg.evaluation.task == "transfer":
        #     env_id = "gym_aloha/AlohaTransferCube-v0"
        task_description = "cube transfer right to left "
        # elif cfg.evaluation.task == "insertion":
        #     env_id = "gym_aloha/AlohaInsertion-v0"
        #     task_description = "Insert the peg into the socket."
        # else:
        #     raise ValueError("Invalid task")


        seed = 0

        env = real_env.make_real_env(init_node=False) #= SubprocVecEnv([make_env(env_id, i, 0, max_episode_steps) for i in range(n_parallel_envs)])




        print("==================================================================================")
        print(f'Evaluation for replan_after_nsteps: {combi["replan_after_nsteps"]}, ensemble_strategy: {combi["ensemble_strategies"]}')
        vlp_agent = VLPEvalWrapper(
                     saved_model_base_dir = cfg.path.saved_model_base_dir,
                     saved_model_path = cfg.path.checkpoint,
                     ema_path = cfg.path.ema_file,
                     use_ema = cfg.path.use_ema,
                     act_min_max_path = cfg.path.dataset_statistics_path,
                     device = cfg.device,
                     pred_action_horizon = cfg.evaluation.pred_horizon,
                     replan_after_nsteps = combi["replan_after_nsteps"],
                     ensemble_strategy = combi["ensemble_strategies"],
                     adaptive_ensemble_alpha= 0.1,
                     exp_decay= 0.0,
                     num_parallel_envs = n_parallel_envs)


        n_episodes = cfg.evaluation.num_episodes

        use_wandb = cfg.use_wandb

        logging_dict = {**cfg.evaluation, **cfg.path}

        if use_wandb:
            wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity,
                       group=cfg.wandb.group, name=cfg.wandb.name + f"Denoise{combi['sampling_steps']}_Replan{combi['replan_after_nsteps']}_ENtype{combi['ensemble_strategies']}",
                       config=logging_dict)

        # sum_rewards = []
        # max_rewards = []

        all_successes = 0

        #n_batches = n_episodes // n_parallel_envs + int((n_episodes % n_parallel_envs) != 0)
        

        for episode in tqdm.tqdm(range(n_episodes)):
            env.reset()
            rollout_data = rollout(env, vlp_agent, task_description, max_episode_steps)


            # if rollout_data[ "success"][-1] :
                # all_successes += 1



            #Multisim stuff


            # Figure out where in each rollout sequence the first done condition was encountered (results after
            # this won't be included).
            #n_steps = rollout_data["done"].shape[1]
            # Note: this relies on a property of argmax: that it returns the first occurrence as a tiebreaker.
            #done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)

            # Make a mask with shape (batch, n_steps) to mask out rollout data after the first done
            # (batch-element-wise). Note the `done_indices + 1` to make sure to keep the data from the done step.
            # mask = (torch.arange(n_steps) <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)).int()
            # # Extend metrics.
            # batch_sum_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "sum")
            # sum_rewards.extend(batch_sum_rewards.tolist())
            # batch_max_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "max")
            # max_rewards.extend(batch_max_rewards.tolist())
            # batch_successes = einops.reduce((rollout_data["success"] * mask), "b n -> b", "any")
            # all_successes.extend(batch_successes.tolist())
            #batch_success_rate = batch_successes.numpy().mean()
            #running_mean_success = np.nanmean(all_successes[:(batch_episode+1) * n_parallel_envs])

            # print("batch success rate:", batch_success_rate)
            # print("running success rate:", running_mean_success)

            # if use_wandb:
            #     wandb.log({"batch_success": batch_success_rate})
            #     wandb.log({"running_success": running_mean_success})

        # info = {
        #     "per_episode": [
        #         {
        #             "episode_ix": i,
        #             "sum_reward": sum_reward,
        #             "max_reward": max_reward,
        #             "success": success,
        #         }
        #         for i, (sum_reward, max_reward, success) in enumerate(
        #             zip(
        #                 sum_rewards[:n_episodes],
        #                 max_rewards[:n_episodes],
        #                 all_successes[:n_episodes],
        #                 strict=True,
        #             )
        #         )
        #     ],
        #     "aggregated": {
        #         "avg_sum_reward": float(np.nanmean(sum_rewards[:n_episodes])),
        #         "avg_max_reward": float(np.nanmean(max_rewards[:n_episodes])),
        #         "pc_success": float(np.nanmean(all_successes[:n_episodes])),
        #     },
        # }

        # print("final success rate:", info['aggregated']['pc_success'])
        # print("==================================================================================")

        # if use_wandb:
        #     wandb.log({"final_success_rate": info['aggregated']['pc_success']})
        #     wandb.finish()

        print(f"\n\n\n\n\n {all_successes} of {n_episodes} succeded \n\n\n\n\n")

if __name__=="__main__":
    main()