from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import torch as th
import time
import os
import time
import json

class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0
        self.t_env = 0
        self.episode_num = 0
        
        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}
        

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,#
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.n_agents = mac.n_agents
    def get_env_info(self):
        return self.env.get_env_info()
    
    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False, teacher_forcing=False, mask_model=None):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)
        time0 = time.time()
        while not terminated:
            
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
                #"mask_obs": [self.env.get_mask_dimension()],
            }
            self.batch.update(pre_transition_data, ts=self.t)
            mask_obs = self.mac._get_mask_obs(self.batch, t=self.t)
            self.batch.update({"mask_obs": mask_obs}, ts=self.t)
            # meta_mask_obs = self._get_meta_obs(mask_model,self.batch, t=self.t)
            # self.batch.update({"meta_mask_obs": meta_mask_obs}, ts=self.t)
            #metamask

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            
            flag = 0
            # if self.episode_num == 100:
            #     flag = 1
            # if test_mode == True:
            #     flag = 1
            if teacher_forcing:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, flag=flag, test_mode=test_mode, teacher_forcing=True, mask_model=mask_model)
            else:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, flag=flag, test_mode=test_mode, mask_model=mask_model)

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            self.batch.update(post_transition_data, ts=self.t)

            # state_repr_t = self.mac.mask_enc_forward(self.batch, t=self.t,mask_model=None)
            # self.log_state_repr_t(state_repr_t,self.t)
            self.t += 1
        time1 = time.time()
        # print("time_run_1",time1 - time0)#0.39~1.3
        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],
            #"mask_obs": [self.env.get_mask_dimension()],
        }
        self.batch.update(last_data, ts=self.t)
        last_mask_obs = self.mac._get_mask_obs(self.batch, t=self.t)
        self.batch.update({"mask_obs":last_mask_obs}, ts=self.t)
        # last_meta_mask_obs = self._get_meta_obs(mask_model,self.batch, t=self.t)
        # self.batch.update({"meta_mask_obs": last_meta_mask_obs}, ts=self.t)
        # Select actions in the last stored state
        # print("self.test_returns")
        flag = 0
        if self.episode_num == 19598:
            flag = 1
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, flag=flag,test_mode=test_mode,mask_model=mask_model)
        self.batch.update({"actions": actions}, ts=self.t)

        
        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        if teacher_forcing:
            log_prefix = "teacher_forcing_" + log_prefix
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

         
        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
        
    def log_episode_num(self,episode_num):
        self.episode_num = episode_num
    
    # def log_test_z_dim(self,):
        
    def log_state_repr_t(self,state_repr_t,t):
        # print("state_repr_t.shape",state_repr_t.shape," t:",t)
        tensor = (state_repr_t.view(-1)).tolist()
        file_path = "z_dim_2.json"
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                z_dict = json.load(file)
        else:
            z_dict = {
            "z_dim_tensor": [],
            "T": []
        }
        z_dict["z_dim_tensor"].append(tensor)
        z_dict["T"].append(t)
        with open(file_path, 'w') as file:
            json.dump(z_dict, file, indent=2)