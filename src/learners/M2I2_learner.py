import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam, SGD
from components.standarize_stream import RunningMeanStd
import os
import math
import time
import json

def _concat(xs):
    return th.cat([x.view(-1) for x in xs])

class M2I2Learner:
    def __init__(self, mac, latent_model, scheme, logger, args, mask_model=None):
        self.args = args
        self.mac = mac
        self.mac_ = copy.deepcopy(mac)
        if args.use_cuda:
            self.mac_.cuda()
        self.latent_model = latent_model
        self.logger = logger

        # add mask_model and mask_optim
        if self.args.use_metamask:
            self.mask_model = mask_model
            self.mask_optim = SGD(params=mask_model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4) 
        else:
            self.mask_model = None
            self.mask_optim = None 
        self.second_order = args.second_order

        if not self.args.rl_signal:
            assert 0, "Must use rl signal in this method !!!"
            self.params = list(mac.rl_parameters())
        else:
            self.params = list(mac.parameters())
        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        if self.args.use_latent_model:
            # use_latent_model means use_spr
            self.params += list(latent_model.parameters())

        self.optimiser = Adam(params=self.params, lr=args.lr)
                
        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.training_steps = 0
        self.last_target_update_step = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,), device=device)

    def mae_train(self, batch: EpisodeBatch, t_env: int, episode_num: int, mac):
        states = batch["state"]
        terminated = batch["terminated"].float()
        ter_mask = batch["filled"].float()
        ter_mask[:, 1:] = ter_mask[:, 1:] * (1 - terminated[:, :-1])
        # go through vae
        recons, z = [], []
        mac.init_hidden(batch.batch_size)  
        for t in range(batch.max_seq_length):  
            recons_t, _, z_t = mac.vae_forward(batch, t)  
            recons.append(recons_t)
            z.append(z_t)
        # recons.shape: [batch_size, seq_len, state_repre_dim]
        recons = th.stack(recons, dim=1)  # Concat over time
        # z = th.stack(z, dim=1)  # 

        mask_recons, mask_z = [], []
        mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            mask_recons_t, _, mask_z_t = mac.mask_vae_forward(batch, t, mask_model=self.mask_model)
            mask_recons.append(mask_recons_t)
            mask_z.append(mask_z_t)
        # recons.shape: [batch_size, seq_len, state_repre_dim]
        mask_recons = th.stack(mask_recons, dim=1)  # Concat over time
        # mask_z = th.stack(mask_z, dim=1)

        bs, seq_len = states.shape[0], states.shape[1]
       
        if self.args.use_metamask == True:
            loss_dict = mac.agent.encoder.loss_function(mask_recons.reshape(bs * seq_len, -1),
                                                             states.reshape(bs * seq_len, -1))  # 
            # print("mask_recons")
        elif self.args.use_metamask == False:
            loss_dict = mac.agent.encoder.loss_function(recons.reshape(bs * seq_len, -1),
                                                             states.reshape(bs * seq_len, -1))
        vae_loss = loss_dict["loss"].reshape(bs, seq_len, 1)
        ter_mask = ter_mask.expand_as(vae_loss)
        vae_loss = (vae_loss * ter_mask).sum() / ter_mask.sum()
        return vae_loss


    def repr_train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # states.shape: [batch_size, seq_len, state_dim]
        states = batch["state"]
        # actions.shape: [batch_size, seq_len, n_agents, 1]
        actions_onehot = batch["actions_onehot"]
        actions = batch["actions"]
        rewards = batch["reward"]
        terminated = batch["terminated"].float()
        mask = batch["filled"].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        
        time0 = time.time()
        
        # go through vae
        recons, z = [], []
        self.mac.init_hidden(batch.batch_size)  # useless in current version
        for t in range(batch.max_seq_length): #
            recons_t, _, z_t = self.mac.vae_forward(batch, t) #
            recons.append(recons_t)
            z.append(z_t)
        # recons.shape: [batch_size, seq_len, state_repre_dim]
        recons = th.stack(recons, dim=1)  # Concat over time
        z = th.stack(z, dim=1) #
        
        # if self.mask_model is None:
        mask_recons, mask_z = [], []
        self.mac.init_hidden(batch.batch_size)  
        for t in range(batch.max_seq_length):
            mask_recons_t, _, mask_z_t = self.mac.mask_vae_forward(batch, t, mask_model=self.mask_model)
            mask_recons.append(mask_recons_t)
            mask_z.append(mask_z_t)
        # recons.shape: [batch_size, seq_len, state_repre_dim]
        mask_recons = th.stack(mask_recons, dim=1)  # Concat over time
        mask_z = th.stack(mask_z, dim=1)
        
        

        bs, seq_len = states.shape[0], states.shape[1]
        #loss_dict = self.mac.agent.encoder.loss_function(recons.reshape(bs*seq_len, -1), states.reshape(bs*seq_len, -1))#
        if self.args.use_mask == True:
            loss_dict = self.mac.agent.encoder.loss_function(mask_recons.reshape(bs*seq_len, -1), states.reshape(bs*seq_len, -1))#
        elif self.args.use_mask == False:
            loss_dict = self.mac.agent.encoder.loss_function(recons.reshape(bs*seq_len, -1), states.reshape(bs*seq_len, -1))
        vae_loss = loss_dict["loss"].reshape(bs, seq_len, 1)
        mask = mask.expand_as(vae_loss)
        vae_loss = (vae_loss * mask).sum() / mask.sum()#
        repr_loss = vae_loss
        # time1 = time.time()
        
        if self.args.use_inverse_model:
            predicted_act = F.softmax(self.latent_model.predict_action(z[:,:-1],z[:,1:]),dim=-1)
            predicted_act = predicted_act.reshape(*predicted_act.shape[:-2], -1)
            sample_act = actions_onehot[:,:-1].reshape(*actions_onehot[:,:-1].shape[:-2], -1)
            tot_inv_loss = self.compute_inv_loss(predicted_act, sample_act, mask[:,:-1])
            repr_loss += tot_inv_loss
        
        if self.args.with_reconstruction_loss == False:
            repr_loss -= vae_loss
        
        time1 = time.time()
        
        # if self.args.use_latent_model:
        #     # Compute target z first
        #     target_projected = []
        #     with th.no_grad():
        #         self.mac.init_hidden(batch.batch_size)#w
        #         for t in range(batch.max_seq_length):
        #             target_projected_t = self.mac.target_transform(batch, t)
        #             target_projected.append(target_projected_t)
        #     target_projected = th.stack(target_projected, dim=1)  # Concat over time, shape: [bs, seq_len, spr_dim]

        #     curr_z = z
        #     # Do final vector prediction*******************************************************
        #     predicted_f = self.mac.agent.online_projection(curr_z)   # 
        #     tot_spr_loss = self.compute_spr_loss(predicted_f, target_projected, mask)#

        #     if  self.args.use_rew_pred:
        #         predicted_rew = self.latent_model.predict_reward(curr_z)   # [bs, seq_len, 1]
        #         tot_rew_loss = self.compute_rew_loss(predicted_rew, rewards, mask)
        #     for t in range(self.args.pred_len):#
        #         original_z = curr_z
        #         # do transition model forward
        #         # curr_z_inv = self.latent_model(curr_z, actions_onehot[:, t:])
        #         curr_z = self.latent_model(curr_z, actions_onehot[:, t:])[:, :-1] #
                
        #         # Do final vector prediction
        #         predicted_f = self.mac.agent.online_projection(curr_z)  # 
        #         tot_spr_loss += self.compute_spr_loss(predicted_f, target_projected[:, t+1:], mask[:, t+1:])#

        #         if self.args.use_rew_pred:
        #             predicted_rew = self.latent_model.predict_reward(curr_z)
        #             tot_rew_loss += self.compute_rew_loss(predicted_rew, rewards[:, t+1:], mask[:, t+1:])
            
        #     if self.args.use_rew_pred:
        #         repr_loss += vae_loss + self.args.spr_coef * tot_spr_loss + self.args.rew_pred_coef * tot_rew_loss
        #     elif self.args.dont_use_latent_loss:
        #         repr_loss += vae_loss
        #     else:
        #         # repr_loss = vae_loss + self.args.spr_coef * tot_spr_loss
        #         repr_loss += vae_loss + self.args.spr_coef * tot_spr_loss
        # else:
        # repr_loss += vae_loss

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("repr_loss", repr_loss.item(), t_env)
            if self.args.with_reconstruction_loss:
                self.logger.log_stat("vae_loss", vae_loss.item(), t_env)
            # if self.args.use_latent_model:
            #     self.logger.log_stat("model_loss", tot_spr_loss.item(), t_env)
            #     if self.args.use_rew_pred:
            #         self.logger.log_stat("rew_pred_loss", tot_rew_loss.item(), t_env)
            if self.args.use_inverse_model:
                self.logger.log_stat("inverse_model_loss",tot_inv_loss.item(), t_env)
            # if self.args.env == "sc2" :
            #     mask_move_feature_ratio,mask_enemy_feature_ratio,mask_ally_feature_ratio,mask_own_feature_ratio = self.mac.get_feature_mask_ratio()
            #     self.logger.log_stat("mask_move_feature_ratio", mask_move_feature_ratio,t_env)
            #     self.logger.log_stat("mask_enemy_feature_ratio", mask_enemy_feature_ratio,t_env)
            #     self.logger.log_stat("mask_ally_feature_ratio", mask_ally_feature_ratio,t_env)
            #     self.logger.log_stat("mask_own_feature_ratio", mask_own_feature_ratio,t_env)
                
        return repr_loss
    
    def compute_rew_loss(self, pred_rew, env_rew, mask):
        # pred_rew.shape: [bs, seq_len, 1]
        # mask.shape: [bs, seq_len, 1]
        mask = mask.squeeze(-1)
        rew_loss = F.mse_loss(pred_rew, env_rew, reduction="none").sum(-1)
        masked_rew_loss = (rew_loss * mask).sum() / mask.sum()
        return masked_rew_loss

    def compute_spr_loss(self, pred_f, target_f, mask):
        # pred_f.shape: [bs, seq_len, spr_dim]
        # mask.shape: [bs, seq_len, 1]
        mask = mask.squeeze(-1)
        spr_loss = F.mse_loss(pred_f, target_f, reduction="none").sum(-1)
        mask_spr_loss = (spr_loss * mask).sum() / mask.sum()
        return mask_spr_loss
    
    def compute_inv_loss(self, pred_a, target_a, mask):
        # pred_a.shape: [bs, seq_len, n_agents*n_actions]
        # mask.shape: [bs, seq_len, 1]
        mask = mask.squeeze(-1)
        act_loss = F.mse_loss(pred_a, target_a, reduction="none").sum(-1)
        mask_action_loss = (act_loss * mask).sum() / mask.sum()
        return mask_action_loss

    def rl_train(self, batch: EpisodeBatch, t_env: int, episode_num: int, mac):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        start_time = time.time()

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        # Calculate estimated Q-Values
        mac_out = []
        # print("episode_num",episode_num)
        mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            state_repr_t = mac.mask_enc_forward(batch, t=t,mask_model=self.mask_model)
            # if flag == 1:
            #     self.log_state_repr_t(state_repr_t,t)
            # if flag == 1:
            #     self.log_state_repr_t()
            if not self.args.rl_signal:
                state_repr_t = state_repr_t.detach()
            agent_outs = mac.rl_forward(batch, state_repr_t, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            state_repr_t = self.target_mac.mask_enc_forward(batch, t=t,mask_model=self.mask_model)
            target_agent_outs = self.target_mac.rl_forward(batch, state_repr_t, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        if self.args.standardise_returns:
            target_max_qvals = target_max_qvals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals.detach()

        if self.args.standardise_returns:
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())
        
        mask = mask.expand_as(td_error)
        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        rl_loss = (masked_td_error ** 2).sum() / mask.sum()

        return rl_loss

    def meta_train(self, batch, env_t, episode_num, eta):
        """
        Compute un-rolled loss and backward its gradients
        """
        #  compute unrolled multi-task network theta_1^+ (virtual step)
        loss = self.rl_train(batch, env_t, episode_num, self.mac) + self.mae_train(batch, env_t, episode_num, self.mac) 
        self.optimiser.zero_grad()
        self.mask_optim.zero_grad()
        # calculate a trial step
        loss.backward()
        # copy the gradients
        gradients = copy.deepcopy(
            [v.grad.data if v.grad is not None else None for v in self.mac.parameters()])

        self.optimiser.zero_grad()
        self.mask_optim.zero_grad()
        
        
        with th.no_grad():
            for weight, weight_, d_p in zip(self.mac.parameters(),
                                            self.mac_.parameters(),
                                            gradients):
                if d_p is None:
                    weight_.copy_(weight)
                    continue

                d_p = -d_p
                g = self.optimiser.param_groups[0]
                state = self.optimiser.state[weight]
                # if 'exp_avg' not in state:
                #     state['exp_avg'] = th.zeros_like(weight)
                # if 'exp_avg_sq' not in state:
                #     state['exp_avg_sq'] = th.zeros_like(weight)
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                step_t = state['step']
                step_t += 1

                if g['weight_decay'] != 0:
                    d_p = d_p.add(weight, alpha=g['weight_decay'])
                beta1, beta2 = g['betas']
                beta2 = g['betas'][1]
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(g['betas'][0]).add_(d_p, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(d_p, d_p.conj(), value=1 - beta2)

                step = step_t

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                step_size = g['lr'] / bias_correction1

                bias_correction2_sqrt = math.sqrt(bias_correction2)
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(g['eps'])

                weight.addcdiv_(exp_avg, denom, value=-step_size)
                weight_ = copy.deepcopy(weight)
                weight_.grad = None

        loss = self.rl_train(batch, env_t, episode_num, self.mac_) + self.mae_train(batch, env_t, episode_num, self.mac_)

        self.mask_optim.zero_grad()
        loss.backward()

        dalpha = [v.grad for v in self.mask_model.parameters()]
        if self.second_order:
            vector = [v.grad.data if v.grad is not None else None for v in self.mac_.parameters()]
            implicit_grads = self._hessian_vector_product(vector, batch, env_t, episode_num)

            for g, ig in zip(dalpha, implicit_grads):
                g.data.sub_(ig.data, alpha=eta)

        for v, g in zip(self.mask_model.parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)
        
    def _hessian_vector_product(self, gradients, batch, env_t, episode_num, r=1e-2):
        with th.no_grad():
            for weight, weight_ in zip(self.mac.parameters(), self.mac_.parameters()):
                weight_.copy_(weight)
                weight_.grad = None

        valid_grad = []
        for grad in gradients:
            if grad is not None:
                valid_grad.append(grad)
        R = r / _concat(valid_grad).norm()
        for p, v in zip(self.mac_.parameters(), gradients):
            if v is not None:
                p.data.add_(v, alpha=R)

        loss = self.rl_train(batch, env_t, episode_num, self.mac_) + self.mae_train(batch, env_t, episode_num, self.mac_)

        # print(loss)
        # print(self.mask_model.parameters())
        grads_p = th.autograd.grad(loss, self.mask_model.parameters())

        for p, v in zip(self.mac_.parameters(), gradients):
            if v is not None:
                p.data.sub_(v, alpha=2 * R)

        loss = self.rl_train(batch, env_t, episode_num, self.mac_) + self.mae_train(batch, env_t, episode_num, self.mac_)

        grads_n = th.autograd.grad(loss, self.mask_model.parameters())

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]




    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Representation learning training
        time0 = time.time()
        repr_loss = self.repr_train(batch, t_env, episode_num)
        # RL training
        rl_loss = self.rl_train(batch, t_env, episode_num, self.mac)
        
        time1 = time.time()
        tot_loss = rl_loss + self.args.repr_coef * repr_loss
        # Optimise
        # normal update
        
        time4 = time.time()
        self.optimiser.zero_grad()
        tot_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        # meta update mask
        time2 = time.time()
        # print("self.args.use_metamask",self.args.use_metamask)
        # exit()
        if self.args.use_metamask and episode_num % 10 ==0:
            print(1)
            self.meta_train(batch, t_env, episode_num, self.optimiser.param_groups[0]['lr'])
            self.mask_optim.step()
        time3 = time.time()
        
        # print
        self.training_steps += 1
        if self.args.target_update_interval_or_tau > 1 and (
                self.training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0:
            self._update_targets_hard()
            self.mac.agent.momentum_update()
            self.last_target_update_step = self.training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)
            self.mac.agent.momentum_update()

        # if t_env - self.log_stats_t >= 0:
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("rl_loss", rl_loss.item(), t_env)
            self.logger.log_stat("tot_loss", tot_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            self.log_stats_t = t_env

    def test_encoder(self, batch: EpisodeBatch):
        # states.shape: [batch_size, seq_len, state_dim]
        states = batch["state"]
        terminated = batch["terminated"].float()
        mask = batch["filled"].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        # go through vae
        recons, z = [], []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            recons_t, _, z_t = self.mac.vae_forward(batch, t)
            self.log_state_repr_t(z_t,t)#
            recons.append(recons_t)
            z.append(z_t)
        # recons.shape: [batch_size, seq_len, state_repre_dim]
        recons = th.stack(recons, dim=1)
        z = th.stack(z, dim=1)

        
        encoder_result = {
            "recons": recons,
            "z": z,
            "states": states,
            "mask": mask,
        }
        th.save(encoder_result, os.path.join(self.args.encoder_result_direc, "result.pth"))

    def _update_targets_hard(self):
        # not quite good, but don't have bad effect
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _update_targets_soft(self, tau):
        # not quite good, but don't have bad effect
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        if self.mixer is not None:
            for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.latent_model.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.latent_model.state_dict(), "{}/latent_model.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
        if self.args.use_metamask:
            th.save(self.mask_model.state_dict(), "{}/mask_model.th".format(path))
            th.save(self.mask_optim.state_dict(), "{}/mask_optim.th".format(path))
        # print(f"Mixer Model Parameters: {count_parameters(self.mixer)}")
        # print(f"mac Model Parameters: {count_parameters(self.mac)}")
        # print(f"target_mac Model Parameters: {count_parameters(self.target_mac)}")
        # print(f"latent_model Model Parameters: {count_parameters(self.latent_model)}")
        # print(f"mask_model Model Parameters: {count_parameters(self.mask_model)}")
        # print(f"mask_optim Model Parameters: {count_parameters(self.mask_optim)}")
        # print(f"optimiser Model Parameters: {count_parameters(self.optimiser)}")
        # print(f"sum Parameters: {count_parameters(self.mac) + count_parameters(self.target_mac) + count_parameters(self.mixer) + count_parameters(self.optimiser)}")



    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        self.latent_model.load_state_dict(th.load("{}/latent_model.th".format(path), map_location=lambda storage, loc: storage))
        self.mask_model.load_state_dict(th.load("{}/mask_model.th".format(path), map_location=lambda storage, loc: storage))
        self.mask_optim.load_state_dict(th.load("{}/mask_optim.th".format(path), map_location=lambda storage, loc: storage))
        
    def log_state_repr_t(self,state_repr_t,t):
        # print("state_repr_t.shape",state_repr_t.shape," t:",t)
        tensor = (state_repr_t.view(-1)).tolist()
        file_path = "z_dim_3.json"
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)