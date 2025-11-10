from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import random
import os
import json
from envs import REGISTRY as env_REGISTRY

from modules.mask_generator import FeatureMask


# This multi-agent controller shares parameters between agents
class M2I2MAC:
    def __init__(self, scheme, groups, args, logger):
        self.n_agents = args.n_agents
        self.args = args
        self.input_shape = self._get_input_shape(scheme)
        self._build_agents(self.input_shape)
        self.agent_output_type = args.agent_output_type
        self.mask_obs = []
        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.hidden_states = None
        self.encoder_hidden_states = None
        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        if self.args.evaluate == True:
            self.evaluate_episode = 0
        if self.args.env == "sc2" :
            self.move_feats_dim = self.env.get_obs_move_feats_size()
            #print(self.move_feats_dim)
            self.enemy_feats_dim = self.env.get_obs_enemy_feats_size()
            #print(self.enemy_feats_dim)
            self.ally_feats_dim = self.env.get_obs_ally_feats_size()
            #print(self.ally_feats_dim)
            self.own_feats_dim = self.env.get_obs_own_feats_size()
            #print(self.own_feats_dim)
            self.mask_move_feature_ratio = 0
            self.mask_enemy_feature_ratio = 0
            self.mask_ally_feature_ratio = 0
            self.mask_own_feature_ratio = 0
            
            self.mask_agent_ratio = []
            
        if "map_name" in self.args.env_args and self.args.env_args['map_name'] == '1o_10b_vs_1r' and self.args.use_prior == True:
            self.meta_input_shape = (self.input_shape - self.n_agents - 14 ) * (self.n_agents-1)
        elif "map_name" in self.args.env_args and self.args.env_args['map_name'] == '1o_10b_vs_1r' and self.args.use_prior == False:
            self.meta_input_shape = (self.input_shape - self.n_agents) * self.n_agents
            
        elif "map_name" in self.args.env_args and self.args.env_args['map_name'] == '1o_2r_vs_4r' and self.args.use_prior == True:
            self.meta_input_shape = (self.input_shape - self.n_agents - 35 ) * (self.n_agents-1)
        elif "map_name" in self.args.env_args and self.args.env_args['map_name'] == '1o_2r_vs_4r' and self.args.use_prior == False:
            self.meta_input_shape = (self.input_shape - self.n_agents ) * self.n_agents
            
        elif "map_name" in self.args.env_args and self.args.env_args['map_name'] == '5z_vs_1ul' and self.args.use_prior == True:
            self.meta_input_shape = (self.input_shape - self.n_agents - 11 ) * self.n_agents
        elif "map_name" in self.args.env_args and self.args.env_args['map_name'] == '5z_vs_1ul' and self.args.use_prior == False:
            self.meta_input_shape = (self.input_shape - self.n_agents) * self.n_agents
        
        else:
            self.meta_input_shape = self.input_shape * self.n_agents
        # 156/3=52
        # print("self.meta_input_shape",self.meta_input_shape)
        

    def select_actions(self, ep_batch, t_ep, t_env, flag, bs=slice(None), test_mode=False, teacher_forcing=False,mask_model=None):
        # Only select actions for the selected batch elements in bs
        
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, flag, test_mode=test_mode,mask_model=mask_model)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, flag, test_mode=False,mask_model=None):
        agent_inputs = self._build_inputs(ep_batch, t)
        # print(agent_inputs.shape)
        agent_mask_inputs = self._build_meta_mask_inputs(ep_batch, t)
        # print(agent_mask_inputs.shape)
        # print("agent_mask_inputs",agent_mask_inputs.shape)
        if self.args.use_metamask:
            if mask_model is not None:
                feature_mask = mask_model(agent_mask_inputs)
                if flag == 1:
                    self.log_feature_mask_ratio(feature_mask,agent_inputs,ep_batch,t)
                if self.args.evaluate == True:
                    model = mask_model
                    feature_mask_1 = model.get_maskout_1()
                    if t == 0:
                        self.evaluate_episode += 1
                    self.log_feature_mask(feature_mask,feature_mask_1,t)
                    
                # self.save_tensor_to_json(t,feature_mask)
                agent_mask_inputs = agent_mask_inputs * feature_mask
        agent_mask_inputs = self._get_cat_input(agent_mask_inputs,ep_batch,t)
        # print("agent_mask_inputs",agent_mask_inputs.shape)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states, self.encoder_hidden_states = self.agent(agent_inputs, agent_mask_inputs, self.hidden_states, self.encoder_hidden_states)
        
        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def rl_forward(self, ep_batch, state_repr, t, test_mode=False):#
        # Go through downstream rl agent
        agent_inputs = self._build_inputs(ep_batch, t)
        agent_outs, self.hidden_states = self.agent.rl_forward(agent_inputs, state_repr, self.hidden_states)
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)        

    def enc_forward(self, ep_batch, t, test_mode=False):#
        agent_inputs = self._build_inputs(ep_batch, t)
        state_repr, self.encoder_hidden_states = self.agent.enc_forward(agent_inputs, self.encoder_hidden_states)
        if self.args.state_encoder in ["ob_attn_ae", "ob_attn_skipsum_ae", "ob_attn_skipcat_ae"]:
            return state_repr.view(ep_batch.batch_size, -1)
        else:
            return state_repr.view(ep_batch.batch_size, self.n_agents, -1)

    def mask_enc_forward(self, ep_batch, t, test_mode=False,mask_model=None):#
        if self.args.use_metamask:
            agent_inputs = self._build_meta_mask_inputs(ep_batch, t)
            if mask_model is not None:
                feature_mask = mask_model(agent_inputs)
                agent_inputs = agent_inputs * feature_mask
            agent_inputs = self._get_cat_input(agent_inputs,ep_batch,t)
        else:
            agent_inputs = self._build_inputs(ep_batch,t)
        state_repr, self.encoder_hidden_states = self.agent.enc_forward(agent_inputs, self.encoder_hidden_states)
        if self.args.state_encoder in ["ob_attn_ae", "ob_attn_skipsum_ae", "ob_attn_skipcat_ae"]:
            return state_repr.view(ep_batch.batch_size, -1)
        else:
            return state_repr.view(ep_batch.batch_size, self.n_agents, -1)
        

    def vae_forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        if "vae" in self.args.state_encoder:
            recons, input, mu, log_var, self.encoder_hidden_states = self.agent.vae_forward(agent_inputs, self.encoder_hidden_states)
            return recons, input, mu, log_var
        elif "ae" in self.args.state_encoder:
            recons, input, z, self.encoder_hidden_states = self.agent.vae_forward(agent_inputs, self.encoder_hidden_states)
            return recons, input, z
        else:
            raise ValueError("Unsupported state encoder type!")

    # to be che
    def mask_vae_forward(self, ep_batch, t, test_mode=False, mask_model=None):
        if self.args.use_metamask:
            agent_inputs = self._build_meta_mask_inputs(ep_batch, t)
            if mask_model is not None:
                feature_mask = mask_model(agent_inputs)
                agent_inputs = agent_inputs * feature_mask
            agent_inputs = self._get_cat_input(agent_inputs,ep_batch,t)
        else:
            agent_inputs = self._build_mask_inputs(ep_batch,t)
        if "vae" in self.args.state_encoder:
            # todo
            recons, input, mu, log_var, self.encoder_hidden_states = self.agent.vae_forward(agent_inputs, self.encoder_hidden_states)
            return recons, input, mu, log_var
        elif "ae" in self.args.state_encoder:
            recons, input, z, self.encoder_hidden_states = self.agent.vae_forward(agent_inputs, self.encoder_hidden_states)
            return recons, input, z
        else:
            raise ValueError("Unsupported state encoder type!")
    
    def target_transform(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        assert "vae" not in self.args.state_encoder, "Shouldn't use vae."
        # traget_projected.shape: [bs, spr_dim]
        target_projected, self.encoder_hidden_states = self.agent.target_transform(agent_inputs, self.encoder_hidden_states)
        return target_projected

    def init_hidden(self, batch_size, fat=False):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        # We share encoder_hidden_states between online encoder and target encoder
        if not fat:
            self.encoder_hidden_states = self.agent.encoder_init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)    # bav
        else:
            self.encoder_hidden_states = self.agent.encoder_init_hidden().unsqueeze(0).expand(batch_size*self.n_agents, self.n_agents, -1)    # bav

    def parameters(self):
        return self.agent.parameters()

    def rl_parameters(self):
        return self.agent.rl_parameters()

    def enc_parameters(self):
        return self.agent.enc_parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs
    

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape

    def _get_meta_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape * self.n_agents
               
        
    # todo
    def _build_mask_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        mask_inputs = batch["mask_obs"][:, t] # batch * n * dim
        inputs.append(mask_inputs)
        # print("mask_input_shape",len(mask_inputs),",",len(mask_inputs[0]),",",len(mask_inputs[0][0]))
        # print("mask input",mask_inputs)
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        
        return inputs
    
    def _build_meta_mask_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        metainputs = batch["obs"][:, t] # batch * n * dim
        metainputs = th.tensor(metainputs)
        if "map_name" in self.args.env_args and self.args.env_args['map_name'] == '1o_10b_vs_1r' and self.args.use_prior == True:
            mask_metainputs = metainputs[:,:-1, 11:-3]
        elif "map_name" in self.args.env_args and self.args.env_args['map_name'] == '1o_2r_vs_4r' and self.args.use_prior == True:
            mask_metainputs = metainputs[:,1:, 32:-3]
            # print("mask_metainputs_shape",mask_metainputs)
        elif "map_name" in self.args.env_args and self.args.env_args['map_name'] == '5z_vs_1ul' and self.args.use_prior == True:
            mask_metainputs = metainputs[:,:, 9:-2]
        else:
            mask_metainputs = metainputs
        # print("mask_metainputs_shape",mask_metainputs.shape)
        if "map_name" in self.args.env_args and self.args.env_args['map_name'] == '1o_10b_vs_1r' and self.args.use_prior == True:
            mask_metainputs = mask_metainputs.reshape(bs,self.meta_input_shape)
        elif "map_name" in self.args.env_args and self.args.env_args['map_name'] == '1o_2r_vs_4r' and self.args.use_prior == True:
            mask_metainputs = mask_metainputs.reshape(bs,self.meta_input_shape)
        else:
            mask_metainputs = mask_metainputs.reshape(bs,self.meta_input_shape)
            
        return mask_metainputs
    
    def _get_mask_obs_dimension(self, batch, t):
        # bs = batch.batch_size
        # obs = batch["obs"][:, t]
        # dimension = len(obs[0][0]) 
        # ratio = self.args.ratio
        # mask_num = int(ratio*dimension)
        # # 将指定行的数据全部设为0
        # for i in range(bs):
        #     for j in range(len(obs[0])):
        #         mask_indices = random.sample(range(dimension), mask_num)
        #         for idx in mask_indices:
        #             obs[i][j][idx] = 0
        
        # foster the excute speed 2025-05-15
        obs = batch["obs"][:, t]  # shape: (batch_size, n_agents, dimension)
        bs, n_agents, dimension = obs.shape
        ratio = self.args.ratio
        mask_num = int(ratio * dimension)
        mask = th.ones_like(obs)
        mask_indices = th.randint(0, dimension, (bs, n_agents, mask_num), device=obs.device)
        mask = mask.scatter(2, mask_indices, 0)
        obs = obs * mask
        return obs
        
    
    def _get_cat_input(self,meta_mask,ep_batch,t):
        bs = ep_batch.batch_size
        if "map_name" in self.args.env_args and self.args.env_args['map_name'] == '1o_10b_vs_1r' and self.args.use_prior == True:
            masktensor = meta_mask.view(bs,self.n_agents-1,-1)
        elif "map_name" in self.args.env_args and self.args.env_args['map_name'] == '1o_2r_vs_4r' and self.args.use_prior == True:
            masktensor = meta_mask.view(bs,self.n_agents-1,-1)
        else:
            masktensor = meta_mask.view(bs,self.n_agents,-1)
        
        if "map_name" in self.args.env_args and self.args.env_args['map_name'] == '1o_10b_vs_1r' and self.args.use_prior == True:
            new_meta_mask = th.cat([th.tensor(ep_batch["obs"][:,t,:-1,:11]), masktensor, th.tensor(ep_batch["obs"][:,t,:-1,-3:])], dim=2)#
            new_meta_mask_with_o = th.cat([th.tensor(ep_batch["obs"][:,t,-1,:]).unsqueeze(1),new_meta_mask],dim=1)#
        elif "map_name" in self.args.env_args and self.args.env_args['map_name'] == '1o_2r_vs_4r' and self.args.use_prior == True:
            new_meta_mask = th.cat([ ep_batch["obs"][:,t,1:,:32], masktensor, th.tensor(ep_batch["obs"][:,t,1:,-3:])], dim=2)#
            new_meta_mask_with_o = th.cat([th.tensor(ep_batch["obs"][:,t,0,:]).unsqueeze(1),new_meta_mask],dim=1)#
        elif "map_name" in self.args.env_args and self.args.env_args['map_name'] == '5z_vs_1ul' and self.args.use_prior == True:
            new_meta_mask = th.cat([ ep_batch["obs"][:,t,:,:9], masktensor, th.tensor(ep_batch["obs"][:,t,:,-2:])], dim=2)#
            new_meta_mask_with_o = new_meta_mask
        else:
            new_meta_mask_with_o = masktensor
        
        if self.args.obs_agent_id:
            identity_matrix = th.eye(self.n_agents,device=ep_batch.device)
            tensor_with_identity = th.cat([new_meta_mask_with_o, identity_matrix.unsqueeze(0).expand(bs, -1, -1)], dim=-1)
        else:
            tensor_with_identity = new_meta_mask_with_o
        tensor_with_identity = tensor_with_identity.view(bs*self.n_agents,-1)
        return tensor_with_identity
    
    def _get_meta_obs(self,mask_model,batch,t):
        bs = batch.batch_size
        metainputs = batch["obs"][:, t]
        metainputs = th.tensor(metainputs)
        if "map_name" in self.args.env_args and self.args.env_args['map_name'] == '1o_10b_vs_1r' and self.args.use_prior == True:
            mask_metainputs = metainputs[:,:, 11:-3]
        elif "map_name" in self.args.env_args and self.args.env_args['map_name'] == '1o_2r_vs_4r' and self.args.use_prior == True:
            mask_metainputs = metainputs[:,:, 32:-3]
        elif "map_name" in self.args.env_args and self.args.env_args['map_name'] == '5z_vs_1ul' and self.args.use_prior == True:
            mask_metainputs = metainputs[:,:, 9:-2]
        else:
            mask_metainputs = metainputs
            
        # print("mask_metainputs",mask_metainputs.shape)
        mask_metainputs = mask_metainputs.reshape(bs*self.n_agents,-1)
        meta_mask = mask_model(mask_metainputs)
        # print("meta_mask_before",meta_mask.shape)
        if "map_name" in self.args.env_args and self.args.env_args['map_name'] == '1o_10b_vs_1r' and self.args.use_prior == True:
            new_meta_mask = th.cat([th.ones((self.n_agents, 11)).to(meta_mask.device), meta_mask, th.ones((self.n_agents, 3)).to(meta_mask.device)], dim=1)
        elif "map_name" in self.args.env_args and self.args.env_args['map_name'] == '1o_2r_vs_4r' and self.args.use_prior == True:
            new_meta_mask = th.cat([th.ones((self.n_agents, 32)).to(meta_mask.device), meta_mask, th.ones((self.n_agents, 3)).to(meta_mask.device)], dim=1)
        elif "map_name" in self.args.env_args and self.args.env_args['map_name'] == '5z_vs_1ul' and self.args.use_prior == True:
            new_meta_mask = th.cat([th.ones((self.n_agents, 9)).to(meta_mask.device), meta_mask, th.ones((self.n_agents, 2)).to(meta_mask.device)], dim=1)
        else:
            new_meta_mask = meta_mask
        # print("meta_mask_after",new_meta_mask.shape)
        # print("metainputs",metainputs.shape)
        metainputs = new_meta_mask * metainputs
        metainputs = metainputs.view(bs,self.n_agents,-1)
        metainputs = metainputs.tolist()
        return metainputs

    #
    def _get_mask_obs(self, batch, t):
        if self.args.mask_method == "dimension":
            return self._get_mask_obs_dimension(batch,t)
        if self.args.mask_method == "agent":
            return self._get_mask_obs_agent(batch,t)
        
    def log_feature_mask_ratio(self, feature, inputs, ep_batch, t):
        bs = ep_batch.batch_size
        feature_tensor = feature.view(bs, self.n_agents, -1)
        inputs_tensor = inputs.view(bs, self.n_agents, -1)

        ratio_file_path = "ratio_3.json"


        if os.path.exists(ratio_file_path):
            with open(ratio_file_path, 'r') as file:
                ratio_dict = json.load(file)
        else:
            ratio_dict = {
            "mask_move_feature_ratio": [],
            "mask_enemy_feature_ratio": [],
            "mask_ally_feature_ratio": [],
            "mask_own_feature_ratio": [],
            "T": []
        }
        
        b = self.move_feats_dim
        c = self.move_feats_dim+(self.enemy_feats_dim[0]*self.enemy_feats_dim[1])
        d = self.move_feats_dim+(self.enemy_feats_dim[0]*self.enemy_feats_dim[1])+(self.ally_feats_dim[0]*self.ally_feats_dim[1])
        e = self.move_feats_dim+(self.enemy_feats_dim[0]*self.enemy_feats_dim[1])+(self.ally_feats_dim[0]*self.ally_feats_dim[1])+self.own_feats_dim
        move_feature = feature_tensor[:,:,:b]
        enemy_feature = feature_tensor[:,:,b:c]
        ally_feature = feature_tensor[:,:,c:d]
        own_feature = feature_tensor[:,:,d:e]
        
        move_inputs = inputs_tensor[:,:,:b]
        enemy_inputs = inputs_tensor[:,:,b:c]
        ally_inputs = inputs_tensor[:,:,c:d]
        own_inputs = inputs_tensor[:,:,d:e]

        count_move_feature_zero_all = th.count_nonzero(move_feature == 0)
        count_enemy_feature_zero_all = th.count_nonzero(enemy_feature == 0)
        count_ally_feature_zero_all = th.count_nonzero(ally_feature == 0)
        count_own_feature_zero_all = th.count_nonzero(own_feature == 0)
        

        count_move_feature_zero = th.sum((move_feature == 0) & (move_inputs != 0))
        count_enemy_feature_zero = th.sum((enemy_feature == 0) & (enemy_inputs != 0))
        count_ally_feature_zero = th.sum((ally_feature == 0) & (ally_inputs != 0))
        count_own_feature_zero = th.sum((own_feature == 0) & (own_inputs != 0))
            
        total_sum = count_move_feature_zero.item()+count_enemy_feature_zero.item()+count_ally_feature_zero.item()+count_own_feature_zero.item()

        self.mask_move_feature_ratio = 0
        self.mask_enemy_feature_ratio = 0
        self.mask_ally_feature_ratio = 0
        self.mask_own_feature_ratio = 0
        
        if count_move_feature_zero_all.item() != 0:
            self.mask_move_feature_ratio = count_move_feature_zero.item()/count_move_feature_zero_all.item()
        if count_enemy_feature_zero_all.item() != 0:
            self.mask_enemy_feature_ratio = count_enemy_feature_zero.item()/count_enemy_feature_zero_all.item()
        if count_ally_feature_zero_all.item() != 0:
            self.mask_ally_feature_ratio = count_ally_feature_zero.item()/count_ally_feature_zero_all.item()
        if count_own_feature_zero_all.item() != 0:
            self.mask_own_feature_ratio = count_own_feature_zero.item()/count_own_feature_zero_all.item()
            

        
        ratio_dict['mask_move_feature_ratio'].append(self.mask_move_feature_ratio)
        ratio_dict['mask_enemy_feature_ratio'].append(self.mask_enemy_feature_ratio)
        ratio_dict['mask_ally_feature_ratio'].append(self.mask_ally_feature_ratio)
        ratio_dict['mask_own_feature_ratio'].append(self.mask_own_feature_ratio)
        ratio_dict['T'].append(t)
        with open(ratio_file_path, 'w') as file:
            json.dump(ratio_dict, file, indent=4)

        # zero_counts_per_agent = th.sum(feature_tensor == 0, dim=(0,2))
        # total_zero_count = th.sum(zero_counts_per_agent)
        # ratio_per_agent = zero_counts_per_agent.float()/total_zero_count.float()
        # self.mask_agent_ratio = ratio_per_agent.tolist()
    
    def log_feature_mask(self,feature_mask_0,feature_mask_1,t):
        
        feature_mask_0_list = feature_mask_0.tolist()
        feature_mask_1_list = feature_mask_1.tolist()

        feature_load_file_path = "feature_mask_load_json_40%_2/feature_mask_" + str(self.evaluate_episode) + ".json"
        
        if os.path.exists(feature_load_file_path):
            with open(feature_load_file_path, 'r') as file:
                feature_dict = json.load(file)
        else:
            feature_dict = {
            "feature_mask_0": [],
            "feature_mask_1": [],
            "T": []
        }
            
        feature_dict["feature_mask_0"].append(feature_mask_0_list)
        feature_dict["feature_mask_1"].append(feature_mask_1_list)
        feature_dict['T'].append(t)
        
        with open(feature_load_file_path, 'w') as file:
            json.dump(feature_dict, file, indent=4)

        
    def get_feature_mask_ratio(self):
        return self.mask_move_feature_ratio,self.mask_enemy_feature_ratio,self.mask_ally_feature_ratio,self.mask_own_feature_ratio
    
    def get_mask_agent_ratio(self):
        return self.mask_agent_ratio
    
    def save_tensor_to_json(self,t, tensor):
        target_dir = ""
        th.save(tensor.to(th.device('cpu')),target_dir)
