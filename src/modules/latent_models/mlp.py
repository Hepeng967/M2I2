import torch.nn as nn
import torch.nn.functional as F
import torch as th


class TransitionModel(nn.Module):

    def __init__(self, args):
        super(TransitionModel, self).__init__()
        self.args = args

        if args.state_encoder in ["ob_ind_ae", "ob_attn_ae", "ob_attn_skipsum_ae", "ob_attn_skipcat_ae"]:
            state_repre_dim = args.state_repre_dim * args.n_agents
        else:
            state_repre_dim = args.state_repre_dim
        # Define network architecture relating to transition model
        self.action_embed = nn.Linear(args.n_actions, args.action_embed_dim)
        self.joint_action_embed = nn.Sequential(
            nn.Linear(args.action_embed_dim * args.n_agents, args.model_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.model_hidden_dim, args.model_hidden_dim),
        )
        self.network = nn.Sequential(
            nn.Linear(args.model_hidden_dim * 2, args.model_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.model_hidden_dim, state_repre_dim),
        )

        # Define network for reward prediction
        self.reward_predictor = nn.Sequential(
            nn.Linear(state_repre_dim, args.model_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.model_hidden_dim, 1),
        )
        self.state_repre_embed = nn.Sequential(
            nn.Linear(state_repre_dim, args.model_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.model_hidden_dim, args.model_hidden_dim),
        )
        
        #加入predict_action需要的网络定义
        self.inverse_state_embed = nn.Sequential(
            nn.Linear(state_repre_dim, args.model_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.model_hidden_dim, args.model_hidden_dim),
        )
        
        self.next_state_embed = nn.Sequential(
            nn.Linear(state_repre_dim, args.model_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.model_hidden_dim, args.model_hidden_dim),
        )
        
        self.joint_state_embed = nn.Sequential(
            nn.Linear(args.model_hidden_dim * 2, args.model_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.model_hidden_dim, args.action_embed_dim * args.n_agents),
        )
         # Define network for action prediction
        self.action_predictor = nn.Linear(args.action_embed_dim * args.n_agents, args.n_agents * args.n_actions)
        
    def init_hidden(self):
        # make hidden states on same device as model
        pass

    def forward(self, state_repre, actions):
        origin_shape = state_repre.shape
        if self.args.state_encoder == "ob_ind_ae":
            state_repre = state_repre.flatten(-2, -1)
            
        # actions.shape: [batch_size, seq_len, n_agents, n_actions]
        batch_size, seq_len, n_agents, _ = actions.shape
        # Comptute action embedding
        action_embed = self.action_embed(actions).reshape(batch_size, seq_len, n_agents * self.args.action_embed_dim)   # [bs, seq_len, n_agents * action_embed_dim]
        joint_action_embed = self.joint_action_embed(F.relu(action_embed)) # [bs, seq_len, model_hidden_dim]
        # Compute state repre embedding
        z_embed = self.state_repre_embed(state_repre)   # [bs, seq_len, model_hidden_dim]
        # Do forward prediction
        net_inputs = th.cat([z_embed, joint_action_embed], dim=-1)
        next_state = self.network(net_inputs)   # [bs, seq_len, state_repre_dim]
        if self.args.use_residual:
            next_state = next_state + state_repre
        return next_state.reshape(*origin_shape)
    
    def predict_action(self, state_repre, next_state):
        origin_shape = state_repre.shape
        if self.args.state_encoder == "ob_ind_ae":
            state_repre = state_repre.flatten(-2, -1)
            next_state = next_state.flatten(-2, -1)
        # Compute embeddings for next state and current state representations
        next_state_embed = self.next_state_embed(next_state)
        state_repre_embed = self.inverse_state_embed(state_repre)
        # Concatenate embeddings for prediction
        joint_state_embed = th.cat([next_state_embed, state_repre_embed], dim=-1)
        # Predict actions
        actions = self.action_predictor(self.joint_state_embed(F.relu(joint_state_embed))).reshape(self.args.batch_size, -1, self.args.n_agents,self.args.n_actions)
        # Reshape the output to match the input shape of actions [batch_size, seq_len, n_agents, n_actions]
        # actions = actions.reshape(*origin_shape[:-1],self.args.n_agent,self.args.n_actions)
        return actions
    
    def predict_reward(self, state_repre):
        if self.args.state_encoder == "ob_ind_ae":
            state_repre = state_repre.flatten(-2, -1)
        return self.reward_predictor(state_repre)

        