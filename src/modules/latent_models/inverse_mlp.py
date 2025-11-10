import torch.nn as nn
import torch.nn.functional as F
import torch as th

class InverseTransitionModel(nn.Module):

    def __init__(self, args):
        super(InverseTransitionModel, self).__init__()
        self.args = args

        if args.state_encoder in ["ob_ind_ae", "ob_attn_ae", "ob_attn_skipsum_ae", "ob_attn_skipcat_ae"]:
            state_repre_dim = args.state_repre_dim * args.n_agents#args.state_repre_dim用来动态定义网络架构的维度
        else:
            state_repre_dim = args.state_repre_dim

        # Define network architecture relating to inverse transition model
        self.next_state_embed = nn.Sequential(
            nn.Linear(state_repre_dim, args.model_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.model_hidden_dim, args.model_hidden_dim),
        )
        self.state_repre_embed = nn.Sequential(
            nn.Linear(state_repre_dim, args.model_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.model_hidden_dim, args.model_hidden_dim),
        )
        self.joint_state_embed = nn.Sequential(
            nn.Linear(args.model_hidden_dim * 2, args.model_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.model_hidden_dim, args.action_embed_dim * args.n_agents),
        )

        # Define network for reward prediction
        self.reward_predictor = nn.Sequential(
            nn.Linear(state_repre_dim, args.model_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.model_hidden_dim, 1),
        )
        
        # Define network for action prediction
        self.action_predictor = nn.Linear(args.action_embed_dim * args.n_agents, args.n_actions)

    def forward(self, state_repre, next_state):
        origin_shape = state_repre.shape
        if self.args.state_encoder == "ob_ind_ae":
            state_repre = state_repre.flatten(-2, -1)
            next_state = next_state.flatten(-2, -1)
        # Compute embeddings for next state and current state representations
        next_state_embed = self.next_state_embed(next_state)
        state_repre_embed = self.state_repre_embed(state_repre)
        # Concatenate embeddings for prediction
        joint_state_embed = th.cat([next_state_embed, state_repre_embed], dim=-1)
        # Predict actions
        actions = self.action_predictor(self.joint_state_embed(F.relu(joint_state_embed)))
        # Reshape the output to match the input shape of actions [batch_size, seq_len, n_agents, n_actions]
        actions = actions.reshape(*origin_shape[:-1], self.args.n_actions)
        return actions
    
    def predict_reward(self, state_repre):
        if self.args.state_encoder == "ob_ind_ae":
            state_repre = state_repre.flatten(-2, -1)
        return self.reward_predictor(state_repre)