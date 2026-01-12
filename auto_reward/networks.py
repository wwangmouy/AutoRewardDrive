import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================================================
# State Encoder: Maps state to embedding
# ======================================================
class StateEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, encode_dim, activation_function=F.relu):
        super(StateEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, encode_dim)
        self.activation_function = activation_function
        self._init_weights()

    def forward(self, state):
        x = self.activation_function(self.fc1(state))
        x = self.activation_function(self.fc2(x))
        state_embedding = self.activation_function(self.fc3(x))
        return state_embedding
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)


# ======================================================
# Action Encoder: Maps action to embedding
# ======================================================
class ActionEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, encode_dim, activation_function=F.relu):
        super(ActionEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, encode_dim)
        self.activation_function = activation_function
        self._init_weights()

    def forward(self, action):
        x = self.activation_function(self.fc1(action))
        x = self.activation_function(self.fc2(x))
        action_embedding = self.activation_function(self.fc3(x))
        return action_embedding
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)


# ======================================================
# Forward Model: Predicts reward from embeddings
# ======================================================
class ForwardModel(nn.Module):
    def __init__(self, encode_dim, output_dim=1, last_activation=None):
        super(ForwardModel, self).__init__()
        self.last_fc = nn.Linear(encode_dim * 2, output_dim)
        self.last_activation = last_activation
        self._init_weights()

    def forward(self, state_embedding, action_embedding):
        x = torch.cat([state_embedding, action_embedding], dim=-1)
        reward = self.last_fc(x)
        if self.last_activation is not None:
            reward = self.last_activation(reward)
        return reward
    
    def _init_weights(self):
        # Small initialization for reward shaping
        nn.init.uniform_(self.last_fc.weight, -0.001, 0.001)
        nn.init.zeros_(self.last_fc.bias)


# ======================================================
# Reward Network: R(s, a)
# ======================================================
class RewardNetwork(nn.Module):
    """
    State-Action Reward Function R_omega(s, a).
    Architecture: StateEncoder + ActionEncoder --> ForwardModel --> scalar reward
    """
    def __init__(self, state_dim, action_dim, hidden_dim, encode_dim, 
                 activation_function=F.relu, last_activation=None):
        super(RewardNetwork, self).__init__()
        self.state_encoder = StateEncoder(state_dim, hidden_dim, encode_dim, activation_function)
        self.action_encoder = ActionEncoder(action_dim, hidden_dim, encode_dim, activation_function)
        self.forward_model = ForwardModel(encode_dim, output_dim=1, last_activation=last_activation)

    def forward(self, state, action):
        """
        Args:
            state: (batch_size, state_dim)
            action: (batch_size, action_dim)
        Returns:
            reward: (batch_size, 1)
        """
        state_embedding = self.state_encoder(state)
        action_embedding = self.action_encoder(action)
        reward = self.forward_model(state_embedding, action_embedding)
        return reward


# ======================================================
# Value Function: V(s)
# ======================================================
class ValueFunction(nn.Module):
    """
    State Value Function for current policy.
    Used to estimate baseline V(s) in meta-gradient computation.
    """
    def __init__(self, input_dim, hidden_dim=256, activation_function=F.relu):
        super(ValueFunction, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.activation_function = activation_function
        self._init_weights()

    def forward(self, state):
        """
        Args:
            state: (batch_size, state_dim)
        Returns:
            value: (batch_size, 1)
        """
        x = self.activation_function(self.fc1(state))
        x = self.activation_function(self.fc2(x))
        value = self.fc3(x)
        return value
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)
