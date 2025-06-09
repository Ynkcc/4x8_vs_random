import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):
    """ResNet 的基本残差块 (不含 BatchNorm)"""
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        out = F.relu(out)
        return out

class NeuralNetwork(nn.Module):
    """支持双输入分支的神经网络，适配4x8棋盘"""
    def __init__(self, conv_input_shape=(15, 8, 4), fc_input_size=17,
                 action_size=160, num_res_blocks=5, num_hidden_channels=64):
        super(NeuralNetwork, self).__init__()
        
        self.conv_input_channels = conv_input_shape[0]
        self.conv_input_height = conv_input_shape[1]
        self.conv_input_width = conv_input_shape[2]

        self.conv_in = nn.Conv2d(self.conv_input_channels, num_hidden_channels,
                                kernel_size=3, padding=1, bias=False)
        
        self.res_blocks = nn.Sequential(*[ResidualBlock(num_hidden_channels)
                                        for _ in range(num_res_blocks)])
        
        self.fc_branch = nn.Sequential(
            nn.Linear(fc_input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        self.conv_flat_size = num_hidden_channels * self.conv_input_height * self.conv_input_width
        self.combined_features_size = self.conv_flat_size + 64
                
        self.policy_head = nn.Sequential(
            nn.Linear(self.combined_features_size, 512), # 增加一层和宽度
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )
        
        # 价值头 - 加深网络结构
        self.value_head = nn.Sequential(
            nn.Linear(self.combined_features_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),  # 添加 Dropout 防止过拟合
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x_conv, x_fc):
        conv_out = F.relu(self.conv_in(x_conv))
        conv_out = self.res_blocks(conv_out)
        conv_flat = conv_out.view(conv_out.size(0), -1) 
        
        fc_out = self.fc_branch(x_fc)
        
        combined = torch.cat([conv_flat, fc_out], dim=1)
        
        policy_logits = self.policy_head(combined)
        value_state = self.value_head(combined)
        
        return policy_logits, value_state
        
    # 中文注释: 新增的辅助函数，用于分离网络参数
    def get_parameter_groups(self):
        """分离并返回骨干网络、Actor头和Critic头的参数"""
        trunk_params = list(self.conv_in.parameters()) + \
                       list(self.res_blocks.parameters()) + \
                       list(self.fc_branch.parameters())
        actor_head_params = list(self.policy_head.parameters())
        critic_head_params = list(self.value_head.parameters())
        return trunk_params, actor_head_params, critic_head_params

    def predict(self, state_np):
        if not isinstance(state_np, np.ndarray):
            state_np = np.array(state_np, dtype=np.float32)
        
        num_conv_features = self.conv_input_channels * self.conv_input_height * self.conv_input_width
        x_conv_data = state_np[:num_conv_features]
        x_fc_data = state_np[num_conv_features:]

        x_conv_reshaped = x_conv_data.reshape(self.conv_input_channels,
                                              self.conv_input_height,
                                              self.conv_input_width)
        
        device = next(self.parameters()).device
        x_conv_t = torch.FloatTensor(x_conv_reshaped).unsqueeze(0).to(device)
        x_fc_t = torch.FloatTensor(x_fc_data).unsqueeze(0).to(device)
        
        self.eval()
        with torch.no_grad():
            policy_logits, value_s = self.forward(x_conv_t, x_fc_t)
        
        q_values_np = policy_logits.cpu().numpy()[0]
        value_scalar_np = value_s.cpu().numpy()[0][0]
        
        return q_values_np, value_scalar_np