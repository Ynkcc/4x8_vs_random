# train_vs_random.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim.lr_scheduler import LambdaLR
from collections import deque
import numpy as np
import os
import random # 导入 random 模块

from torch.utils.tensorboard import SummaryWriter

# 确保 Game.py 和 model.py 在同一目录下
from Game import GameEnvironment
from model import NeuralNetwork

# --- 定义超参数 ---
# 基础参数
GAMMA = 0.99
GAE_LAMBDA = 0.95
VALUE_LOSS_COEF = 1.0 # 在合并损失时，此系数仍然重要
MAX_GRAD_NORM = 0.5
MAX_T = 1000
N_EPISODES = 20000
LOG_DIR = "runs/ppo_vs_random"
INITIAL_MODEL_PATH = 'ppo_vs_random_latest.pth'

# 批量更新参数
UPDATE_EVERY_N_EPISODES = 16

# PPO 专属超参数
PPO_EPOCHS = 10
MINIBATCH_SIZE = 256
PPO_CLIP_EPSILON = 0.2

# 动态调整的超参数
LR_INITIAL = 3e-4
LR_FINAL = 1e-5
ENTROPY_COEF_INITIAL = 0.02
ENTROPY_COEF_FINAL = 0.001
WEIGHT_DECAY = 1e-5

# 全局统计变量，用于回报的运行标准化
running_mean = 0.0
running_std = 1.0

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class A2CAgent:
    """
    Agent 类被修改为使用单一优化器和合并损失，以解决训练冲突问题。
    """
    def __init__(self, conv_input_shape, fc_input_size, action_size, total_episodes):
        self.conv_input_shape = conv_input_shape
        self.fc_input_size = fc_input_size
        self.action_size = action_size
        self.num_conv_features = np.prod(conv_input_shape)

        self.network = NeuralNetwork(conv_input_shape=conv_input_shape,
                                     fc_input_size=fc_input_size,
                                     action_size=action_size).to(DEVICE)

        # 中文注释: 创建一个统一的优化器来管理所有网络参数
        self.optimizer = optim.Adam(
            self.network.parameters(), lr=LR_INITIAL, weight_decay=WEIGHT_DECAY, eps=1e-5
        )

        # 中文注释: 相应地，也只使用一个学习率调度器
        lr_lambda = lambda episode: max(1.0 - episode / total_episodes, 0.0) * (LR_INITIAL - LR_FINAL) / LR_INITIAL + LR_FINAL / LR_INITIAL
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)


    def select_action(self, state, valid_actions_mask):
        state_conv = torch.from_numpy(state[:self.num_conv_features].reshape(self.conv_input_shape)).float().unsqueeze(0).to(DEVICE)
        state_fc = torch.from_numpy(state[self.num_conv_features:]).float().unsqueeze(0).to(DEVICE)
        
        self.network.eval()
        with torch.no_grad():
            policy_logits, value = self.network(state_conv, state_fc)

        valid_action_indices = np.where(valid_actions_mask == 1)[0]
        if len(valid_action_indices) == 0:
            raise ValueError("逻辑错误，无有效行动")
        
        mask = torch.ones(self.action_size, dtype=torch.bool).to(DEVICE)
        mask[valid_action_indices] = False
        policy_logits[0, mask] = -float('inf')

        probs = F.softmax(policy_logits, dim=-1)
        dist = Categorical(probs)
        
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, value

    def learn(self, states_conv, states_fc, actions, old_log_probs, advantages, returns, current_entropy_coef):
        self.network.train()
        
        # 中文注释: 执行一次前向传播，同时获得策略和价值
        policy_logits, values = self.network(states_conv, states_fc)
        values = values.squeeze(-1)
        
        # --- Actor Loss 计算 ---
        dist = Categorical(F.softmax(policy_logits, dim=-1))
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP_EPSILON, 1.0 + PPO_CLIP_EPSILON) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = -entropy
        
        # --- Critic Loss 计算 ---
        critic_loss = F.smooth_l1_loss(values, returns)
        
        # --- 合并总损失 ---
        total_loss = actor_loss + current_entropy_coef * entropy_loss + VALUE_LOSS_COEF * critic_loss
        
        # --- 统一进行反向传播和优化 ---
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), MAX_GRAD_NORM)
        self.optimizer.step()
        
        with torch.no_grad():
            kl_div_approx = torch.mean((torch.exp(new_log_probs - old_log_probs) - 1) - (new_log_probs - old_log_probs)).item()

        # 返回各项损失和指标用于监控
        return (total_loss.item(), actor_loss.item(), critic_loss.item(), 
                advantages.mean().item(), values.mean().item(),
                kl_div_approx, entropy.item())
    
    def scheduler_step(self):
        # 中文注释: 更新统一的调度器
        self.scheduler.step()

def compute_gae(rewards, values, gamma, lam):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t < len(rewards) - 1 else 0
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return advantages

def normalize_tensor(tensor):
    return (tensor - tensor.mean()) / (tensor.std() + 1e-8)


def train_vs_random(n_episodes=N_EPISODES, max_t=MAX_T, log_dir=LOG_DIR):
    global running_mean, running_std
    
    os.makedirs(log_dir, exist_ok=True)
    
    env = GameEnvironment()
    conv_input_shape = (15, 8, 4)
    fc_input_size = 17
    action_size = 160

    agent = A2CAgent(conv_input_shape, fc_input_size, action_size, n_episodes)
    
    try:
        agent.network.load_state_dict(torch.load(INITIAL_MODEL_PATH, map_location=DEVICE))
        print(f"成功加载初始模型: {INITIAL_MODEL_PATH}")
    except FileNotFoundError:
        print(f"未找到初始模型 {INITIAL_MODEL_PATH}，将从头开始训练。")
    
    writer = SummaryWriter(log_dir)
    scores_window = deque(maxlen=100)
    winner_window = deque(maxlen=100)
    steps_window = deque(maxlen=100)
    draw_window = deque(maxlen=100)
    explained_variance_window = deque(maxlen=100)
    kl_div_window = deque(maxlen=100)
    entropy_window = deque(maxlen=100)

    batch_states_episodes, batch_actions_episodes, batch_log_probs_episodes = [], [], []
    batch_rewards_episodes, batch_values_episodes = [], []

    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        p1_player_id = 1
        
        episode_states, episode_actions, episode_log_probs, episode_values, episode_rewards = [], [], [], [], []
        
        agent.network.eval()
        for t in range(max_t):
            valid_actions_mask = env.valid_actions()
            current_player = env.current_player
            
            if current_player == p1_player_id:
                # Agent 做出选择
                action, log_prob, value = agent.select_action(state, valid_actions_mask)
            else:
                # 对手随机选择
                valid_indices = np.where(valid_actions_mask == 1)[0]
                if len(valid_indices) == 0:
                    break 
                action = random.choice(valid_indices)
                log_prob, value = None, None
            
            next_state, reward, done, info = env.step(action)
            
            # 只记录 Agent 的经验 (当它是当前玩家时)
            if current_player == p1_player_id:
                episode_states.append(state)
                episode_actions.append(action)
                episode_log_probs.append(log_prob)
                episode_values.append(value)
                episode_rewards.append(reward)

            state = next_state
            if done:
                break
        
        winner = info.get('winner', 0)
        final_reward = 0
        if len(episode_rewards) > 0:
            if winner == p1_player_id: 
                final_reward = 1.0
            elif winner == -p1_player_id: 
                final_reward = -1.0
            else:
                final_reward = -0.1
            
            episode_rewards[-1] += final_reward

            batch_states_episodes.append(episode_states)
            batch_actions_episodes.append(episode_actions)
            batch_log_probs_episodes.append(episode_log_probs)
            batch_values_episodes.append(episode_values)
            batch_rewards_episodes.append(episode_rewards)

        scores_window.append(final_reward)
        winner_window.append(1 if winner == p1_player_id else 0)
        draw_window.append(1 if winner == 0 else 0)
        steps_window.append(t + 1)
        
        if i_episode % UPDATE_EVERY_N_EPISODES == 0 and len(batch_states_episodes) > 0:
            
            batch_advantages = []
            batch_returns = []
            
            for rewards_list, values_list in zip(batch_rewards_episodes, batch_values_episodes):
                values_array = torch.stack(values_list).squeeze().detach().cpu().numpy()
                advantages = compute_gae(rewards_list, values_array, GAMMA, GAE_LAMBDA)
                returns = [adv + val for adv, val in zip(advantages, values_array)]
                batch_advantages.extend(advantages)
                batch_returns.extend(returns)
            
            if not batch_advantages: continue
            
            flat_states = [item for sublist in batch_states_episodes for item in sublist]
            flat_actions = [item for sublist in batch_actions_episodes for item in sublist]
            flat_log_probs_list = [log_prob for ep_list in batch_log_probs_episodes for log_prob in ep_list]
            flat_log_probs = torch.stack(flat_log_probs_list)

            states_tensor = torch.from_numpy(np.vstack(flat_states)).float().to(DEVICE)
            actions_tensor = torch.tensor(flat_actions, dtype=torch.long).to(DEVICE)
            log_probs_tensor = flat_log_probs.detach()
            
            returns_tensor_raw = torch.tensor(batch_returns, device=DEVICE)
            states_conv_full = states_tensor[:, :agent.num_conv_features].reshape(-1, *agent.conv_input_shape)
            states_fc_full = states_tensor[:, agent.num_conv_features:]
            with torch.no_grad():
                _, values_pred_full = agent.network(states_conv_full, states_fc_full)
            values_pred_full = values_pred_full.squeeze(-1)
            
            running_mean = 0.99 * running_mean + 0.01 * returns_tensor_raw.mean().item()
            running_std = 0.99 * running_std + 0.01 * returns_tensor_raw.std().item()
            returns_tensor = (returns_tensor_raw - running_mean) / (running_std + 1e-8)
            returns_tensor = torch.clamp(returns_tensor, -10, 10)

            try:
                var_returns_normalized = torch.var(returns_tensor)
                variance_of_residuals = torch.var(returns_tensor - values_pred_full)
                explained_var = 1 - variance_of_residuals / (var_returns_normalized + 1e-8)
                explained_variance_window.append(explained_var.item())
            except Exception:
                explained_variance_window.append(0)

            advantages_tensor = torch.tensor(batch_advantages, device=DEVICE)
            advantages_tensor = normalize_tensor(advantages_tensor)
            advantages_tensor = torch.clamp(advantages_tensor, -10, 10)
            
            decay_progress = min(1.0, (i_episode - 1) / n_episodes)
            current_entropy_coef = ENTROPY_COEF_INITIAL - (ENTROPY_COEF_INITIAL - ENTROPY_COEF_FINAL) * decay_progress
            
            batch_size = states_tensor.shape[0]
            batch_kl_divs, batch_entropies = [], []
            
            for _ in range(PPO_EPOCHS):
                permutation = torch.randperm(batch_size)
                for start in range(0, batch_size, MINIBATCH_SIZE):
                    end = start + MINIBATCH_SIZE
                    indices = permutation[start:end]

                    mini_states, mini_actions, mini_log_probs = states_tensor[indices], actions_tensor[indices], log_probs_tensor[indices]
                    mini_advantages, mini_returns = advantages_tensor[indices], returns_tensor[indices]

                    num_conv_features = agent.num_conv_features
                    mini_states_conv = mini_states[:, :num_conv_features].reshape(-1, *agent.conv_input_shape)
                    mini_states_fc = mini_states[:, num_conv_features:]
                    
                    res = agent.learn(
                        mini_states_conv, mini_states_fc, mini_actions, mini_log_probs, 
                        mini_advantages, mini_returns.float(), current_entropy_coef
                    )
                    total_loss, actor_loss, critic_loss, avg_advantage, avg_value, kl_div, entropy = res
                    batch_kl_divs.append(kl_div)
                    batch_entropies.append(entropy)
            
            agent.scheduler_step()

            if batch_kl_divs: kl_div_window.append(np.mean(batch_kl_divs))
            if batch_entropies: entropy_window.append(np.mean(batch_entropies))
            
            writer.add_scalar('损失/总损失', total_loss, i_episode)
            writer.add_scalar('损失/Actor损失', actor_loss, i_episode)
            writer.add_scalar('损失/Critic损失', critic_loss, i_episode)
            writer.add_scalar('学习/平均优势', avg_advantage, i_episode)
            writer.add_scalar('学习/平均价值', avg_value, i_episode)
            writer.add_scalar('学习/解释方差', np.mean(explained_variance_window), i_episode)
            writer.add_scalar('学习/KL散度', np.mean(kl_div_window), i_episode)
            writer.add_scalar('学习/策略熵', np.mean(entropy_window), i_episode)
            writer.add_scalar('学习/学习率', agent.optimizer.param_groups[0]['lr'], i_episode)
            writer.add_scalar('对局/平均胜率', np.mean(winner_window), i_episode)
            writer.add_scalar('对局/平均步数', np.mean(steps_window), i_episode)


            batch_states_episodes.clear(); batch_actions_episodes.clear(); batch_log_probs_episodes.clear()
            batch_rewards_episodes.clear(); batch_values_episodes.clear()

        if i_episode % 100 == 0:
            win_rate = np.mean(winner_window) if winner_window else 0
            draw_rate = np.mean(draw_window) if draw_window else 0
            avg_steps = np.mean(steps_window) if steps_window else 0
            avg_ev = np.mean(explained_variance_window) if explained_variance_window else 0
            
            print(
                f'\n回合 {i_episode}\t胜率: {win_rate:.2f}\t平局率: {draw_rate:.2f}\t均步: {avg_steps:.1f}\t'
                f'EV: {avg_ev:.3f}\tLR: {agent.optimizer.param_groups[0]["lr"]:.7f}'
            )

            torch.save(agent.network.state_dict(), f'{log_dir}/ppo_vs_random_latest.pth')

    writer.close()
    print("\nPPO对抗随机玩家训练完成！")

if __name__ == '__main__':
    train_vs_random()