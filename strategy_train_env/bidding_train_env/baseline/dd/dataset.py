from torch.utils.data import Dataset
import pandas as pd
import ast
import numpy as np
import torch
import os
import pickle
from typing import List, Tuple


class aigb_dataset(Dataset):
    def __init__(self, step_len, train_data_path, scaler_path, **kwargs) -> None:
        super().__init__()
        states, actions, rewards, terminals, cpa_constraints, realAllcosts = load_local_data_nips(
            train_data_path=train_data_path)
        # 线性归一化的scale
        self.return_scale = 500.0
        # # log归一
        # self.return_min = 0.0
        # self.return_max = 100.0
        # self.epsilon = 1e-5
        # print(f"【Return log归一化】设定 Max 系数为: {self.return_max}, Min 系数: {self.return_min}")
        # self.log_min = np.log(self.return_min + self.epsilon)
        # self.log_max = np.log(self.return_max + self.epsilon)
        # self.log_denom = self.log_max - self.log_min + 1e-8
        # rank归一化

        # =================【新增归一化逻辑】=================
        print(f"【归一化前】States Max: {states.max():.2f}, Min: {states.min():.2f}")
        print(f"【归一化前】Actions Max: {actions.max():.2f}, Min: {actions.min():.2f}, Mean: {actions.mean():.2f}")
        
        # 1. 计算统计量
        self.state_min = states.min(axis=0)
        self.state_max = states.max(axis=0)
        self.action_min = actions.min(axis=0)
        self.action_max = actions.max(axis=0)
        
        # 2. 防止除以 0 (如果某列数值完全一样)
        state_denom = self.state_max - self.state_min
        state_denom[state_denom < 1e-5] = 1.0
        
        action_denom = self.action_max - self.action_min
        action_denom[action_denom < 1e-5] = 1.0

        # 3. 执行 MinMax 归一化到 [0, 1]
        states_norm = (states - self.state_min) / state_denom
        actions_norm = (actions - self.action_min) / action_denom
        
        # 4. 映射到 [-1, 1] (Diffusion 最佳区间)
        self.states = 2 * states_norm - 1
        self.actions = 2 * actions_norm - 1
        
        print(f"【归一化后】States Max: {self.states.max():.2f}, Min: {self.states.min():.2f}")
        print(f"【归一化后】Actions Max: {self.actions.max():.2f}, Min: {self.actions.min():.2f}")

        # # 5. 保存统计量 (推理时必须用同样的参数还原！)
        # save_scaler_path="./saved_model"
        # if not os.path.exists(save_scaler_path):
        #     os.makedirs(save_scaler_path)
            
        scaler_data = {
            'state_min': self.state_min,
            'state_max': self.state_max,
            'action_min': self.action_min,
            'action_max': self.action_max
        }
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler_data, f)
        print(f"【系统提示】归一化参数已保存至: {scaler_path}")
        # ===================================================

        # self.states = states
        # self.actions = actions
        self.rewards = rewards
        # self.rewards = rewards / self.return_scale
        self.terminals = terminals
        self.step_len = step_len
        self.num_of_states = states.shape[1]
        self.num_of_actions = actions.shape[1]
        self.cpa_constraints = cpa_constraints
        self.realAllcosts = realAllcosts

        # 分割序列
        # 每个序列的开头
        self.candidate_pos = (self.terminals == 0).nonzero()[0]
        self.candidate_pos += 1
        self.candidate_pos = [0] + self.candidate_pos.tolist()[:-1]
        # 后面再加上序列的结尾
        self.candidate_pos = self.candidate_pos + [self.states.shape[0]]

    def __len__(self):
        return len(self.candidate_pos) - 1

    def __getitem__(self, index):
        start_idx = self.candidate_pos[index]
        end_idx = self.candidate_pos[index + 1]
        # 获取序列
        state = torch.tensor(self.states[self.candidate_pos[index]:self.candidate_pos[index + 1], :],
                             dtype=torch.float32)
        action = torch.tensor(self.actions[self.candidate_pos[index]:self.candidate_pos[index + 1], :],
                              dtype=torch.float32)
        reward = torch.tensor(self.rewards[self.candidate_pos[index]:self.candidate_pos[index + 1], :],
                              dtype=torch.float32)

        # =================【核心修改：动态计算惩罚后 Return】=================
        # 1. 计算原始总 Return (all_value)
        raw_return = reward.sum()
        
        # 2. 获取轨迹的总消耗 (取轨迹最后一步的 realAllCost)
        # 注意：realAllCost 在 CSV 里是整条轨迹每行都一样的，所以取最后一行即可
        # end_idx 指向下一条轨迹的开头，所以当前轨迹最后一步是 end_idx - 1
        traj_total_cost = self.realAllcosts[end_idx - 1][0]
        traj_constraint = self.cpa_constraints[end_idx - 1][0]
        
        # # 3. 计算带惩罚的分数
        penalized_score = calculate_penalized_score(raw_return, traj_total_cost, traj_constraint)
        # # Log 归一化
        # log_penalized_score = np.log(penalized_score + self.epsilon)
        # final_return = (log_penalized_score - self.log_min) / self.log_denom
        # # 4. 线性归一化
        final_return = penalized_score / self.return_scale
        
        # 5. 转为 Tensor 并截断
        returns = torch.tensor([final_return], dtype=torch.float32)
        returns = torch.clamp(returns, min=0.0, max=1.0)
        # ===================================================================
        
        # 当前序列的长度
        len_state = len(state)
        # 进行padding
        state = torch.nn.functional.pad(state, (0, 0, 0, self.step_len - len(state)), "constant", 0)
        action = torch.nn.functional.pad(action, (0, 0, 0, self.step_len - len(action)), "constant", 0)
        # 计算returns
        # returns = reward.sum().sigmoid() # sigmoid归一化不合适，大量return集中在1
        # returns = reward.sum()
        # returns = torch.clamp(returns, max=1.0).reshape(1)
        # 计算masks
        masks = torch.zeros(self.step_len)
        masks[:len_state] = 1
        masks = masks.bool()
        # 返回
        return state, action, returns, masks


# 加载本地数据
def load_local_data(data_version):
    states = pd.read_csv("simulation_platform/data/offline_trajectory/" + data_version + "/states.csv").values[:,
             0::]
    actions = pd.read_csv("simulation_platform/data/offline_trajectory/" + data_version + "/actions.csv").values[:,
              0::]
    rewards = pd.read_csv("simulation_platform/data/offline_trajectory/" + data_version + "/rewards.csv").values[:,
              0::]
    terminals = pd.read_csv("simulation_platform/data/offline_trajectory/" + data_version + "/terminal.csv").values[
                :,
                0::]
    return states, actions, rewards, terminals


def load_local_data_nips(train_data_path="data/traffic/training_data_rlData_folder/training_data_all-rlData.csv"):
    if isinstance(train_data_path, str):
        path_list = [train_data_path]
    else:
        path_list = train_data_path

    all_dfs = []
    for path in path_list:
        print(f"Reading: {path}")
        df = pd.read_csv(path)
        all_dfs.append(df)
    
    # 合并所有 DataFrame
    training_data = pd.concat(all_dfs, axis=0, ignore_index=True)
    # training_data = pd.read_csv(train_data_path)

    def safe_literal_eval(val):
        if pd.isna(val):
            return val  # 如果是NaN，返回NaN
        try:
            # return ast.literal_eval(val)
            return eval(val, {"np": np})
        except (ValueError, SyntaxError):
            print(ValueError)
            return val  # 如果解析出错，返回原值

    # 使用apply方法应用上述函数
    training_data["state"] = training_data["state"].apply(safe_literal_eval)
    training_data["terminal"] = training_data["timeStepIndex"] != 47
    training_data["terminal"] = training_data["terminal"].astype(int)
    states = np.array(training_data['state'].tolist())
    actions = training_data["action"].to_numpy().reshape(-1, 1)
    rewards = training_data["reward"].to_numpy().reshape(-1, 1)
    terminals = training_data["terminal"].to_numpy().reshape(-1, 1)
    # =================【新增：读取 CPA 相关列】=================
    # realAllCost: 整条轨迹的总消耗 (每一行都存了这个值)
    # CPAConstraint: CPA 约束
    cpa_constraints = training_data["CPAConstraint"].to_numpy().reshape(-1, 1)
    realAllcosts = training_data["realAllCost"].to_numpy().reshape(-1, 1)
    # =========================================================
    
    return states, actions, rewards, terminals, cpa_constraints, realAllcosts

def calculate_penalized_score(all_value, all_cost, cpa_constraint):
    """
    计算带惩罚的分数 (复刻 PlayerAnalysis 逻辑)
    all_value: 轨迹总 Reward (转化数)
    all_cost: 轨迹总消耗
    cpa_constraint: CPA 约束
    """
    # 计算实际 CPA
    real_cpa = all_cost / (all_value + 1e-10)
    
    # 计算惩罚系数
    beta = 2
    # beta = 0
    penalty = 1.0
    if real_cpa > cpa_constraint:
        coef = cpa_constraint / (real_cpa + 1e-10)
        penalty = pow(coef, beta)
        
    return penalty * all_value

# 补一个dual_Condition的dataset
class DualConditionDataset(Dataset):
    """
    一个为 explorer 设计的数据集，用于处理变长轨迹，
    并同时打包 "Reward" (来自距离) 和 "Scarcity" (来自M1重构) 
    作为两个独立的条件。
    """
    def __init__(
        self,
        trajectories_list: List[np.ndarray],    # List[ (T_i, D) ]
        action_list: List[np.ndarray],          # List[ (T_i, 1) ]
        reward_scores: np.ndarray,              # (N,) 奖励分数, 归一化后
        scarcity_scores: np.ndarray,            # (N,) 稀缺性分数, 归一化后
        horizon: int,
        pad_value: float = 0.0
    ):
        """
        Args:
            trajectories_list: 轨迹状态列表
            action_list: 动作列表
            reward_scores: 每条轨迹的奖励分数 (e.g., from calculate_trajectory_rewards)
            scarcity_scores: 每条轨迹的稀缺性分数 (e.g., from score_trajectories...)
            horizon: 最大轨迹长度
        """
        print(f"--- Initializing DualConditionDataset with {len(trajectories_list)} trajectories ---")
        self.trajectories_list = trajectories_list
        self.action_list = action_list
        # 将 (N,) 的分数数组转为 Tensor
        self.reward_scores = torch.from_numpy(reward_scores).float()
        self.scarcity_scores = torch.from_numpy(scarcity_scores).float()
        self.horizon = horizon
        self.action_dim = 1 # 占位
        self.pad_value = pad_value

    def __len__(self) -> int:
        return len(self.trajectories_list)

    def __getitem__(self, idx) -> Tuple[Tuple, Tuple]:
        # 1. 获取原始轨迹
        state_traj: np.ndarray = self.trajectories_list[idx]  # (T_i, D)
        action_traj: np.ndarray = self.action_list[idx]  # (T_i, 1)
        T = state_traj.shape[0]

        # 2. 将状态和占位动作转为 torch tensor
        states = torch.from_numpy(state_traj).float()  # (T, D)
        actions = torch.from_numpy(action_traj).float()     # (T, 1)

        # 3. 构造 mask
        masks = torch.zeros(self.horizon, dtype=torch.bool)  # (H,)
        masks[:T] = True  # 有效步标记为 True
        
        returns = torch.clamp(self.reward_scores[idx], max=1.0).reshape(1) 
        scarcity_score = torch.clamp(self.scarcity_scores[idx], max=1.0).reshape(1) 
        
        # 6. 打包输出 5 个元素
        # (states, actions, returns, masks, scarcity)
        # 这将对应 get_loss 中的 batch[0]...batch[4]
        sample_tuple = [states, actions, returns, masks, scarcity_score]
        
        # 兼容原 dataloader 结构
        return sample_tuple
