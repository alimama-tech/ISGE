import torch
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg') # 显式指定非交互式后端，防止 SSH 环境下的显示问题
import matplotlib.pyplot as plt
from typing import List
from scipy.stats import pearsonr, spearmanr

# 确保能找到项目根目录下的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bidding_train_env.baseline.dd.DFUSER import DFUSER
from bidding_train_env.baseline.dd.dataset import aigb_dataset

# ================= 配置区域 =================
Iteration = 1.103          # 数据版本/模型版本
SAVE_PATH = "saved_model_dense/DDjudger" # 模型存放路径
# model_path = os.path.join(SAVE_PATH, f"diffuser_{iteration}.pt")
STEP_LEN = 48
test_path = f"AuctionNet/data/traffic/training_data_dense_iter1/training_data_all-rlData_1680.csv"
explore_test_path = f"AuctionNet/data/traffic/training_data_dense_iter1_r3s1/training_data_all-rlData.csv"
non_explore = False # 如果是explore版本写成 False
if non_explore: postfix = ""
else: 
    postfix = "_explore" 
    test_path=explore_test_path
# ===========================================

def apply_judger(iteration=Iteration,
    test_data_path=test_path, postfix=postfix):
    iteration = iteration
    model_path = os.path.join(SAVE_PATH, f"diffuser_{iteration}.pt")
    # 1. 初始化
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1)
    np.random.seed(1)

    print(f"🚀 启动 Judger (纯评测模式)")
    print(f"Device: {device}")
    
    # 2. 加载数据集 (关键：为了保证归一化参数一致)
    # 必须使用与训练时相同的数据集路径来初始化 dataset，这样 Min/Max 统计量才对
    test_data_path = test_data_path
    print(f"Loading dataset from: {test_data_path}")
    
    args_dict = {'data_version': 'monk_data_small'}
    scaler_path = os.path.join(os.path.dirname(SAVE_PATH),   # 回到 saved_model_dense
                    f"scaler_params_iter{iteration}.pkl")
    full_dataset = aigb_dataset(STEP_LEN, test_data_path, scaler_path, **args_dict)
    print(f"Dataset loaded. Total trajectories: {len(full_dataset)}")

    # 3. 初始化模型并加载权重
    # 注意：returns_condition=False 必须与训练时配置一致
    algorithm = DFUSER(returns_condition=False, scaler_path=scaler_path)
    algorithm = algorithm.to(device)
    
    # 加载权重
    try:
        print(f"Loading Model from: {model_path}")
        algorithm.load_net(model_path, device = device)
        print("✅ 模型权重加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 4. 准备评分数据
    print("\nExtracting trajectories for scoring...")
    judge_model = algorithm
    judge_model.eval() 

    all_states = []
    all_actions = []
    all_return_penalized = []

    # 遍历数据集提取 State, Action 和 Raw Reward
    for i in range(len(full_dataset)):
        # dataset[i] 返回归一化后的 tensor: (state, action, return, mask)
        s, a, r, m = full_dataset[i]
        
        # traj_x = torch.cat([a, s], dim=-1) 
        all_states.append(s.numpy())
        all_actions.append(a.numpy())
        penalized_score = r.item()
        all_return_penalized.append(penalized_score)

    s_gen = all_states # List[np.ndarray], shape: (48, 16) # 已经严格归一化到-1,1
    a_gen = all_actions # (48,1)
    real_return = np.array(all_return_penalized, dtype=np.float32)

    # 5. 执行评分
    n_timesteps = judge_model.diffuser.n_timesteps
    t_score = int(n_timesteps * 0.75) # 在 75% 的噪声水平处进行重构
    return_cond = 0.0 # 无条件生成，占位符
    
    print(f"Scoring {len(s_gen)} trajectories at timestep t={t_score}...")

    scarcity_scores = score_trajectories_by_reconstruction(
        model=judge_model, 
        trajectories_list=s_gen, 
        timestep=t_score, 
        return_cond=return_cond, 
        device=device
    ) 

    # 6. 分析与画图
    analyze_and_plot_scarcity(
        scarcity_scores=scarcity_scores, 
        rewards=real_return, 
        iteration=f"{str(iteration)}"+postfix, 
        save_folder=SAVE_PATH
    )
    
    print("🎉 Scarcity scoring pipeline finished.")
    print("--- Applying Log Transform + Normalization for scarcity_score--")
    log_scarcity_scores = np.log(scarcity_scores + 1e-7)
    print("Applied log transform. New stats (before normalization):")
    print(f"  - Max: {np.max(log_scarcity_scores):.4f}, Min: {np.min(log_scarcity_scores):.4f}, Avg: {np.mean(log_scarcity_scores):.4f}")
    log_min_sc = np.min(log_scarcity_scores)
    log_max_sc = np.max(log_scarcity_scores)
    normalized_scarcity_scores = (log_scarcity_scores - log_min_sc) / (log_max_sc - log_min_sc)
    print(f"Normalized scarcity scores (min: {normalized_scarcity_scores.min()}, max: {normalized_scarcity_scores.max()})")
    if non_explore: 
        analyze_and_plot_scarcity(
        scarcity_scores=normalized_scarcity_scores, 
        rewards=real_return, 
        iteration=f"{str(iteration)}_normalized", 
        save_folder=SAVE_PATH
    )
    correlation(scarcity_scores, real_return)
    # correlation(normalized_scarcity_scores, real_return)
    # 高return截断
    # return_threshold = np.quantile(real_return, 0.5) # reward_shaping比例
    # print(f"Return threshold set to: {return_threshold:.4f}")
    # inclusion_mask = real_return > return_threshold
    # inclusion_indices = np.where(inclusion_mask)[0]
    # s_gen_select = [s_gen[i] for i in inclusion_indices]
    # a_gen_select = [a_gen[i] for i in inclusion_indices]
    # real_return_select = real_return[inclusion_mask]
    # scarcity_scores_select = scarcity_scores[inclusion_mask]
    # normalized_scarcity_scores_select = normalized_scarcity_scores[inclusion_mask] # 我们只保留高奖励样本的稀缺性分数
    # if non_explore: 
    #     analyze_and_plot_scarcity(
    #     scarcity_scores=scarcity_scores_select, 
    #     rewards=real_return_select, 
    #     iteration=f"{str(iteration)}_select", 
    #     save_folder=SAVE_PATH
    # )
    # correlation(normalized_scarcity_scores_select, real_return_select)
    print("Mean Scarcity:", np.mean(scarcity_scores))

    return s_gen, a_gen, normalized_scarcity_scores, real_return # List[np.ndarray],List[np.ndarray],np.ndarray,np.ndarray


# =========================================================================
#  辅助函数
# =========================================================================

def score_trajectories_by_reconstruction(model, trajectories_list: List[np.ndarray], timestep: int, return_cond: float, device: str) -> np.ndarray:
    """
    使用重构误差法为一批轨迹批量打分。
    """
    print("\n" + "="*20 + f" Starting Scarcity Scoring for t={timestep} " + "="*20)
    model.diffuser.eval()
    
    # 堆叠所有轨迹 (N, 48, 16)
    trajectories_np = np.stack(trajectories_list)
    x_0_all = torch.from_numpy(trajectories_np).float()
    num_trajectories = x_0_all.shape[0]
    print(f"Input Shape: {x_0_all.shape} (Expected: [N, 48, 16])")
    all_scores = []
    # 统计维度误差
    total_dim_error = torch.zeros(x_0_all.shape[2], device=device)
    total_samples = 0

    # 批次处理以防止显存溢出
    batch_size = 2048 

    for start_idx in range(0, num_trajectories, batch_size):
        end_idx = min(start_idx + batch_size, num_trajectories)
        x_0_batch = x_0_all[start_idx:end_idx].to(device)
        current_batch_size = x_0_batch.shape[0]
        
        with torch.no_grad():
            # 1. 准备条件 (s0 作为 cond)
            s0_cond_batch = x_0_batch[:, 0:1, :]
            returns_batch = torch.full((current_batch_size, 1), return_cond, device=device, dtype=torch.float32)

            # 2. 前向加噪 (q_sample)
            t_tensor = torch.full((current_batch_size,), timestep, device=device, dtype=torch.long)
            noise_batch = torch.randn_like(x_0_batch, device=x_0_batch.device)
            x_t_batch = model.diffuser.q_sample(x_start=x_0_batch, t=t_tensor, noise=noise_batch)
            
            # 3. 逆向重构 (reconstruct)
            # 从 t 时刻去噪回到 t=0 (或者 t-1)
            # 注意：此处假设 reconstruct_from_t 的实现是正确的
            x_0_recon_batch = model.diffuser.reconstruct_from_t(
                x_t=x_t_batch, 
                t_start=timestep - 1,
                cond=s0_cond_batch, 
                returns=returns_batch
            )
            # 4. 计算误差 (MAE)
            # 计算误差 (直接计算，无需切片)
            diff_abs = torch.abs(x_0_batch - x_0_recon_batch) # shape: [Batch, Sequence, 16]
            # 目前这一版屏蔽了pvalue和volumn等环境噪声
            weights = torch.tensor([
                0.0,   # 0: Time Left
                1.0,   # 1: Budget Left (⭐⭐⭐ 核心：花钱节奏)
                1.0,   # 2: Avg Bid All
                1.0,   # 3: Avg Bid L3 (⭐⭐ 核心：出价突变)
                0.0,   # 4: Cost All
                0.0,   # 5: pVal All (屏蔽)
                0.0,   # 6: Conv All
                0.0,   # 7: Xi All
                0.0,   # 8: Cost L3
                0.0,   # 9: pVal L3 (屏蔽)
                0.0,   # 10: Conv L3
                0.0,   # 11: Xi L3
                0.0,   # 12: pVal Agg (⭐⭐⭐ 屏蔽：最大噪声源)
                0.0,   # 13: Curr Vol (屏蔽)
                0.0,   # 14: Vol L3 (屏蔽)
                0.0    # 15: Hist Vol (屏蔽)
            ], device=device).view(1, 1, 16)
            # weights = torch.ones((1, 1, 16),device=device)
            # 对维度 dim=2 求和，得到每个 step 的分数
            weighted_error = (diff_abs * weights).sum(dim=2)
            # 对时间步 dim=1 求和 (或取平均)，得到整条轨迹的分数
            trajectory_score = weighted_error.mean(dim=1)
            all_scores.append(trajectory_score.cpu().numpy())

            # 统计
            total_dim_error += torch.sum(diff_abs, dim=[0, 1])
            total_samples += (current_batch_size * 48)
    
    final_scores = np.concatenate(all_scores)
    
    # 打印维度分析 (只有 State)
    avg_dim_error = (total_dim_error / total_samples).cpu().numpy()
    print("\n🔍 [Dimension Analysis - State Only]")
    print(f"{'Dim':<5} | {'Name':<20} | {'MAE':<10} | {'Contrib'}")
    print("-" * 50)
    feat_names = ["Time Left", "Budget Left", "Avg Bid(All)", "Avg Bid(L3)", 
    "Avg Cost(All)", "Avg pVal(All)", "Avg Conv(All)", "Avg Xi(All)",
    "Avg Cost(L3)", "Avg pVal(L3)", "Avg Conv(L3)", "Avg Xi(L3)",
    "pVal Agg", "Curr Vol", "Vol(L3)", "Hist Vol"]
    # 循环打印
    for i, err in enumerate(avg_dim_error):
        cont = (err / np.sum(avg_dim_error)) * 100
        name = feat_names[i] if i < len(feat_names) else f"Feat {i}"
        print(f"{i:<5} | {name:<20} | {err:.4f} | {cont:.1f}%")
    print("-" * 50)

    return final_scores
    
    # final_scores = np.concatenate(all_scores)
    
    # # 打印统计信息
    # print(f"Scoring complete. Generated {len(final_scores)} scores.")
    # print("="*50 + "\n")
    # print("--- Scarcity Score Statistical Analysis ---")
    # print(f"  Results for noise level t={timestep}:")
    # print(f"    - Mean:   {np.mean(final_scores):.6f}")
    # print(f"    - Std:    {np.std(final_scores):.6f}")
    # print(f"    - Max:    {np.max(final_scores):.6f}")
    # print(f"    - Min:    {np.min(final_scores):.6f}")
    
    # return final_scores

def analyze_and_plot_scarcity(scarcity_scores: np.ndarray, rewards: np.ndarray, iteration: str, save_folder: str):
    """
    分析稀缺性分数和奖励，并绘制直方图、散点图和密度图。
    (精简版：仅包含绘图逻辑)
    """
    
    print(f"\n--- Plotting Analysis for Iteration {iteration} ---")

    # 确保数据格式统一 (不进行额外的 NaN 清洗，假设模型输出正常)
    scores = np.array(scarcity_scores)
    rewards = np.array(rewards)

    # ================= 1. Scarcity Histogram (直方图) =================
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=50, edgecolor='black', alpha=0.7)
    
    # 标题包含简单统计信息
    title_str = (f"{iteration}: Scarcity Distribution\n"
                 f"Mean={scores.mean():.4f}, Std={scores.std():.4f}, Max={scores.max():.4f}, Min={scores.min():.4f}")
    plt.title(title_str)
    plt.xlabel("Scarcity Score")
    plt.ylabel("Count")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    hist_path = os.path.join(save_folder, f"scarcity_hist_{iteration}.png")
    plt.savefig(hist_path, dpi=300)
    plt.close() # 及时关闭图表释放内存
    print(f"Saved: {hist_path}")

    # ================= 2. Scatter Plot (散点图) =================
    plt.figure(figsize=(10, 8))
    plt.scatter(rewards, scores, alpha=0.3, s=10) # alpha 低一点以便观察重叠
    
    plt.title(f"{iteration}: Reward vs. Scarcity")
    plt.xlabel("Reward")
    plt.ylabel("Scarcity Score")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    scatter_path = os.path.join(save_folder, f"reward_vs_scarcity_{iteration}.png")
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    print(f"Saved: {scatter_path}")

    # ================= 3. Hexbin Plot (密度图) =================
    plt.figure(figsize=(10, 8))
    hb = plt.hexbin(rewards, scores, gridsize=50, cmap='inferno', mincnt=1)
    plt.colorbar(hb, label='Density')
    
    plt.title(f"{iteration}: Reward vs. Scarcity (Density)")
    plt.xlabel("Reward")
    plt.ylabel("Scarcity Score")
    plt.grid(True, linestyle='--', alpha=0.3)
    
    hex_path = os.path.join(save_folder, f"hb_reward_vs_scarcity_{iteration}.png")
    plt.savefig(hex_path, dpi=300)
    plt.close()
    print(f"Saved: {hex_path}")
    
    print("✅ All plots generated successfully.")

def correlation(scarcity_scores: np.ndarray, rewards: np.ndarray):
    # ================= [新增] 数据可学性诊断报告 =================
    print("\n" + "="*20 + " Data Learnability Diagnosis " + "="*20)
    
    # 1. 计算相关性系数
    # Pearson: 衡量线性相关性 (-1: 强负相关, 0: 无关, 1: 强正相关)
    p_corr, _ = pearsonr(rewards, scarcity_scores)
    # Spearman: 衡量单调相关性 (非线性关系)
    s_corr, _ = spearmanr(rewards, scarcity_scores)
    
    print(f"📊 Correlation Metrics:")
    print(f"   - Pearson Correlation (Linear):   {p_corr:.4f}")
    print(f"   - Spearman Correlation (Rank):    {s_corr:.4f}")
    
    # 2. 判定“双引导”有效性
    print(f"\n🧐 Diagnosis Result:")
    if abs(p_corr) < 0.3:
        print("   ✅ [Ideal] Weak Correlation (Blob/Triangle Shape).")
        print("      结论: Reward 和 Scarcity 维度正交(不相关)。")
        print("      意味着: 存在'高稀缺但低回报'或'高回报低稀缺'的独立分布。")
        print("      建议: 双引导 (Dual-Guidance) 效果最好，能合成出指向'高回报+高稀缺'真空区的合力。")
    elif p_corr <= -0.3:
        print("   ⚠️ [Hard] Negative Correlation (Line Shape).")
        print("      结论: 稀缺性越高，回报越低 (Scarcity ≈ Bad).")
        print("      意味着: '新奇'的策略通常都是错的。")
        print("      建议: 慎用 Scarcity 引导，或者降低 w_s 权重，否则模型会为了追求稀缺而牺牲回报。")
    else: # p_corr > 0.3
        print("   ⚠️ [Easy] Positive Correlation.")
        print("      结论: 稀缺的往往就是高回报的 (Scarcity ≈ Good).")
        print("      意味着: 数据中'好的策略'本身就很少见。")
        print("      建议: 直接模仿高分轨迹即可，不需要复杂的探索机制，Standard Diffusion 就能学好。")

    # 3. "真空区"探测 (The Void Analysis)
    # 我们希望找到：目前数据里没有，但理论上存在的区域 (High Reward + High Scarcity)
    # 阈值设定：取前 10% 的高分线 和 前 10% 的稀缺线
    top_r_threshold = np.percentile(rewards, 90)
    top_s_threshold = np.percentile(scarcity_scores, 90)
    
    # 统计落在“双高”区域的点数
    mask_high_both = (rewards > top_r_threshold) & (scarcity_scores > top_s_threshold)
    count_high_both = np.sum(mask_high_both)
    total_count = len(rewards)
    ratio = count_high_both / total_count * 100
    
    print(f"\n🌌 Exploration Gap Analysis:")
    print(f"   - High Reward Threshold (Top 10%):  {top_r_threshold:.4f}")
    print(f"   - High Scarcity Threshold (Top 10%): {top_s_threshold:.4f}")
    print(f"   - Samples in 'High-High' Zone:       {count_high_both} / {total_count} ({ratio:.2f}%)")
    
    if ratio < 1.0: # 如果双高区域样本极少 (<1%)
        print("   🚀 发现真空区 (Vacuum Detected)!")
        print("      目前数据中缺乏'既高分又独特'的样本。")
        print("      这正是 Explorer 模型需要生成的'创新策略'目标区域！")
    else:
        print("   💾 区域已覆盖 (Zone Covered).")
        print("      数据中已经包含了很多高分且独特的样本。模型主要是做'回忆'而非'创造'。")
        
    print("="*60 + "\n")

if __name__ == '__main__':
    s_gen, a_gen, scarcity_scores, real_return = apply_judger()