import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bidding_train_env.baseline.dd.DFUSER_SC import (DFUSER_SC)
import time
from bidding_train_env.baseline.dd.dataset import aigb_dataset, DualConditionDataset
from judger_explorer.apply_judger import apply_judger
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np

torch.manual_seed(1)
np.random.seed(1)
iteration = 1.1
def run_decision_diffuser(
        save_path="saved_model/DDexplorer", # 原先是saved_model/DDtest
        train_data_path = f"/AuctionNet/data/traffic/training_data_rlData_folder_iter1_cut/training_data_all-rlData_0.6.csv",
        train_epoch=1000,
        batch_size=300):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # ================= 1. TensorBoard 初始化 =================
    # 日志保存在 save_path 下的 logs 文件夹
    log_dir = os.path.join(save_path, "logs")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard 日志将保存至: {log_dir}")
    print(f"启动命令: tensorboard --logdir={log_dir}")
    
    print("train_epoch", train_epoch)
    print("batch-size", batch_size)
    print("iteration", iteration)

    step_len = 48
    args_dict = {'data_version': 'monk_data_small'}
    # dataset = aigb_dataset(step_len, train_data_path, **args_dict)
    # dataloader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True, num_workers=2, pin_memory=True)
    # ================= 2. 数据集切分 (Train/Test Split) =================
    # TODO 在apply_judger里面实现sc_score归一化
    s_gen, a_gen, scarcity_scores, real_return = apply_judger(iteration=iteration, test_data_path=train_data_path, postfix="")
    print("训练数据return最大值: ", np.max(real_return))
    print("训练数据Scarcity最大值: ", np.max(scarcity_scores))
    full_dataset = DualConditionDataset(trajectories_list=s_gen, action_list=a_gen,
                    reward_scores=real_return, scarcity_scores=scarcity_scores, horizon=step_len)
    total_size = len(full_dataset)
    train_size = int(0.9 * total_size)
    test_size = total_size - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=generator)
    print(f"数据集切分完成: 训练集 {len(train_dataset)} 条轨迹, 测试集 {len(test_dataset)} 条轨迹")
    # 创建两个 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=int(batch_size), shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=int(batch_size), shuffle=False, num_workers=2, pin_memory=True)
    test_loader_iter = iter(test_loader)
    # 归一化参数，一个数据集对应一个, 截断数据集也先用一样的
    scaler_path = os.path.join(os.path.dirname(save_path), f"scaler_params_iter{iteration}.pkl")
    algorithm = DFUSER_SC(returns_condition=True, scarcity_condition=True, scaler_path=scaler_path)
    algorithm = algorithm.to(device)

    # 参数数量
    total_params = sum(p.numel() for p in algorithm.parameters())
    print(f"参数数量：{total_params}")

    # 3. 迭代训练
    global_step = 0 # 用于 TensorBoard 的 x 轴步数
    epi = 1
    for epoch in range(0, train_epoch):
        for batch_index, (states, actions, returns, masks, scarcity_score) in enumerate(train_loader):
            states.to(device)
            actions.to(device)
            returns.to(device)
            masks.to(device)
            scarcity_score.to(device)

            start_time = time.time()

            # 训练
            all_loss, (diffuse_loss, inv_loss) = algorithm.trainStep(states, actions, returns, masks, scarcity_score)
            all_loss = all_loss.detach().clone()
            diffuse_loss = diffuse_loss.detach().clone()
            inv_loss = inv_loss.detach().clone()
            # 记录到 TensorBoard (记录每个 Step)
            writer.add_scalar('Loss/Train_Total', all_loss.item(), global_step)
            writer.add_scalar('Loss/Train_Diffuse', diffuse_loss.item(), global_step)
            writer.add_scalar('Loss/Train_Inv', inv_loss.item(), global_step)
            global_step += 1
            end_time = time.time()
            if epi % 500 == 0: print(
                f"第{epi}个batch训练时间为: {end_time - start_time} s, all_loss: {all_loss}, diffuse_loss: {diffuse_loss}, inv_loss: {inv_loss}")
            epi += 1
            if global_step % 500 == 0:
                try:
                    t_states, t_actions, t_returns, t_masks, t_sc_score = next(test_loader_iter)
                except StopIteration:
                    test_loader_iter = iter(test_loader)
                    t_states, t_actions, t_returns, t_masks, t_sc_score = next(test_loader_iter)
                with torch.no_grad():
                    t_states, t_actions, t_returns, t_masks, t_sc_score = t_states.to(device), t_actions.to(device), t_returns.to(device), t_masks.to(device), t_sc_score.to(device)
                    t_loss, (t_diff, t_inv) = algorithm.evaluateStep(t_states, t_actions, t_returns, t_masks, t_sc_score)
                t_loss_val = t_loss.detach().clone()
                t_diff_val = t_diff.detach().clone()
                t_inv_val  = t_inv.detach().clone()

                print(f"Step {global_step} [Test Batch] >> Total: {t_loss_val:.4f} | Diffuse: {t_diff_val:.4f} | Inv: {t_inv_val:.4f}")
                writer.add_scalar('Test/Loss_Total', t_loss_val.item(), global_step)
                writer.add_scalar('Test/Loss_Diffuse', t_diff_val.item(), global_step)
                writer.add_scalar('Test/Loss_Inv', t_inv_val.item(), global_step)

    # algorithm.save_model(save_path, epi)
    algorithm.save_net(save_path, epi, iteration)
    writer.close()

if __name__ == '__main__':
    run_decision_diffuser()
