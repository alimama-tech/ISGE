import numpy as np
import torch
from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy
import numpy as np
from bidding_train_env.baseline.dd.DFUSER import DFUSER
from bidding_train_env.baseline.dd.DFUSER_SC import DFUSER_SC
import os

iteration = 0.1
class DdBiddingStrategy(BaseBiddingStrategy):
    """
    Decision-Diffuser-PlayerStrategy
    """

    def __init__(self, budget=100, name="Decision-Diffuser-PlayerStrategy", cpa=2, category=1):
        super().__init__(budget, name, cpa, category)
        file_name = os.path.dirname(os.path.realpath(__file__))
        
        dir_name = os.path.dirname(file_name)
        dir_name = os.path.dirname(dir_name)
        model_path = os.path.join(dir_name, "saved_model_dense", "DDtest", f"diffuser_{iteration}.pt")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = DFUSER(return_cond=0.35,
                    scaler_path = os.path.join(os.path.dirname(os.path.dirname(model_path)), f"scaler_params_iter{iteration}.pkl"))
        self.model.load_net(model_path, device =self.device)
        self.model.eval()
        self.state_dim = 16
        self.input = np.zeros((48,self.state_dim+1))
        self.last_planned_spend = 0.0 # 初始化


    def reset(self):
        self.remaining_budget = self.budget
        self.input = np.zeros((48, self.state_dim+1))
        self.last_planned_spend = 0.0 # 测试规划花费和实际花费是否一致
        self.planned_trajectory_at_t0 = None # 测试t0规划轨迹和实际执行轨迹的差异 (48, 16)

    # def bidding(self, timeStepIndex, pValues, pValueSigmas, historyPValueInfo, historyBid,
    #             historyAuctionResult, historyImpressionResult, historyLeastWinningCost, **kwargs):
    #     """
    #     Bids for all the opportunities in a delivery period

    #     parameters:
    #      @timeStepIndex: the index of the current decision time step.
    #      @pValues: the conversion action probability.
    #      @pValueSigmas: the prediction probability uncertainty.
    #      @historyPValueInfo: the history predicted value and uncertainty for each opportunity.
    #      @historyBid: the advertiser's history bids for each opportunity.
    #      @historyAuctionResult: the history auction results for each opportunity.
    #      @historyImpressionResult: the history impression result for each opportunity.
    #      @historyLeastWinningCosts: the history least wining costs for each opportunity.

    #     return:
    #         Return the bids for all the opportunities in the delivery period.
    #     """
    #     time_left = (48 - timeStepIndex) / 48
    #     budget_left = self.remaining_budget / self.budget if self.budget > 0 else 0
    #     history_xi = [result[:, 0] for result in historyAuctionResult]
    #     history_pValue = [result[:, 0] for result in historyPValueInfo]
    #     history_conversion = [result[:, 1] for result in historyImpressionResult]

    #     historical_xi_mean = np.mean([np.mean(xi) for xi in history_xi]) if history_xi else 0

    #     historical_conversion_mean = np.mean(
    #         [np.mean(reward) for reward in history_conversion]) if history_conversion else 0

    #     historical_LeastWinningCost_mean = np.mean(
    #         [np.mean(price) for price in historyLeastWinningCost]) if historyLeastWinningCost else 0

    #     historical_pValues_mean = np.mean([np.mean(value) for value in history_pValue]) if history_pValue else 0

    #     historical_bid_mean = np.mean([np.mean(bid) for bid in historyBid]) if historyBid else 0

    #     def mean_of_last_n_elements(history, n):
    #         last_three_data = history[max(0, n - 3):n]
    #         if len(last_three_data) == 0:
    #             return 0
    #         else:
    #             return np.mean([np.mean(data) for data in last_three_data])

    #     last_three_xi_mean = mean_of_last_n_elements(history_xi, 3)
    #     last_three_conversion_mean = mean_of_last_n_elements(history_conversion, 3)
    #     last_three_LeastWinningCost_mean = mean_of_last_n_elements(historyLeastWinningCost, 3)
    #     last_three_pValues_mean = mean_of_last_n_elements(history_pValue, 3)
    #     last_three_bid_mean = mean_of_last_n_elements(historyBid, 3)

    #     current_pValues_mean = np.mean(pValues)
    #     current_pv_num = len(pValues)

    #     historical_pv_num_total = sum(len(bids) for bids in historyBid) if historyBid else 0
    #     last_three_ticks = slice(max(0, timeStepIndex - 3), timeStepIndex)
    #     last_three_pv_num_total = sum(
    #         [len(historyBid[i]) for i in range(max(0, timeStepIndex - 3), timeStepIndex)]) if historyBid else 0

    #     test_state = np.array([
    #         time_left, budget_left, historical_bid_mean, last_three_bid_mean,
    #         historical_LeastWinningCost_mean, historical_pValues_mean, historical_conversion_mean,
    #         historical_xi_mean, last_three_LeastWinningCost_mean, last_three_pValues_mean,
    #         last_three_conversion_mean, last_three_xi_mean,
    #         current_pValues_mean, current_pv_num, last_three_pv_num_total,
    #         historical_pv_num_total
    #     ])


    #     for i in range(self.state_dim):
    #         self.input[timeStepIndex,i] = test_state[i]
    #     self.input[:,-1] = timeStepIndex
    #     x = torch.tensor(self.input.reshape(-1), device=self.device)
    #     alpha,_  = self.model(x)
    #     alpha = alpha.item()
    #     alpha = max(0,alpha)
    #     bids = alpha * pValues
    #     return bids
    def bidding(self, timeStepIndex, pValues, pValueSigmas, historyPValueInfo, historyBid,
                historyAuctionResult, historyImpressionResult, historyLeastWinningCost,
                env=None, competitor_bids=None, agent_index=None, pv_values=None, pvalue_sigmas=None, current_budgets=None, 
                cheat=True): # perfect control (True)
        """
        Bids for all the opportunities in a delivery period

        parameters:
         @timeStepIndex: the index of the current decision time step.
         @pValues: the conversion action probability.
         @pValueSigmas: the prediction probability uncertainty.
         @historyPValueInfo: the history predicted value and uncertainty for each opportunity.
         @historyBid: the advertiser's history bids for each opportunity.
         @historyAuctionResult: the history auction results for each opportunity.
         @historyImpressionResult: the history impression result for each opportunity.
         @historyLeastWinningCosts: the history least wining costs for each opportunity.

        return:
            Return the bids for all the opportunities in the delivery period.
        """
        time_left = (48 - timeStepIndex) / 48
        budget_left = self.remaining_budget / self.budget if self.budget > 0 else 0
        history_xi = [result[:, 0] for result in historyAuctionResult]
        history_pValue = [result[:, 0] for result in historyPValueInfo]
        history_conversion = [result[:, 1] for result in historyImpressionResult]

        historical_xi_mean = np.mean([np.mean(xi) for xi in history_xi]) if history_xi else 0

        historical_conversion_mean = np.mean(
            [np.mean(reward) for reward in history_conversion]) if history_conversion else 0

        historical_LeastWinningCost_mean = np.mean(
            [np.mean(price) for price in historyLeastWinningCost]) if historyLeastWinningCost else 0

        historical_pValues_mean = np.mean([np.mean(value) for value in history_pValue]) if history_pValue else 0

        historical_bid_mean = np.mean([np.mean(bid) for bid in historyBid]) if historyBid else 0

        def mean_of_last_n_elements(history, n):
            last_three_data = history[max(0, n - 3):n]
            if len(last_three_data) == 0:
                return 0
            else:
                return np.mean([np.mean(data) for data in last_three_data])

        last_three_xi_mean = mean_of_last_n_elements(history_xi, 3)
        last_three_conversion_mean = mean_of_last_n_elements(history_conversion, 3)
        last_three_LeastWinningCost_mean = mean_of_last_n_elements(historyLeastWinningCost, 3)
        last_three_pValues_mean = mean_of_last_n_elements(history_pValue, 3)
        last_three_bid_mean = mean_of_last_n_elements(historyBid, 3)

        current_pValues_mean = np.mean(pValues)
        current_pv_num = len(pValues)

        historical_pv_num_total = sum(len(bids) for bids in historyBid) if historyBid else 0
        last_three_ticks = slice(max(0, timeStepIndex - 3), timeStepIndex)
        last_three_pv_num_total = sum(
            [len(historyBid[i]) for i in range(max(0, timeStepIndex - 3), timeStepIndex)]) if historyBid else 0

        test_state = np.array([
            time_left, budget_left, historical_bid_mean, last_three_bid_mean,
            historical_LeastWinningCost_mean, historical_pValues_mean, historical_conversion_mean,
            historical_xi_mean, last_three_LeastWinningCost_mean, last_three_pValues_mean,
            last_three_conversion_mean, last_three_xi_mean,
            current_pValues_mean, current_pv_num, last_three_pv_num_total,
            historical_pv_num_total
        ])


        for i in range(self.state_dim):
            self.input[timeStepIndex,i] = test_state[i]
        self.input[:,-1] = timeStepIndex
        x = torch.tensor(self.input.reshape(-1), device=self.device)
        # =================【作弊核心逻辑】=================
        # 1. 调用模型，获取 MLP 动作 和 规划的下一步状态
        pred_alpha_tensor, planned_next_state, planned_traj_real = self.model(x) # sparse 0.0005, dense 0.001
        if timeStepIndex == 0: 
            self.planned_trajectory_at_t0 = planned_traj_real # planned_traj_real已经detach转成numpy了
            np.savetxt(f"/AuctionNet/plan_traj/planned_traj_step0.csv", planned_traj_real, delimiter=",")
        # 2. 获取 MLP 预测的 Alpha (作为对比参考)
        mlp_alpha = pred_alpha_tensor.item()
        # 3. "作弊"：通过 Budget 差分计算 Alpha
        current_budget_left = test_state[1] # 归一化前的比例，例如 0.9
        # 获取模型规划的下一步 Budget Left # planned_next_state 是反归一化后的真实物理含义
        planned_next_budget_left = planned_next_state[1]   
        # 计算模型希望这一步花多少钱 (比例)
        planned_spend_ratio = current_budget_left - planned_next_budget_left
        # 边界处理：如果模型规划涨钱(不合理)或者花钱为负，截断
        planned_spend_ratio = max(0.0, planned_spend_ratio)
        # 计算绝对花费金额
        planned_spend = planned_spend_ratio * self.budget # self.budget 是总预算绝对值
        self.last_planned_spend = planned_spend
        cheat_alpha = 100
        if cheat == False:
            cheat_alpha = max(0, mlp_alpha)
            pass
        else:
            if planned_spend == 0: 
                cheat_alpha=0
                self.last_predicted_cost = 0 
                self.last_alpha = cheat_alpha
            else:
                # # ================= [加速版作弊逻辑] =================
                # # 1. 【预处理阶段】
                # # 先用一个基准 Alpha (比如 1.0 或者上一次的 best_alpha) 探测一下环境
                # # 这一步是为了找出“在这个时刻，哪些对手注定会没钱”
                # baseline_alpha = self.last_alpha if hasattr(self, 'last_alpha') else 1.0
                # baseline_bids = (baseline_alpha * pValues).reshape(-1, 1)
                
                # # 调用环境的一个特殊方法，获取“净化后”的对手出价
                # # 我们需要 env 提供一个 helper 来返回处理过 budget 后的 competitor_bids
                # clean_competitor_bids = env.preprocess_competitors_budget(
                #     my_bids=baseline_bids,
                #     competitor_bids=competitor_bids, # 原始对手出价
                #     pv_values=pv_values,
                #     pvalue_sigmas=pvalue_sigmas,
                #     current_budgets=current_budgets)
                # 1. 定义搜索范围
                low = 0.0
                high = 500.0 
                
                # 2. 暴力搜索 100 次
                best_alpha = 0.0
                best_pred_cost = 0.0
                min_diff = float('inf')

                # [新增] 专门记录"悬崖上方"的数据
                # 寻找：大于 0 的 Cost 中，最小的那个 (即悬崖边缘)
                cliff_upper_cost = float('inf')
                cliff_upper_alpha = 0.0

                for _ in range(10): 
                    mid = (low + high) / 2
                    my_try_bids = (mid * pValues).reshape(-1, 1)
                    
                    pred_cost = env.predict_exact_cost_via_env(
                        my_bids=my_try_bids, competitor_bids=competitor_bids, env=env,
                        agent_index=agent_index, pv_values=pv_values, pvalue_sigmas=pvalue_sigmas, 
                        current_budgets=current_budgets, fast_mode=False
                    )
                    
                    # --- 核心逻辑 A: 寻找最接近 Target 的解 (主任务) ---
                    diff = abs(pred_cost - planned_spend)
                    if diff < min_diff:
                        min_diff = diff
                        best_alpha = mid
                        best_pred_cost = pred_cost
                    
                    # --- 核心逻辑 B: 寻找最小的非零值 (副任务 - 侦探) ---
                    # 如果这个 cost 是非零的，且比我们要找的"悬崖上沿"更小，就记录下来
                    if pred_cost > 1e-6:
                        if pred_cost < cliff_upper_cost:
                            cliff_upper_cost = pred_cost
                            cliff_upper_alpha = mid
                    # 二分移动
                    if pred_cost < planned_spend:
                        low = mid
                    else:
                        high = mid
                
                # 3. 最终决策
                cheat_alpha = best_alpha
                
                # 4. 悬崖报告打印
                # 触发条件：Diffuser 想花钱，但最终算法决定花 0 块
                # if planned_spend > 1.0 and best_pred_cost < 1e-6:
                #     print(f"🧗 [Cliff Detector] Step {timeStepIndex} Investigation:")
                #     print(f"   |-- 🎯 Target (Planned): {planned_spend:.4f}")
                #     print(f"   |-- 📉 Chosen (Best):    {best_pred_cost:.4f} (Alpha={best_alpha:.4f})")
                    
                #     if cliff_upper_cost < float('inf'):
                #         print(f"   |-- 🧗 Next Non-Zero:    {cliff_upper_cost:.4f} (Alpha={cliff_upper_alpha:.4f})")
                #         print(f"   |-- ⚠️ Drop Height:      {cliff_upper_cost:.4f} -> 0.0")
                #     else:
                #         print(f"   |-- 💀 Dead End:         搜索范围内全是 0，根本买不起！")
                
                # 保存数据供 run_test 使用
                self.last_planned_spend = planned_spend
                self.last_predicted_cost = best_pred_cost 
                self.last_alpha = cheat_alpha
        # # ======【对照组 Action Space Exploration】======
        # exploration_noise = np.random.normal(loc=0.0, scale=0.1)  # 可调标准差 sparse:0.1, dense:0.1
        # noisy_alpha = cheat_alpha + exploration_noise
        # noisy_alpha = max(0.0, noisy_alpha)  # 保证非负
        bids = cheat_alpha * pValues
        # 如果调整cheat_alpha(扩大或缩小10%)，对结果影响会打折扣约8%, 扩大或减小1%, 对结果打折5%
        # 1->0.0235,1.01->0.02233,0.99->,1.1->0.021,0.9->0.021
        return bids
.386