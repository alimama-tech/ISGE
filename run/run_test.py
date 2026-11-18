import sys
import time
import math
import os
from typing import List, Optional

import gin
import numpy as np
import psutil

from simul_bidding_env.Tracker.BiddingTracker import BiddingTracker
from simul_bidding_env.Tracker.PlayerAnalysis import PlayerAnalysis
from simul_bidding_env.Controller.Controller import Controller
from collections import Iterable
import logging
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




def initialize_player_agent() -> Optional['PlayerAgent']:
    """Initializes the PlayerAgent with appropriate strategy.

    Returns:
        Optional[PlayerAgent]: An instance of PlayerAgent.
    """
    try:
        # raise ImportError("")
        from bidding_train_env.strategy import PlayerBiddingStrategy as PlayerAgent
        logger.info("Successfully imported PlayerBiddingStrategy as PlayerAgent.")
        return PlayerAgent()
    except ImportError as import_error:
        logger.error(f"Failed to import PlayerAgent: {import_error}")
        try:
            from simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy as PlayerAgent
            import numpy as np
            agent = PlayerAgent(exp_tempral_ratio=np.ones(48))
            agent.name += "0"
            logger.info("Successfully loaded PidBiddingStrategy as PlayerAgent in local run mode.")
            return agent
        except ImportError as e:
            logger.error(f"Failed to import PidBiddingStrategy: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during PlayerAgent initialization: {e}")
        sys.exit(1)




def get_winner(slot_pit: np.ndarray) -> np.ndarray:
    """Determines the winners for each slot in the pit.

    """
    slot_pit = slot_pit.T
    num_pv, num_agent = slot_pit.shape
    num_slot = 3
    winner = np.full((num_pv, num_slot), -1, dtype=int)

    for pos in range(1, num_slot + 1):
        winning_agents_indices = np.argwhere(slot_pit == pos)
        if winning_agents_indices.size > 0:
            pv_indices, agent_indices = winning_agents_indices.T
            winner[pv_indices, pos - 1] = agent_indices
    return winner


def adjust_over_cost(bids: np.ndarray, over_cost_ratio: np.ndarray, envs_slots: List[int],winner_pit) -> None:
    """Adjusts the bids to prevent overcost.

    """
    overcost_agent_indices = np.where(over_cost_ratio > 0)[0]
    for agent_index in overcost_agent_indices:
        for i, coefficient in enumerate(envs_slots):
            winner_indices = winner_pit[:, i]
            pv_indices = np.where(winner_indices == agent_index)[0]
            rng = np.random.default_rng(seed=1)
            num_to_drop = math.ceil(pv_indices.size * over_cost_ratio[agent_index])
            if num_to_drop > 0:
                dropped_pv_indices = rng.choice(pv_indices, num_to_drop, replace=False)
                bids[dropped_pv_indices, agent_index] = 0


def initialize_player_analysis() -> PlayerAnalysis:
    """Initializes the PlayerAnalysis instance.

    """
    return PlayerAnalysis("player_analysis")


def log_memory_usage() -> None:
    """Logs the current memory usage of the process."""
    pid = os.getpid()
    current_process = psutil.Process(pid)
    memory_info = current_process.memory_info()
    memory_mb = memory_info.rss / (1024 * 1024)
    logger.info(f"Memory usage: {memory_mb:.2f} MB")


@gin.configurable
def run_test(generate_log: bool = False,
             num_episode: int = 2,
             num_tick: int = 48,
             player_index: int = 0,
             start_epi: int = 0,     # [新增] 接收起始 episode
             save_dir: str = None
             ):
    """Runs the bidding simulation test.

    Args:
        generate_log (bool): Flag to generate logs.
        num_episode (int): Number of episodes to run.
        num_tick (int): Number of ticks per episode.
        player_index (int): Index of the player agent.

    Returns:
        ReturnType: Result of the player analysis.
    """
    # t0 = time.perf_counter()

    player_agent = initialize_player_agent()
    player_analysis = initialize_player_analysis()
    bidding_controller = Controller(player_index=player_index, player_agent=player_agent)

    agents = bidding_controller.agents
    num_agent = len(agents)
    agents_category = np.array([agent.category for agent in agents])
    agents_cpa = np.array([agent.cpa for agent in agents])

    envs = bidding_controller.biddingEnv
    pv_generator = bidding_controller.pvGenerator
    train_data_tracker = BiddingTracker("train_data_generate_tracker")
    logger.info(
        f"playerAgentName:{agents[player_index].name} playerAgentCpa:{agents[player_index].cpa} playerAgentBudget:{agents[player_index].budget} playerAgentCategory:{agents[player_index].category} PlayerIndex:{player_index}")

    total_pv_num = 0
    begin_time = time.time()
    # start_epi = 0
    # 用于存储所有 step0 规划的轨迹
    planned_data_buffer = []
    for episode in range(start_epi, start_epi + num_episode):
        logger.info(f" PlayerIndex:{player_index} episode:{episode} evaluate")
        if generate_log:
            train_data_tracker.reset()

        rewards = np.zeros(num_agent)
        costs = np.zeros(num_agent)
        budgets = np.array([agent.budget for agent in agents])

        history_pvalue_infos = []
        history_bids = []
        history_auction_results = []
        history_impression_results = []
        history_least_winning_costs = []

        bidding_controller.reset(episode=episode)


        for tick_index in range(num_tick):
            pv_values = pv_generator.pv_values[tick_index]
            pvalue_sigmas = pv_generator.pValueSigmas[tick_index]
            # ================= [修改开始] 上帝视角分步出价 =================
            # 1. 先让所有非 Player Agent 出价 (Competitors)
            # 我们先初始化一个全 0 的 bids 矩阵
            # shape: (num_pv, num_agent)
            current_tick_bids = np.zeros((pv_values.shape[0], num_agent))
            
            # 遍历所有 Agent
            for i, agent in enumerate(agents):
                # 如果是我们要测试的 Diffuser Agent (Player Index)，先跳过
                if i == player_index:
                    continue
                
                # 其他 Agent 正常出价
                if agent.remaining_budget >= envs.min_remaining_budget:
                    bid = agent.bidding(
                        tick_index, pv_values[:, i], pvalue_sigmas[:, i],
                        [x[i] for x in history_pvalue_infos],
                        [x[i] for x in history_bids],
                        [x[i] for x in history_auction_results],
                        [x[i] for x in history_impression_results],
                        history_least_winning_costs
                    )
                    current_tick_bids[:, i] = bid
            # 2. 轮到 Diffuser Agent (Player Index) "作弊"出价
            # 我们需要把 current_tick_bids (包含了对手的出价) 传给它
            current_remaining_budgets = np.array([agent.remaining_budget for agent in agents])
            target_agent = agents[player_index]
            if target_agent.remaining_budget >= envs.min_remaining_budget:
                # 关键：我们需要修改 Agent 的 bidding 接口，或者利用 getattr/setattr 动态注入信息
                # 建议：修改 BaseBiddingStrategy 或 DdBiddingStrategy 的 bidding 方法，
                # 增加 **kwargs 来接收额外信息
                
                # 这里假设你修改了 bidding 签名，传入 env 和 competitor_bids
                target_bid = target_agent.bidding(
                    tick_index, pv_values[:, player_index], pvalue_sigmas[:, player_index],
                    [x[player_index] for x in history_pvalue_infos],
                    [x[player_index] for x in history_bids],
                    [x[player_index] for x in history_auction_results],
                    [x[player_index] for x in history_impression_results],
                    history_least_winning_costs,
                    # --- 新增参数 ---
                    env=envs, 
                    competitor_bids=current_tick_bids, # 此时这里面已经有了别人的出价
                    agent_index=player_index,
                    pv_values=pv_values,
                    pvalue_sigmas=pvalue_sigmas,
                    current_budgets=current_remaining_budgets,
                )
                current_tick_bids[:, player_index] = target_bid
            bids = current_tick_bids
            # # 把原来的bids注释掉（各个agent同时出价）
            # bids = [
            #     agent.bidding(
            #         tick_index,
            #         pv_values[:, i],
            #         pvalue_sigmas[:, i],
            #         [x[i] for x in history_pvalue_infos],
            #         [x[i] for x in history_bids],
            #         [x[i] for x in history_auction_results],
            #         [x[i] for x in history_impression_results],
            #         history_least_winning_costs
            #     ) if agent.remaining_budget >= envs.min_remaining_budget
            #     else np.zeros(pv_values.shape[0])
            #     for i, agent in enumerate(agents)
            # ]

            # bids = np.array(bids).transpose()
            bids[bids < 0] = 0

            remaining_budget_list = np.array([agent.remaining_budget for agent in agents])
            done_list = np.ones(len(agents), dtype=int) if tick_index == (num_tick - 1) else (
                remaining_budget_list < envs.min_remaining_budget
            ).astype(int)
            # 回标机制
            ratio_max = None
            while ratio_max is None or ratio_max > 0:
                if ratio_max and ratio_max > 0:
                    over_cost_ratio = np.maximum((cost - remaining_budget_list) / (cost + 1e-4), 0)
                    adjust_over_cost(bids, over_cost_ratio, envs.slot_coefficients,winner_pit)

                xi_pit, slot_pit, cost_pit, is_exposed_pit, conversion_action_pit, least_winning_cost_pit, market_price_pit = \
                    envs.simulate_ad_bidding(pv_values, pvalue_sigmas, bids)

                real_cost = cost_pit * is_exposed_pit
                cost = real_cost.sum(axis=1)
                reward = conversion_action_pit.sum(axis=1)

                winner_pit = get_winner(slot_pit)
                over_cost_ratio = np.maximum((cost - remaining_budget_list) / (cost + 1e-4), 0)
                ratio_max = over_cost_ratio.max()
                
            # # ================= [可注释] 实际花费与规划花费误差监控与打印 =================
            # # 获取 Player (Diffuser Agent) 的真实花费
            # real_cost_val = cost[player_index]
            # # 获取 Player 的规划花费 (从 Strategy 中读取)
            # target_agent = agents[player_index]
            # # 防御性编程：确保 agent 有这个属性 (防止跑其他 Baseline 时报错)
            # if hasattr(target_agent, 'last_planned_spend'):
            #     plan_val = target_agent.last_planned_spend # 规划花的
            #     if hasattr(target_agent, 'last_predicted_cost'):
            #         pred_val = target_agent.last_predicted_cost # 预测花的
            #         alpha_val = target_agent.last_alpha
            #         # 计算两个误差
            #         # 1. 搜索误差 (Search Gap): 想花的 vs 预测能花的 (由二分搜索能力决定)
            #         search_err = pred_val - plan_val
            #         # 2. 预测误差 (Prediction Gap): 预测能花的 vs 实际花的 (由环境随机性/Bug决定)
            #         pred_err = real_cost_val - pred_val
                    
            #         print(f"📉 [Step {tick_index}] Cost Autopsy (Alpha={alpha_val:.4f}):")
            #         print(f"   1. Planned (Diffuser): {plan_val:.4f}")
            #         print(f"   2. Predicted (Oracle): {pred_val:.4f}  (Search Err: {search_err:+.4f})")
            #         print(f"   3. Real      (Env):    {real_cost_val:.4f}  (Pred   Err: {pred_err:+.4f})")
                    
            #         if abs(pred_err) > max(1.0, plan_val * 0.1):
            #             print("   ❌ PREDICTION FAIL: 预测值和真实值偏差巨大！环境预测逻辑有误或Competitor数据不对。")
            #         elif abs(search_err) > max(1.0, plan_val * 0.1):
            #             print("   ⚠️ SEARCH FAIL: 二分搜索找不到解 (悬崖效应)。")
            #         else:
            #             print("   ✅ PERFECT: 完美执行。")
            #         # 如果是作弊模式，理论上误差应该来源于随机曝光(Bernoulli Trial)
            #     else:
            #         print(f"📉 [Step {tick_index}]:")
            #         print(f"   1. Planned (Diffuser): {plan_val:.4f}")
            #         print(f"   2. Real      (Env):    {real_cost_val:.4f}  (Err: {real_cost_val-plan_val:+.4f})")
            # # =======================================================

            for i, agent in enumerate(agents):
                agent.remaining_budget -= cost[i]

            rewards += reward
            costs += cost
            
            history_bids.append(bids.transpose())
            history_least_winning_costs.append(least_winning_cost_pit)
            pvalue_info = np.stack((pv_values.T, pvalue_sigmas.T), axis=-1)
            history_pvalue_infos.append(pvalue_info)
            auction_info = np.stack((xi_pit, slot_pit, cost_pit), axis=-1)
            history_auction_results.append(auction_info)
            impression_info = np.stack((is_exposed_pit, conversion_action_pit), axis=-1)
            history_impression_results.append(impression_info)
            # after agent bids
            # logging.info('agent_bids: %.3fs', time.perf_counter() - t1)
            # t2 = time.perf_counter()
            if generate_log:
                # 如果想只保存player_index的轨迹
                p_idx = player_index
                start = p_idx
                end = p_idx + 1
                real_agent_ids = np.arange(start, end)
                train_data_tracker.train_logging(
                    episode, tick_index, 
                    pv_values[:, start:end], 
                    budgets[start:end], 
                    agents_cpa[start:end], 
                    agents_category[start:end],
                    remaining_budget_list[start:end], 
                    total_pv_num, 
                    pvalue_sigmas[:, start:end], 
                    bids[:, start:end], 
                    xi_pit[start:end, :], 
                    slot_pit[start:end, :], 
                    cost_pit[start:end, :], 
                    is_exposed_pit[start:end, :],
                    conversion_action_pit[start:end, :], 
                    least_winning_cost_pit, 
                    done_list[start:end],
                    agent_ids=real_agent_ids
                )
                # 如果48个player都要保存
                # train_data_tracker.train_logging(
                #     episode, tick_index, pv_values, budgets, agents_cpa, agents_category,
                #     remaining_budget_list, total_pv_num, pvalue_sigmas, bids,
                #     xi_pit, slot_pit, cost_pit, is_exposed_pit,
                #     conversion_action_pit, least_winning_cost_pit, done_list
                # )

            if player_analysis:
                tick_win_pv = np.sum(is_exposed_pit[player_index])
                tick_compete_pv = len(xi_pit[player_index])
                tick_all_win_bid = np.sum(bids[:, player_index] * is_exposed_pit[player_index])
                bid_mean = np.mean(bids[:, player_index])
                player_analysis.logging_player_tick(
                    episode, tick_index, player_index, agents_cpa[player_index],
                    budgets[player_index], reward[player_index],
                    cost[player_index], tick_compete_pv, tick_win_pv,
                    tick_all_win_bid, bid_mean
                )
        # 保存step0的planned_traj
        target_agent = agents[player_index]
        if hasattr(target_agent, 'planned_trajectory_at_t0') and target_agent.planned_trajectory_at_t0 is not None:
            # 存下来的应该是 Unnormalized 的真实物理值，Shape (48, 16)
            plan_traj = np.array(target_agent.planned_trajectory_at_t0)
            planned_data_buffer.append(plan_traj)
        else:
            logger.warning(f"Ep {episode}: No planned trajectory found for agent {player_index}")
            
        logger.info(f"Ep {episode} Finished. Collected Plans: {len(planned_data_buffer)}")
        if generate_log:
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"player{player_index}_episode{episode}.csv")
            else:
                # 兼容旧逻辑
                save_path = f"data/log_{player_index}/player{player_index}_episode{episode}.csv"
            train_data_tracker.generate_train_data(save_path)

    # after simulate
    # logging.info('simulate: %.3fs', time.perf_counter() - t2)
    end_time = time.time()
    logger.info(f"Total time elapsed: {end_time - begin_time} seconds")
    # log_memory_usage()

    player_analysis.player_multi_episode(agents[player_index].name)
    result = player_analysis.get_return_res(
        agents[player_index].name,
        player_index,
        agents[player_index].category
    )
    # ================= [修改] 保存单纯的规划数据 =================
    save_path = f"/AuctionNet/plan_traj/planned_step0_player{player_index}.npz"
    try:
        # 只保存 planned 数组
        # print(np.array(planned_data_buffer).shape) # (1, 48, 16)
        np.savez(save_path, planned=np.array(planned_data_buffer))
        logger.info(f"💾 Open-Loop Planned Trajectories saved to: {save_path}")
    except Exception as e:
        logger.error(f"Failed to save planned data: {e}")
    # ===========================================================
    return result




if __name__ == "__main__":
    pass
