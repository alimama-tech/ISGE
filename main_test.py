import sys
import argparse
sys.path.append("./strategy_train_env")
import gin
from run.run_test import run_test
from simul_bidding_env.Tracker.PlayerAnalysis import PlayerAnalysis
import torch
import numpy as np
import time
import logging
import json
import os
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.manual_seed(1)
np.random.seed(1)


@gin.configurable
def main():
    parser = argparse.ArgumentParser(description="Run test with specific start player.")
    parser.add_argument('--start_player_num', type=int, default=1, help='The starting index for player testing')
    parser.add_argument('--start_epi', type=int, default=0, help='Starting episode number')
    parser.add_argument('--task_id', type=int, default=0, help='Task ID for folder naming')
    args = parser.parse_args()
    gin_file = ["./config/test.gin"]
    gin.parse_config_files_and_bindings(gin_file, None)
    all_results = []
    global_raw_data = []
    total_test_player_num = 3
    task_save_dir = f"data/data_task_{args.task_id}"
    start_player_num = args.start_player_num
    print(f"当前测试的 Player 数量为: {total_test_player_num}, 从 Player {start_player_num} 开始")
    print(f"当前测试的 Start Episode {args.start_epi}, 保存至 {task_save_dir}")
    begin_time = time.time()
    for player_index in range(start_player_num, start_player_num + total_test_player_num):
        result = run_test(player_index=player_index, start_epi=args.start_epi, save_dir=task_save_dir)
        if "rawData" in result:
            global_raw_data.extend(result["rawData"])
        formated_result = {
            'score': float(result["score"]),
            'reward': int(result["reward"]),
            'playerIndex': player_index,
            'category': int(result["category"]),
            'win_pv_ratio': result["win_pv_ratio"],
            'budget_consumer_ratio': result["budget_consumer_ratio"],
            'second_price_ratio': result["second_price_ratio"],
            'cpa_exceedance_Rate': min(result["cpa_exceedance_Rate"], 49999), # 没有conversion就是49999
            'last_compete_tick_index': int(result["last_compete_tick_index"])
        }
        all_results.append(formated_result)

    collect_fields = ["score", "category", "reward", "win_pv_ratio", "budget_consumer_ratio", "second_price_ratio",
                      "last_compete_tick_index", "cpa_exceedance_Rate"]
    rank_score_field = "score"
    analysis_info = {}
    rank_score = 0.0
    for result in all_results:
        for key in collect_fields:
            if key == rank_score_field:
                rank_score += float(result[key])
            else:
                if key not in analysis_info.keys():
                    analysis_info[key] = []
                analysis_info[key].append(
                    float(format(result[key], '.3f')) if isinstance(result[key], float) else result[key])
    print("rank_score:", rank_score)
    print("analysis_info", analysis_info)
    
    # === 修改后的保存逻辑 ===
    # 获取按 Episode 分组的统计数据 (Dict[int, Dict])
    episode_stats = PlayerAnalysis.print_global_stats(global_raw_data)
    
    # 构造更清晰的保存结构
    save_data = {
        "metadata": {
            "start_player_num": start_player_num,
            "total_test_player_num": total_test_player_num,
            "end_player_num": start_player_num + total_test_player_num - 1,
            "count": len(global_raw_data) # 本次 Batch 包含的总轨迹数
        },
        "episode_stats": episode_stats  # 核心数据: { "0": {...}, "1": {...} }
    }

    results_dir = "./batch_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
        
    save_path = os.path.join(results_dir, f"player_{start_player_num}-{save_data['metadata']['end_player_num']}_epi{args.start_epi}.json")
    
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=4)
    print(f"✅ 统计结果已保存至: {save_path}")
    # 保存json
    # stats = PlayerAnalysis.print_global_stats(global_raw_data)
    # stats["start_player_num"] = start_player_num
    # stats["total_test_player_num"] = total_test_player_num
    # results_dir = "./batch_results"
    # if not os.path.exists(results_dir):
    #     os.makedirs(results_dir, exist_ok=True)
    # end_player_num = start_player_num + total_test_player_num - 1
    # save_path = os.path.join(results_dir, f"player_{start_player_num}-{end_player_num}_epi{args.start_epi}.json")
    # with open(save_path, "w") as f:
    #     json.dump(stats, f, indent=4)
    # print(f"✅ 统计结果已保存至: {save_path}")
    
    end_time = time.time()
    logger.info(f"Main_test Total time elapsed: {end_time - begin_time} seconds")


if __name__ == "__main__":
    main()
