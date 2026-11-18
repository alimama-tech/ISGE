import os
import pandas as pd
import warnings
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings('ignore')
iteration = 3  # 您可以根据需要修改 iteration 版本

class TrainDataGenerator:
    """
    Train Data Generator (Adapted for Task-based Parallel Data Collection).
    Reads raw auction logs from 'data/data_task_X' folders and constructs RL training trajectories.
    """
    def __init__(self):
        # 基础数据目录 (包含 data_task_1, data_task_2 ... 的父目录)
        self.BASE_DATA_DIR = "/AuctionNet/data"
        
        # 输出目录
        self.OUTPUT_DIR = f"/AuctionNet/data/traffic/training_data_rlData_folder_iter{iteration}_cut_ISGE_explore2/"
        
        # 任务数量 (对应 tmux_data_gen.sh 中的 TOTAL_TASKS)
        self.NUM_TASKS = 16
        
        # 确保输出目录存在
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

    def _process_single_task(self, task_id):
        """
        [子进程任务] 处理单个 Task 文件夹下的所有 Episode 数据，并合并为一个 CSV。
        """
        task_data_list = []
        # 拼接任务文件夹路径: data/data_task_{task_id}
        task_sub_dir = os.path.join(self.BASE_DATA_DIR, f"data_task_{task_id}")
        
        if not os.path.exists(task_sub_dir):
            print(f"⚠️ Task {task_id} 文件夹不存在: {task_sub_dir}")
            return None

        # 获取该文件夹下所有的 csv 文件 (player1_episode*.csv)
        csv_files = glob.glob(os.path.join(task_sub_dir, "*.csv"))
        
        if not csv_files:
            print(f"⚠️ Task {task_id} 文件夹为空")
            return None

        print(f"Task {task_id}: 发现 {len(csv_files)} 个数据文件，开始处理...")

        for source_path in csv_files:
            try:
                # 1. 读取原始数据
                df = pd.read_csv(source_path)
                
                # 2. 转换为 RL 轨迹数据 (调用原有核心函数)
                df_processed = self._generate_train_data(df)
                
                # 3. 添加到列表
                task_data_list.append(df_processed)
                
            except Exception as e:
                print(f"❌ Error processing {os.path.basename(source_path)}: {e}")
                continue

        if not task_data_list:
            return None

        # 4. 合并该 Task 的所有数据
        task_combined_df = pd.concat(task_data_list, axis=0, ignore_index=True)
        
        # 5. 保存单 Task 的轨迹文件 (便于调试和中间存储)
        output_filename = f"training_data_task{task_id}-rlData.csv"
        output_path = os.path.join(self.OUTPUT_DIR, output_filename)
        task_combined_df.to_csv(output_path, index=False)
        
        return output_path

    def batch_generate_train_data(self):
        """
        并行批量处理所有 Task 文件夹的数据。
        """
        print(f"🚀 开始处理数据...")
        print(f"📂 源目录模式: {self.BASE_DATA_DIR}/data_task_X")
        print(f"💾 输出目录: {self.OUTPUT_DIR}")

        all_task_files = []
        
        # 使用进程池并行处理 (按 Task 并行)
        # max_workers 设置为任务数即可
        with ProcessPoolExecutor(max_workers=self.NUM_TASKS) as executor:
            future_to_task = {
                executor.submit(self._process_single_task, task_id): task_id 
                for task_id in range(1, self.NUM_TASKS + 1) # task_id 从 1 到 16
            }
            
            for future in as_completed(future_to_task):
                task_id = future_to_task[future]
                try:
                    result_path = future.result()
                    if result_path:
                        print(f"✅ Task {task_id} 处理完成 -> {os.path.basename(result_path)}")
                        all_task_files.append(result_path)
                    else:
                        print(f"⚠️ Task {task_id} 无有效数据")
                except Exception as exc:
                    print(f"❌ Task {task_id} 发生异常: {exc}")

        # 6. 最后整合所有 Task 的数据生成一个总文件
        print("\n正在整合所有 Task 的数据生成 training_data_all-rlData.csv ...")
        if all_task_files:
            # 读取所有分片文件并合并
            # 注意：如果数据量特别大 (5040条轨迹)，内存可能紧张，但一般 DataFrame 还可以承受
            df_list = [pd.read_csv(f) for f in all_task_files]
            combined_dataframe = pd.concat(df_list, axis=0, ignore_index=True)
            
            # # 按 episode (deliveryPeriodIndex) 排序，保证顺序整洁
            # if 'deliveryPeriodIndex' in combined_dataframe.columns:
            #     combined_dataframe = combined_dataframe.sort_values(by=['deliveryPeriodIndex', 'timeStepIndex'])
            # 【修改代码】
            # 顺序说明：先按玩家排 -> 再按局数排 -> 最后按时间步排
            sort_cols = ['advertiserNumber', 'deliveryPeriodIndex', 'timeStepIndex']
            existing_sort_cols = [c for c in sort_cols if c in combined_dataframe.columns]
            combined_dataframe = combined_dataframe.sort_values(by=existing_sort_cols)

            combined_path = os.path.join(self.OUTPUT_DIR, "training_data_all-rlData.csv")
            combined_dataframe.to_csv(combined_path, index=False)
            print(f"🎉 全部完成！总数据已保存至: {combined_path}")
        else:
            print("❌ 没有生成任何有效数据，跳过总文件合并。")

    def _generate_train_data(self, df):
        """
        核心逻辑：将原始日志转换为 RL 轨迹格式 (State, Action, Reward, Next State)
        保持原有逻辑不变。
        """
        training_data_rows = []

        # 按 Episode (deliveryPeriodIndex) 和 Agent 分组处理
        for (deliveryPeriodIndex, advertiserNumber, advertiserCategoryIndex, budget, CPAConstraint), group in df.groupby(
                ['deliveryPeriodIndex', 'advertiserNumber', 'advertiserCategoryIndex', 'budget', 'CPAConstraint']):

            group = group.sort_values('timeStepIndex')

            # --- 特征工程 ---
            group['timeStepIndex_volume'] = group.groupby('timeStepIndex')['timeStepIndex'].transform('size')
            timeStepIndex_volume_sum = group.groupby('timeStepIndex')['timeStepIndex_volume'].first()

            historical_volume = timeStepIndex_volume_sum.cumsum().shift(1).fillna(0).astype(int)
            group['historical_volume'] = group['timeStepIndex'].map(historical_volume)

            last_3_timeStepIndexs_volume = timeStepIndex_volume_sum.rolling(window=3, min_periods=1).sum().shift(1).fillna(0).astype(int)
            group['last_3_timeStepIndexs_volume'] = group['timeStepIndex'].map(last_3_timeStepIndexs_volume)

            group_agg = group.groupby('timeStepIndex').agg({
                'bid': 'mean',
                'leastWinningCost': 'mean',
                'conversionAction': 'mean',
                'xi': 'mean',
                'pValue': 'mean',
                'timeStepIndex_volume': 'first'
            }).reset_index()

            for col in ['bid', 'leastWinningCost', 'conversionAction', 'xi', 'pValue']:
                group_agg[f'avg_{col}_all'] = group_agg[col].expanding().mean().shift(1)
                group_agg[f'avg_{col}_last_3'] = group_agg[col].rolling(window=3, min_periods=1).mean().shift(1)

            group = group.merge(group_agg, on='timeStepIndex', suffixes=('', '_agg'))

            # 计算真实总消耗和总转化
            realAllCost = (group['isExposed'] * group['cost']).sum()
            realAllConversion = group['conversionAction'].sum()

            # --- 逐个 TimeStep 构建 State-Action-Reward ---
            unique_time_steps = group['timeStepIndex'].unique()
            timeStepIndexNum = 48
            
            for timeStepIndex in unique_time_steps:
                current_data = group[group['timeStepIndex'] == timeStepIndex]
                current_data.fillna(0, inplace=True)
                
                # 提取状态特征
                budget_val = current_data['budget'].iloc[0]
                remainingBudget = current_data['remainingBudget'].iloc[0]
                
                timeleft = (timeStepIndexNum - timeStepIndex) / timeStepIndexNum
                bgtleft = remainingBudget / budget_val if budget_val > 0 else 0
                
                feats = current_data.iloc[0].to_dict()

                state = (
                    timeleft, bgtleft,
                    feats.get('avg_bid_all', 0),
                    feats.get('avg_bid_last_3', 0),
                    feats.get('avg_leastWinningCost_all', 0),
                    feats.get('avg_pValue_all', 0),
                    feats.get('avg_conversionAction_all', 0),
                    feats.get('avg_xi_all', 0),
                    feats.get('avg_leastWinningCost_last_3', 0),
                    feats.get('avg_pValue_last_3', 0),
                    feats.get('avg_conversionAction_last_3', 0),
                    feats.get('avg_xi_last_3', 0),
                    feats.get('pValue_agg', 0),
                    feats.get('timeStepIndex_volume_agg', 0),
                    feats.get('last_3_timeStepIndexs_volume', 0),
                    feats.get('historical_volume', 0)
                )

                # 计算 Action (alpha)
                total_bid = current_data['bid'].sum()
                total_value = current_data['pValue'].sum()
                action = total_bid / total_value if total_value > 0 else 0
                
                # 计算 Reward
                reward = current_data[current_data['isExposed'] == 1]['conversionAction'].sum()
                reward_continuous = current_data[current_data['isExposed'] == 1]['pValue'].sum()

                # 判断 Done
                is_end_flag = current_data['isEnd'].iloc[0] if 'isEnd' in current_data else 0
                done = 1 if timeStepIndex == timeStepIndexNum - 1 or is_end_flag == 1 else 0

                training_data_rows.append({
                    'deliveryPeriodIndex': deliveryPeriodIndex,
                    'advertiserNumber': advertiserNumber,
                    'advertiserCategoryIndex': advertiserCategoryIndex,
                    'budget': budget_val,
                    'CPAConstraint': CPAConstraint,
                    'realAllCost': realAllCost,
                    'realAllConversion': realAllConversion,
                    'timeStepIndex': timeStepIndex,
                    'state': str(state),
                    'action': action,
                    'reward': reward,
                    'reward_continuous': reward_continuous,
                    'done': done
                })

        # 构建 DataFrame
        training_data = pd.DataFrame(training_data_rows)
        if not training_data.empty:
            training_data = training_data.sort_values(by=['deliveryPeriodIndex', 'advertiserNumber', 'timeStepIndex'])
            
            # 构建 Next State
            training_data['next_state'] = training_data.groupby(['deliveryPeriodIndex', 'advertiserNumber'])['state'].shift(-1)
            training_data.loc[training_data['done'] == 1, 'next_state'] = None
            
        return training_data


def run_generate_train_data():
    generator = TrainDataGenerator()
    generator.batch_generate_train_data()


if __name__ == '__main__':
    run_generate_train_data()