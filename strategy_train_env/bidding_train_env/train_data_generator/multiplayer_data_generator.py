import os
import pandas as pd
import warnings
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm  # 用于显示进度条

warnings.filterwarnings('ignore')

# --- 全局辅助函数：用于并行进程调用 ---
def process_single_task(args):
    """
    并行工作的核心函数。
    args: (csv_path, output_dir, file_folder_path)
    """
    csv_path, output_dir, root_folder = args
    csv_filename = os.path.basename(csv_path)

    try:
        # 1. 读取数据
        df = pd.read_csv(csv_path)

        # 2. 处理数据
        # 技巧：在这里实例化类，只为了调用 _generate_train_data
        # 这样做的好处是完全不用修改 _generate_train_data 的代码，也不用担心多进程传参问题
        temp_generator = TrainDataGenerator(file_folder_path=root_folder)
        df_processed = temp_generator._generate_train_data(df)

        if df_processed is None or df_processed.empty:
            return None

        # 3. 保存单个文件
        trainData_filename = csv_filename.replace('.csv', '-rlData.csv')
        save_path = os.path.join(output_dir, trainData_filename)
        df_processed.to_csv(save_path, index=False)

        # 4. 返回处理后的DataFrame以便后续合并
        return df_processed

    except Exception as e:
        print(f"!!! 处理文件 {csv_filename} 失败: {e}")
        return None
# ------------------------------------


class TrainDataGenerator:
    """
    Train Data Generator for train models.
    Reads raw data and constructs training data suitable for model training.
    """

    def __init__(self, file_folder_path="./data/traffic"):
        self.file_folder_path = file_folder_path
        self.training_data_path = self.file_folder_path + "/" + "training_data_dense"

    def batch_generate_train_data(self):
        """
        并行批量处理 player[0]_episode[0-20].csv 的所有文件。
        """
        
        Tasknum = 16
        NUM_EPISODES = 7
        # 1. 构建任务列表
        tasks = []
        print(f"正在扫描并构建任务列表...")
        for task in range(1,17):
            LOG_DIR = f"/AuctionNet/data/data_task_{task}"
            os.makedirs(self.training_data_path, exist_ok=True)
            for episode_id in range(NUM_EPISODES):
                csv_filename = f'player1_episode{7*(task-1)+episode_id}.csv'
                csv_path = os.path.join(LOG_DIR, csv_filename)
                
                # 只有文件存在才加入任务
                if os.path.exists(csv_path):
                    # 将必要参数打包：(文件路径, 输出目录, 根目录)
                    tasks.append((csv_path, self.training_data_path, self.file_folder_path))

        total_files = len(tasks)
        print(f"共发现 {total_files} 个文件，开始并行处理...")

        training_data_list = []
        
        # 2. 使用进程池并行处理
        # max_workers如果不填，默认使用CPU核心数
        with ProcessPoolExecutor() as executor:
            # 提交任务
            futures = [executor.submit(process_single_task, task) for task in tasks]
            
            # 使用 tqdm 显示进度条
            for future in tqdm(as_completed(futures), total=total_files, desc="Processing"):
                result_df = future.result()
                if result_df is not None:
                    training_data_list.append(result_df)

        # 3. 整合所有数据
        if training_data_list:
            print("\n开始整合所有处理后的数据...")
            combined_dataframe = pd.concat(training_data_list, axis=0, ignore_index=True)
            combined_dataframe_path = os.path.join(self.training_data_path, "training_data_all-rlData.csv")
            combined_dataframe.to_csv(combined_dataframe_path, index=False)
            print(f"整合 {len(training_data_list)} 个文件的数据成功；保存至: {combined_dataframe_path}")
        else:
            print("警告：没有数据被成功处理。")

    def _generate_train_data(self, df):
        """
        Construct a DataFrame in reinforcement learning format based on the raw data.
        (保持原样，未修改任何逻辑)
        """

        training_data_rows = []

        for (
                deliveryPeriodIndex, advertiserNumber, advertiserCategoryIndex, budget,
                CPAConstraint), group in df.groupby(
            ['deliveryPeriodIndex', 'advertiserNumber', 'advertiserCategoryIndex', 'budget', 'CPAConstraint']):

            group = group.sort_values('timeStepIndex')

            group['timeStepIndex_volume'] = group.groupby('timeStepIndex')['timeStepIndex'].transform('size')

            timeStepIndex_volume_sum = group.groupby('timeStepIndex')['timeStepIndex_volume'].first()

            historical_volume = timeStepIndex_volume_sum.cumsum().shift(1).fillna(0).astype(int)
            group['historical_volume'] = group['timeStepIndex'].map(historical_volume)

            last_3_timeStepIndexs_volume = timeStepIndex_volume_sum.rolling(window=3, min_periods=1).sum().shift(
                1).fillna(0).astype(int)
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
            # 计算 realCost 和 realConversion
            realAllCost = (group['isExposed'] * group['cost']).sum()
            realAllConversion = group['conversionAction'].sum()

            for timeStepIndex in group['timeStepIndex'].unique():
                current_timeStepIndex_data = group[group['timeStepIndex'] == timeStepIndex]

                timeStepIndexNum = 48
                current_timeStepIndex_data.fillna(0, inplace=True)
                budget = current_timeStepIndex_data['budget'].iloc[0]
                remainingBudget = current_timeStepIndex_data['remainingBudget'].iloc[0]
                timeleft = (timeStepIndexNum - timeStepIndex) / timeStepIndexNum
                bgtleft = remainingBudget / budget if budget > 0 else 0

                state_features = current_timeStepIndex_data.iloc[0].to_dict()

                state = (
                    timeleft, bgtleft,
                    state_features['avg_bid_all'],
                    state_features['avg_bid_last_3'],
                    state_features['avg_leastWinningCost_all'],
                    state_features['avg_pValue_all'],
                    state_features['avg_conversionAction_all'],
                    state_features['avg_xi_all'],
                    state_features['avg_leastWinningCost_last_3'],
                    state_features['avg_pValue_last_3'],
                    state_features['avg_conversionAction_last_3'],
                    state_features['avg_xi_last_3'],
                    state_features['pValue_agg'],
                    state_features['timeStepIndex_volume_agg'],
                    state_features['last_3_timeStepIndexs_volume'],
                    state_features['historical_volume']
                )

                total_bid = current_timeStepIndex_data['bid'].sum()
                total_value = current_timeStepIndex_data['pValue'].sum()
                action = total_bid / total_value if total_value > 0 else 0
                reward = current_timeStepIndex_data[current_timeStepIndex_data['isExposed'] == 1][
                    'conversionAction'].sum()
                reward_continuous = current_timeStepIndex_data[current_timeStepIndex_data['isExposed'] == 1][
                    'pValue'].sum()

                done = 1 if timeStepIndex == timeStepIndexNum - 1 or current_timeStepIndex_data['isEnd'].iloc[
                    0] == 1 else 0

                training_data_rows.append({
                    'deliveryPeriodIndex': deliveryPeriodIndex,
                    'advertiserNumber': advertiserNumber,
                    'advertiserCategoryIndex': advertiserCategoryIndex,
                    'budget': budget,
                    'CPAConstraint': CPAConstraint,
                    'realAllCost': realAllCost,
                    'realAllConversion': realAllConversion,
                    'timeStepIndex': timeStepIndex,
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'reward_continuous': reward_continuous,
                    'done': done
                })

        training_data = pd.DataFrame(training_data_rows)
        training_data = training_data.sort_values(by=['deliveryPeriodIndex', 'advertiserNumber', 'timeStepIndex'])

        training_data['next_state'] = training_data.groupby(['deliveryPeriodIndex', 'advertiserNumber'])['state'].shift(
            -1)
        training_data.loc[training_data['done'] == 1, 'next_state'] = None
        return training_data


def run_generate_train_data():
    file_folder_path = "/AuctionNet/data/traffic"
    data_loader = TrainDataGenerator(file_folder_path=file_folder_path)
    data_loader.batch_generate_train_data()


if __name__ == '__main__':
    run_generate_train_data()