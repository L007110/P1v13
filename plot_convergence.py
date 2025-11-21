import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import PercentFormatter

# --- 配置 ---
LOG_DIR = "training_results"

# 1. 定义要加载的收敛日志文件 (直接比较 GAT 两个版本)
CONVERGENCE_FILES = {
    "GAT (Ablation 1: Standard)": "convergence_GNN_Ablation1.csv",
    "GAT (P1v13: Stabilized w/ Gating)": "convergence_GAT_Ablation3_Stabilized.csv"
}

PLOT_FILE = "convergence_plot_Abl1_vs_P1v13_Stabilized.png"
ROLLING_WINDOW = 50


# --- 结束配置 ---

def plot_convergence_results():
    print("--- Loading convergence data (Ablation 1 vs P1v13 Comparison) ---")

    all_data = []
    hue_order = []

    for model_name, filename in CONVERGENCE_FILES.items():
        file_path = os.path.join(LOG_DIR, filename)
        try:
            df = pd.read_csv(file_path)
            df['model'] = model_name
            all_data.append(df)
            hue_order.append(model_name)
            print(f"Loaded {filename} as '{model_name}'")
        except FileNotFoundError:
            print(f"!!! Warning: Log file not found, skipping: {file_path}")

    if not all_data:
        print("!!! Error: No data loaded. Aborting plot.")
        return

    df = pd.concat(all_data, ignore_index=True)

    # 计算平滑后的指标 (分组平滑)
    df['reward_smooth'] = df.groupby('model')['cumulative_reward'].transform(
        lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean())
    df['v2v_success_smooth'] = df.groupby('model')['v2v_success_rate'].transform(
        lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean())
    df['v2i_cap_smooth'] = df.groupby('model')['v2i_sum_capacity'].transform(
        lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean())

    # --- 开始绘图 ---
    sns.set_theme(style="whitegrid", palette="deep")
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    fig.suptitle(f"Ablation P1v13: Effect of Stabilization on Sparse GAT (Window={ROLLING_WINDOW})", fontsize=16,
                 fontweight='bold')

    # 图 1: 累积奖励
    sns.lineplot(data=df, x="epoch", y="reward_smooth", hue="model", hue_order=hue_order, style="model", linewidth=2.5,
                 ax=axes[0])
    axes[0].set_title("Cumulative Reward Convergence", fontsize=14)
    axes[0].set_ylabel("Smoothed Cumulative Reward", fontsize=12)
    axes[0].legend(title="Model", fontsize=11)

    # 图 2: V2V 成功率
    sns.lineplot(data=df, x="epoch", y="v2v_success_smooth", hue="model", hue_order=hue_order, style="model",
                 linewidth=2.5, ax=axes[1])
    axes[1].set_title("V2V Success Rate Convergence", fontsize=14)
    axes[1].set_ylabel("Smoothed V2V Success Rate", fontsize=12)
    axes[1].set_ylim(0, 1.05)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].legend(title="Model", fontsize=11)

    # 图 3: V2I 和容量
    sns.lineplot(data=df, x="epoch", y="v2i_cap_smooth", hue="model", hue_order=hue_order, style="model", linewidth=2.5,
                 ax=axes[2])
    axes[2].set_title("V2I Sum Capacity Convergence", fontsize=14)
    axes[2].set_ylabel("Smoothed V2I Capacity (Mbps)", fontsize=12)
    axes[2].set_ylim(bottom=0)
    axes[2].legend(title="Model", fontsize=11)

    for ax in axes: ax.set_xlabel("Training Epoch", fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, PLOT_FILE), dpi=300, bbox_inches='tight')
    print(f"--- Convergence comparison plot saved to {PLOT_FILE} ---")


if __name__ == "__main__":
    plot_convergence_results()
