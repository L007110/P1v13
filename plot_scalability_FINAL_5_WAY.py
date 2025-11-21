import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import PercentFormatter

# --- 配置 ---
LOG_DIR = "training_results"

# 1. 定义要加载的文件
BASELINE_FILE = "scalability_Baseline.csv"  # 混合基线数据
ABLATION1_FILE = "scalability_GNN_Ablation1.csv"  # Ablation 1 稀疏数据
P1V13_FILE = "scalability_GAT_Ablation3_Stabilized.csv"  # Stabilized GAT 数据

PLOT_FILE = "scalability_plot_FINAL_5_WAY.png"


# --- 核心筛选函数 ---
def filter_and_label(df, model_name, new_name):
    """从文件中提取特定 model_name 的行并贴上新标签"""
    df_filtered = df[df['model'] == model_name].copy()
    if df_filtered.empty:
        print(f"Warning: Could not find model '{model_name}' in the loaded file.")
        return None

    df_filtered['model'] = new_name
    return df_filtered


def plot_scalability_results():
    print(f"--- Loading FINAL 5-Way Scalability Results ---")

    try:
        # 1. 加载基线数据 (混合文件)
        df_baseline_full = pd.read_csv(os.path.join(LOG_DIR, BASELINE_FILE))
        df_abl1_full = pd.read_csv(os.path.join(LOG_DIR, ABLATION1_FILE))
        df_p1v13_full = pd.read_csv(os.path.join(LOG_DIR, P1V13_FILE))

        # 2. 提取并标记基线模型 (Competitors + GNN Full)
        df_baseline_full_gnn = filter_and_label(df_baseline_full, 'GNN-DRL', 'GNN (Baseline Full)')
        df_no_gnn = filter_and_label(df_baseline_full, 'No-GNN DRL', 'No-GNN DRL')
        df_dqn = filter_and_label(df_baseline_full, 'Standard DQN', 'Standard DQN')

        # 3. 提取 Ablation 1 (Sparse Base)
        df_abl1_gnn = filter_and_label(df_abl1_full, 'GNN-DRL', 'GAT (Ablation 1 Sparse)')

        # 4. 提取 P1V13 (Stabilized Sparse)
        df_p1v13_gnn = filter_and_label(df_p1v13_full, 'GNN-DRL', 'GNN (P1v13 Stabilized)')

    except FileNotFoundError as e:
        print(f"!!! Error: Log file not found. {e}")
        return

    # 5. 合并所有数据
    all_dfs = [df for df in [df_p1v13_gnn, df_abl1_gnn, df_baseline_full_gnn, df_no_gnn, df_dqn] if df is not None]
    if not all_dfs:
        print("!!! Error: No valid dataframes to plot.")
        return

    df = pd.concat(all_dfs, ignore_index=True)

    sns.set_theme(style="whitegrid", palette="deep")
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    fig.suptitle("Final Comparison (5-Way): GAT Optimization Scalability", fontsize=16, fontweight='bold')

    # 绘制顺序：最优方案 -> 稀疏方案 -> 原始方案 -> 竞争对手
    hue_order = ["GNN (P1v13 Stabilized)", "GAT (Ablation 1 Sparse)", "GNN (Baseline Full)", "No-GNN DRL",
                 "Standard DQN"]

    # V2V Success Rate:
    axes[0].set_title("V2V Success Rate vs. Vehicle Density", fontsize=14)
    sns.lineplot(data=df, x="vehicle_count", y="v2v_success_rate", hue="model", hue_order=hue_order, style="model",
                 markers=True, markersize=10, linewidth=2.5, ax=axes[0])
    axes[0].set_ylabel("V2V Success Rate", fontsize=12)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].legend(title="Model", fontsize=11)

    # V2I Sum Capacity:
    axes[1].set_title("V2I Sum Capacity vs. Vehicle Density", fontsize=14)
    sns.lineplot(data=df, x="vehicle_count", y="v2i_sum_capacity_mbps", hue="model", hue_order=hue_order, style="model",
                 markers=True, markersize=10, linewidth=2.5, ax=axes[1])
    axes[1].set_ylabel("V2I Sum Capacity (Mbps)", fontsize=12)
    axes[1].legend(title="Model", fontsize=11)

    # Inference Time:
    axes[2].set_title("Inference Time vs. Vehicle Density", fontsize=14)
    sns.lineplot(data=df, x="vehicle_count", y="decision_time_ms", hue="model", hue_order=hue_order, style="model",
                 markers=True, markersize=10, linewidth=2.5, ax=axes[2])
    axes[2].set_ylabel("Decision Time (ms)", fontsize=12)
    axes[2].legend(title="Model", fontsize=11)

    for ax in axes: ax.set_xlabel("Number of Vehicles", fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, PLOT_FILE), dpi=300, bbox_inches='tight')
    print(f"--- Final 5-way scalability plot saved to {PLOT_FILE} ---")


if __name__ == "__main__":
    plot_scalability_results()
