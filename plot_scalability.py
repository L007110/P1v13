import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import PercentFormatter

# --- 配置 ---
LOG_DIR = "training_results"

# 1. 定义要加载的文件
ABLATION1_FILE = "scalability_GNN_Ablation1.csv"  # Ablation 1 (Standard GAT + Sparse Graph)
ABLATION3_FILE = "scalability_GAT_Ablation3_Stabilized.csv"  # P1v13 (Stabilized GAT + Sparse Graph)

PLOT_FILE = "scalability_plot_Abl1_vs_P1v13_Stabilized.png"


# --- 核心筛选函数 (确保只提取 GNN-DRL) ---
def filter_and_label(df, new_name):
    """从混合文件中提取 GNN-DRL 行并贴上新标签"""
    df_gnn = df[df['model'] == 'GNN-DRL'].copy()
    if df_gnn.empty:
        raise ValueError(f"GNN-DRL model data not found in the file labeled {new_name}.")
    df_gnn['model'] = new_name
    return df_gnn


def plot_scalability_results():
    print(f"--- Loading scalability results (Ablation 1 vs P1v13 Comparison) ---")

    try:
        # 1. 加载 Ablation 1 数据 (Standard GAT)
        df_ablation1 = pd.read_csv(os.path.join(LOG_DIR, ABLATION1_FILE))
        df_abl1_gnn = filter_and_label(df_ablation1, 'GAT (Ablation 1: Standard)')

        # 2. 加载 P1v13 数据 (Stabilized GAT)
        df_p1v13 = pd.read_csv(os.path.join(LOG_DIR, ABLATION3_FILE))
        df_p1v13_gnn = filter_and_label(df_p1v13, 'GAT (P1v13: Stabilized w/ Gating)')

    except FileNotFoundError as e:
        print(f"!!! Error: Log file not found. {e}")
        return
    except ValueError as e:
        print(f"!!! Error: {e}. Check model labels.")
        return

    # 3. 合并数据
    df = pd.concat([df_abl1_gnn, df_p1v13_gnn], ignore_index=True)

    sns.set_theme(style="whitegrid", palette="deep")
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    fig.suptitle("P1v13: Effect of Stabilization on Sparse GAT (Ablation 1)", fontsize=16, fontweight='bold')

    hue_order = ["GAT (Ablation 1: Standard)", "GAT (P1v13: Stabilized w/ Gating)"]

    # --- 图 1: V2V 成功率 ---
    ax1 = axes[0]
    sns.lineplot(data=df, x="vehicle_count", y="v2v_success_rate", hue="model", hue_order=hue_order, style="model",
                 markers=True, markersize=10, linewidth=2.5, ax=ax1)
    ax1.set_title("V2V Success Rate vs. Vehicle Density", fontsize=14)
    ax1.set_xlabel("Number of Vehicles", fontsize=12)
    ax1.set_ylabel("V2V Success Rate", fontsize=12)
    ax1.set_ylim(0, 1.05)
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.legend(title="Model", fontsize=11)

    # --- 图 2: V2I 和容量 ---
    ax2 = axes[1]
    sns.lineplot(data=df, x="vehicle_count", y="v2i_sum_capacity_mbps", hue="model", hue_order=hue_order, style="model",
                 markers=True, markersize=10, linewidth=2.5, ax=ax2)
    ax2.set_title("V2I Sum Capacity vs. Vehicle Density", fontsize=14)
    ax2.set_xlabel("Number of Vehicles", fontsize=12)
    ax2.set_ylabel("V2I Sum Capacity (Mbps)", fontsize=12)
    ax2.set_ylim(bottom=0)
    ax2.legend(title="Model", fontsize=11)

    # --- 图 3: 推理时间 ---
    ax3 = axes[2]
    sns.lineplot(data=df, x="vehicle_count", y="decision_time_ms", hue="model", hue_order=hue_order, style="model",
                 markers=True, markersize=10, linewidth=2.5, ax=ax3)
    ax3.set_title("Inference Time vs. Vehicle Density", fontsize=14)
    ax3.set_xlabel("Number of Vehicles", fontsize=12)
    ax3.set_ylabel("Decision Time (ms)", fontsize=12)
    ax3.set_ylim(bottom=0)
    ax3.legend(title="Model", fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, PLOT_FILE), dpi=300, bbox_inches='tight')
    print(f"--- Scalability comparison plot saved to {PLOT_FILE} ---")


if __name__ == "__main__":
    plot_scalability_results()
