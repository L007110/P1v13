import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import PercentFormatter

LOG_DIR = "training_results"

# 1. 定义要加载的文件
ROBUSTNESS_ABLATION1 = "robustness_GNN_Ablation1.csv"
POLICY_ABLATION1 = "policy_GNN_Ablation1.csv"
ROBUSTNESS_P1V13 = "robustness_GAT_Ablation3_Stabilized.csv"
POLICY_P1V13 = "policy_GAT_Ablation3_Stabilized.csv"

# 2. 定义输出文件
PLOT_FILE_ROBUSTNESS = "robustness_plot_Abl1_vs_P1v13_Stabilized.png"
PLOT_FILE_POLICY = "policy_analysis_plot_Abl1_vs_P1v13_Stabilized.png"


def filter_and_label(df, new_name):
    """从混合文件中提取 GNN-DRL 行并贴上新标签"""
    df_gnn = df[df['model'] == 'GNN-DRL'].copy()
    if df_gnn.empty:
        raise ValueError(f"GNN-DRL model data not found in the file labeled {new_name}.")
    df_gnn['model'] = new_name
    return df_gnn


def plot_robustness():
    print(f"--- Loading robustness results (Ablation 1 vs P1v13 Comparison) ---")

    try:
        # 1. 加载 Ablation 1 数据 (Standard GAT)
        df_robustness_abl1 = pd.read_csv(os.path.join(LOG_DIR, ROBUSTNESS_ABLATION1))
        df_abl1_gnn = filter_and_label(df_robustness_abl1, 'GAT (Ablation 1: Standard)')

        # 2. 加载 P1v13 数据 (Stabilized GAT)
        df_robustness_p1v13 = pd.read_csv(os.path.join(LOG_DIR, ROBUSTNESS_P1V13))
        df_p1v13_gnn = filter_and_label(df_robustness_p1v13, 'GAT (P1v13: Stabilized w/ Gating)')

    except FileNotFoundError as e:
        print(f"!!! Error: Results file not found. {e}")
        return
    except ValueError as e:
        print(f"!!! Error: {e}")
        return

    # 3. 合并
    df = pd.concat([df_abl1_gnn, df_p1v13_gnn], ignore_index=True)

    sns.set_theme(style="whitegrid", palette="deep")
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    fig.suptitle("P1v13: Effect of Stabilization on Sparse GAT Robustness", fontsize=16, fontweight='bold')

    hue_order = ["GAT (Ablation 1: Standard)", "GAT (P1v13: Stabilized w/ Gating)"]

    # --- 图 1: V2V 成功率 vs. 速度 ---
    ax1 = axes[0]
    sns.lineplot(data=df, x="speed_kmh", y="v2v_success_rate", hue="model", style="model", markers=True, markersize=10,
                 linewidth=2.5, ax=ax1, hue_order=hue_order, style_order=hue_order)
    ax1.set_title("V2V Success Rate vs. Vehicle Speed", fontsize=14)
    ax1.set_xlabel("Vehicle Speed (km/h)", fontsize=12)
    ax1.set_ylabel("V2V Success Rate", fontsize=12)
    ax1.set_ylim(0.8, 1.05)
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.legend(title="Model", fontsize=11)
    ax1.grid(True, linestyle='--')

    # --- 图 2: V2I 和容量 vs. 速度 ---
    ax2 = axes[1]
    sns.lineplot(data=df, x="speed_kmh", y="v2i_sum_capacity_mbps", hue="model", style="model", markers=True,
                 markersize=10, linewidth=2.5, ax=ax2, hue_order=hue_order, style_order=hue_order)
    ax2.set_title("V2I Sum Capacity vs. Vehicle Speed", fontsize=14)
    ax2.set_xlabel("Vehicle Speed (km/h)", fontsize=12)
    ax2.set_ylabel("V2I Sum Capacity (Mbps)", fontsize=12)
    ax2.set_ylim(bottom=0)
    ax2.legend(title="Model", fontsize=11)
    ax2.grid(True, linestyle='--')

    # --- 图 3: P95 延迟 vs. 速度 ---
    ax3 = axes[2]
    sns.lineplot(data=df, x="speed_kmh", y="p95_delay_ms", hue="model", style="model", markers=True, markersize=10,
                 linewidth=2.5, ax=ax3, hue_order=hue_order, style_order=hue_order)
    ax3.set_title("V2V P95 Latency vs. Vehicle Speed", fontsize=14)
    ax3.set_xlabel("Vehicle Speed (km/h)", fontsize=12)
    ax3.set_ylabel("P95 Delay (ms)", fontsize=12)
    ax3.set_ylim(bottom=0)
    ax3.legend(title="Model", fontsize=11)
    ax3.grid(True, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, PLOT_FILE_ROBUSTNESS), dpi=300, bbox_inches='tight')
    print(f"--- Robustness comparison plot saved to {PLOT_FILE_ROBUSTNESS} ---")
    plt.close()


def plot_policy_comparison():
    print(f"--- Loading policy data (Ablation 1 vs P1v13 Comparison) ---")

    try:
        # 1. 加载 Ablation 1 策略数据
        df_policy_abl1 = pd.read_csv(os.path.join(LOG_DIR, POLICY_ABLATION1))
        df_base_gnn = filter_and_label(df_policy_abl1, 'GAT (Ablation 1: Standard)')

        # 2. 加载 P1v13 策略数据
        df_policy_p1v13 = pd.read_csv(os.path.join(LOG_DIR, POLICY_P1V13))
        df_p1v13_gnn = filter_and_label(df_policy_p1v13, 'GAT (P1v13: Stabilized w/ Gating)')

    except FileNotFoundError as e:
        print(f"!!! Error: Policy file not found. {e}")
        return
    except ValueError as e:
        print(f"!!! Error: {e}")
        return

    # 合并
    df = pd.concat([df_base_gnn, df_p1v13_gnn], ignore_index=True)

    # 数据处理
    snr_bins = [-100, 0, 5, 10, 15, 100]
    snr_labels = ["< 0 dB", "0-5 dB", "5-10 dB", "10-15 dB", "> 15 dB"]
    df['snr_bin'] = pd.cut(df['snr_dB'], bins=snr_bins, labels=snr_labels, right=False)
    df['power_ratio'] = (df['power_level'] + 1) / 10.0

    sns.set_theme(style="whitegrid", palette="viridis")

    # --- 策略图 (1x2 布局) ---
    models_to_plot = ["GAT (Ablation 1: Standard)", "GAT (P1v13: Stabilized w/ Gating)"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    fig.suptitle("P1v13 Policy Analysis: Standard GAT vs. Stabilized GAT", fontsize=16, fontweight='bold')

    for i, model_name in enumerate(models_to_plot):
        ax = axes[i]
        model_data = df[df['model'] == model_name]

        sns.violinplot(data=model_data, x="snr_bin", y="power_ratio", ax=ax, inner="quartile", scale="width",
                       order=snr_labels)
        ax.set_title(f"Policy: {model_name}", fontsize=14)
        ax.set_xlabel("Previous V2V SNR", fontsize=12)
        ax.set_ylim(0, 1.1)
        if i == 0:
            ax.set_ylabel("Chosen Power Ratio", fontsize=12)
        else:
            ax.set_ylabel("")

    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, PLOT_FILE_POLICY), dpi=300, bbox_inches='tight')
    print(f"--- Policy comparison plot saved to {PLOT_FILE_POLICY} ---")
    plt.close()


if __name__ == "__main__":
    plot_robustness()
    plot_policy_comparison()
