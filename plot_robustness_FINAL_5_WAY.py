import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import PercentFormatter

LOG_DIR = "training_results"

# 1. 定义要加载的文件
ROBUSTNESS_BASELINE = "robustness_Baseline.csv"
POLICY_BASELINE = "policy_Baseline.csv"
ROBUSTNESS_ABLATION1 = "robustness_GNN_Ablation1.csv"
POLICY_ABLATION1 = "policy_GNN_Ablation1.csv"
ROBUSTNESS_P1V13 = "robustness_GAT_Ablation3_Stabilized.csv"
POLICY_P1V13 = "policy_GAT_Ablation3_Stabilized.csv"

# 2. 定义输出文件
PLOT_FILE_ROBUSTNESS = "robustness_vs_speed_plot_FINAL_5_WAY.png"
PLOT_FILE_POLICY = "policy_analysis_plot_FINAL_5_WAY.png"


def filter_and_label(df, model_name, new_name):
    """从文件中提取特定 model_name 的行并贴上新标签"""
    df_filtered = df[df['model'] == model_name].copy()
    if df_filtered.empty:
        print(f"Warning: Could not find model '{model_name}' in the loaded file.")
        return None

    df_filtered['model'] = new_name
    return df_filtered


def load_and_combine_data(robustness=True):
    """加载并合并所有 5 个模型的数据"""
    base_file = ROBUSTNESS_BASELINE if robustness else POLICY_BASELINE
    abl1_file = ROBUSTNESS_ABLATION1 if robustness else POLICY_ABLATION1
    p1v13_file = ROBUSTNESS_P1V13 if robustness else POLICY_P1V13

    try:
        # 1. 加载所有原始文件
        df_baseline_full = pd.read_csv(os.path.join(LOG_DIR, base_file))
        df_abl1_full = pd.read_csv(os.path.join(LOG_DIR, abl1_file))
        df_p1v13_full = pd.read_csv(os.path.join(LOG_DIR, p1v13_file))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing file for combination: {e}")

    # 2. 提取并标记所有模型
    df_p1v13_gnn = filter_and_label(df_p1v13_full, 'GNN-DRL', 'GNN (P1v13 Stabilized)')
    df_abl1_gnn = filter_and_label(df_abl1_full, 'GNN-DRL', 'GAT (Ablation 1 Sparse)')
    df_baseline_full_gnn = filter_and_label(df_baseline_full, 'GNN-DRL', 'GNN (Baseline Full)')

    # 竞争对手直接从 Baseline Full 文件中提取
    df_no_gnn = filter_and_label(df_baseline_full, 'No-GNN DRL', 'No-GNN DRL')
    df_dqn = filter_and_label(df_baseline_full, 'Standard DQN', 'Standard DQN')

    # 3. 合并所有数据
    all_dfs = [df for df in [df_p1v13_gnn, df_abl1_gnn, df_baseline_full_gnn, df_no_gnn, df_dqn] if df is not None]

    if not all_dfs:
        raise FileNotFoundError("No valid dataframes found for plotting.")

    return pd.concat(all_dfs, ignore_index=True)


def plot_robustness():
    print(f"--- Loading FINAL 5-Way Robustness Results ---")

    try:
        df = load_and_combine_data(robustness=True)
    except FileNotFoundError as e:
        print(f"!!! Error: {e}")
        return

    sns.set_theme(style="whitegrid", palette="deep")
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    fig.suptitle("Final Comparison (5-Way): GAT Optimization Robustness", fontsize=16, fontweight='bold')

    hue_order = ["GNN (P1v13 Stabilized)", "GAT (Ablation 1 Sparse)", "GNN (Baseline Full)", "No-GNN DRL",
                 "Standard DQN"]

    # V2V Success Rate:
    sns.lineplot(data=df, x="speed_kmh", y="v2v_success_rate", hue="model", style="model", markers=True, markersize=10,
                 linewidth=2.5, ax=axes[0], hue_order=hue_order)
    axes[0].set_title("V2V Success Rate vs. Vehicle Speed", fontsize=14)
    axes[0].set_ylabel("V2V Success Rate", fontsize=12)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].legend(title="Model", fontsize=11)

    # V2I Sum Capacity:
    sns.lineplot(data=df, x="speed_kmh", y="v2i_sum_capacity_mbps", hue="model", style="model", markers=True,
                 markersize=10, linewidth=2.5, ax=axes[1], hue_order=hue_order)
    axes[1].set_title("V2I Sum Capacity vs. Vehicle Speed", fontsize=14)
    axes[1].set_ylabel("V2I Sum Capacity (Mbps)", fontsize=12)
    axes[1].legend(title="Model", fontsize=11)

    # P95 Latency:
    sns.lineplot(data=df, x="speed_kmh", y="p95_delay_ms", hue="model", style="model", markers=True, markersize=10,
                 linewidth=2.5, ax=axes[2], hue_order=hue_order)
    axes[2].set_title("V2V P95 Latency vs. Vehicle Speed", fontsize=14)
    axes[2].set_ylabel("P95 Delay (ms)", fontsize=12)
    axes[2].legend(title="Model", fontsize=11)

    for ax in axes: ax.set_xlabel("Vehicle Speed (km/h)", fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, PLOT_FILE_ROBUSTNESS), dpi=300, bbox_inches='tight')
    print(f"--- Final 5-way robustness plot saved to {PLOT_FILE_ROBUSTNESS} ---")
    plt.close()


def plot_policy_comparison():
    print(f"--- Loading FINAL 5-Way Policy Data ---")

    try:
        df = load_and_combine_data(robustness=False)
    except FileNotFoundError as e:
        print(f"!!! Error: {e}")
        return

    # 数据处理
    snr_bins = [-100, 0, 5, 10, 15, 100]
    snr_labels = ["< 0 dB", "0-5 dB", "5-10 dB", "10-15 dB", "> 15 dB"]
    df['snr_bin'] = pd.cut(df['snr_dB'], bins=snr_bins, labels=snr_labels, right=False)
    df['power_ratio'] = (df['power_level'] + 1) / 10.0

    sns.set_theme(style="whitegrid", palette="viridis")

    # --- 策略图 (1x5 布局) ---
    models_to_plot = ["GNN (P1v13 Stabilized)", "GAT (Ablation 1 Sparse)", "GNN (Baseline Full)", "No-GNN DRL",
                      "Standard DQN"]
    fig, axes = plt.subplots(1, 5, figsize=(30, 7), sharey=True)  # 1x5 布局
    fig.suptitle("Final Policy Analysis (5-Way): Stabilized GAT vs. Baselines", fontsize=16, fontweight='bold')

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
    print(f"--- Final 5-way policy comparison plot saved to {PLOT_FILE_POLICY} ---")
    plt.close()


if __name__ == "__main__":
    plot_robustness()
    plot_policy_comparison()
