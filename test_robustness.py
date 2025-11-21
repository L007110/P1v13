import numpy as np
import torch
import pandas as pd
import os
import time
from logger import global_logger, debug_print, debug, set_debug_mode
from Parameters import *
from Topology import formulate_global_list_dqn, vehicle_movement
from ActionChooser import choose_action, choose_action_from_tensor
from GNNModel import global_gnn_model
from GraphBuilder import global_graph_builder
from Main import move_graph_to_device, calculate_mean_metrics
import Parameters

# 导入指标计算
if USE_UMI_NLOS_MODEL:
    from ChannelModel import global_channel_model
    from NewRewardCalculator import new_reward_calculator
else:
    pass


def test_robustness():
    """
    泛化性与鲁棒性测试 (添加 P95 延迟)
    """
    debug_print("========== STARTING ROBUSTNESS TEST MODE ==========")
    set_debug_mode(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_scenarios = {
        "GNN-DRL": {
            "model_path": MODEL_PATH_GNN,
            "use_gnn": True
        },
        "No-GNN DRL": {
            "model_path": MODEL_PATH_NO_GNN,
            "use_gnn": False
        },
        "Standard DQN": {
            "model_path": MODEL_PATH_DQN,
            "use_gnn": False
        }
    }

    results = []
    policy_data = []

    for model_name, config in test_scenarios.items():
        debug_print(f"--- Testing Model: {model_name} ---")

        Parameters.USE_GNN_ENHANCEMENT = config["use_gnn"]
        if model_name == "Standard DQN":
            Parameters.USE_DUELING_DQN = False
        else:
            Parameters.USE_DUELING_DQN = True

        is_gnn_model = Parameters.USE_GNN_ENHANCEMENT
        debug_print(f"Model is GNN: {is_gnn_model}, Dueling: {Parameters.USE_DUELING_DQN}")

        formulate_global_list_dqn(global_dqn_list, device)

        formulate_global_list_dqn(global_dqn_list, device)
        try:
            is_gnn_model = config["use_gnn"]
            if is_gnn_model:
                checkpoint = torch.load(config["model_path"], map_location=device)
                global_gnn_model.load_state_dict(checkpoint)
                global_gnn_model.eval()
                for dqn in global_dqn_list:
                    dqn.eval()
            else:
                checkpoint = torch.load(config["model_path"], map_location=device)
                for dqn in global_dqn_list:
                    dqn.load_state_dict(checkpoint[f'dqn_{dqn.dqn_id}'])
                    dqn.eval()
            debug_print(f"Successfully loaded model from {config['model_path']}")
        except Exception as e:
            debug_print(f"!!! Error loading model {config['model_path']}: {e}")
            continue

        for speed_kmh in TEST_SPEEDS_KMH:
            debug_print(f"  Testing at {speed_kmh} km/h (Vehicle Count: {ROBUSTNESS_FIXED_VEHICLE_COUNT})...")

            epoch_v2v_success_rates = []
            epoch_p95_delays = []
            epoch_v2i_capacities = []
            epoch_decision_times = []

            global_vehicle_id = 0
            overall_vehicle_list = []

            for i_episode in range(ROBUSTNESS_EPISODES_PER_SETTING):
                global_vehicle_id, overall_vehicle_list = vehicle_movement(
                    global_vehicle_id,
                    overall_vehicle_list,
                    target_count=ROBUSTNESS_FIXED_VEHICLE_COUNT,
                    speed_kmh=speed_kmh
                )

                active_v2v_interferers = []
                all_q_values_gnn = None

                if is_gnn_model:
                    try:
                        start_time = time.time()
                        graph_data = global_graph_builder.build_dynamic_graph(global_dqn_list, overall_vehicle_list,
                                                                              epoch=i_episode)
                        graph_data = move_graph_to_device(graph_data, device)
                        with torch.no_grad():
                            # --- MODIFIED: 确保只接收 Q 值张量 (忽略 logits) ---
                            all_q_values_gnn, _ = global_gnn_model(graph_data)  # <-- ADDED , _
                        end_time = time.time()
                        epoch_decision_times.append((end_time - start_time) * 1000.0)
                    except Exception as e:
                        debug(f"!!! GNN test forward pass failed: {e}")
                        all_q_values_gnn = None

                # B. 动作选择
                for dqn in global_dqn_list:
                    dqn.vehicle_exist_curr = False
                    base_state = []
                    dqn.vehicle_in_dqn_range_by_distance = []

                    for vehicle in overall_vehicle_list:
                        if (dqn.start[0] <= vehicle.curr_loc[0] <= dqn.end[0] and
                                dqn.start[1] <= vehicle.curr_loc[1] <= dqn.end[1]):
                            dqn.vehicle_exist_curr = True
                            vehicle.distance_to_bs = new_reward_calculator.channel_model.calculate_3d_distance(
                                (dqn.bs_loc[0], dqn.bs_loc[1]), vehicle.curr_loc)
                            dqn.vehicle_in_dqn_range_by_distance.append(vehicle)

                    dqn.vehicle_in_dqn_range_by_distance.sort(key=lambda x: x.distance_to_bs, reverse=False)

                    if dqn.vehicle_exist_curr:
                        iState = 0
                        for iVehicle in range(min(RL_N_STATES_BASE // 4, len(dqn.vehicle_in_dqn_range_by_distance))):
                            base_state.append(dqn.vehicle_in_dqn_range_by_distance[iVehicle].curr_loc[0])
                            base_state.append(dqn.vehicle_in_dqn_range_by_distance[iVehicle].curr_loc[1])
                            base_state.append(dqn.vehicle_in_dqn_range_by_distance[iVehicle].curr_dir[0])
                            base_state.append(dqn.vehicle_in_dqn_range_by_distance[iVehicle].curr_dir[1])
                            iState += 4
                        if len(base_state) < RL_N_STATES_BASE:
                            base_state.extend([0.0] * (RL_N_STATES_BASE - len(base_state)))
                        else:
                            base_state = base_state[:RL_N_STATES_BASE]
                        if USE_UMI_NLOS_MODEL and hasattr(dqn, 'update_csi_states'):
                            dqn.update_csi_states(dqn.vehicle_in_dqn_range_by_distance, is_current=True)
                        if not hasattr(dqn, 'prev_v2i_interference'):
                            dqn.prev_v2i_interference = 0.0

                        v2i_state = [dqn.prev_v2i_interference]
                        dqn.curr_state = base_state + dqn.csi_states_curr + v2i_state
                        dqn.epsilon = 0.0

                        if is_gnn_model and all_q_values_gnn is not None:
                            actions_tensor = all_q_values_gnn[dqn.dqn_id - 1]
                            if isinstance(actions_tensor, tuple):
                                actions_tensor = actions_tensor[0]

                            choose_action_from_tensor(dqn, actions_tensor, RL_ACTION_SPACE, device)
                        else:
                            start_time_no_gnn = time.time()
                            choose_action(dqn, RL_ACTION_SPACE, device)
                            end_time_no_gnn = time.time()
                            if not is_gnn_model:
                                epoch_decision_times.append((end_time_no_gnn - start_time_no_gnn) * 1000.0)

                        if dqn.action is not None and USE_UMI_NLOS_MODEL:
                            beam_count = dqn.action[0] + 1
                            horizontal_dir = dqn.action[1]
                            vertical_dir = dqn.action[2]
                            power_ratio = (dqn.action[3] + 1) / 10.0
                            directional_gain = new_reward_calculator._calculate_directional_gain(horizontal_dir,
                                                                                                 vertical_dir)
                            total_power_W = TRANSMITTDE_POWER * power_ratio * beam_count * directional_gain
                            active_v2v_interferers.append({
                                'tx_pos': (dqn.bs_loc[0], dqn.bs_loc[1]),
                                'power_W': total_power_W
                            })

                    if dqn.vehicle_exist_curr and dqn.action is not None:
                        closest_vehicle = dqn.vehicle_in_dqn_range_by_distance[0]
                        distance_3d = closest_vehicle.distance_to_bs
                        beam_count = dqn.action[0] + 1
                        horizontal_dir = dqn.action[1]
                        vertical_dir = dqn.action[2]
                        power_ratio = (dqn.action[3] + 1) / 10.0
                        directional_gain = new_reward_calculator._calculate_directional_gain(horizontal_dir,
                                                                                             vertical_dir)
                        total_power = TRANSMITTDE_POWER * power_ratio * beam_count * directional_gain
                        snr_db, snr_linear, _ = new_reward_calculator.channel_model.calculate_snr(
                            total_power, distance_3d, bandwidth=V2V_CHANNEL_BANDWIDTH)
                        delay = new_reward_calculator.calculate_delay(distance_3d, dqn.action, directional_gain,
                                                                      snr_linear)
                        new_reward_calculator._record_communication_metrics(dqn, delay, snr_db)

                        if speed_kmh == 60 or speed_kmh == 120:
                            policy_data.append({
                                "model": model_name,
                                "speed_kmh": speed_kmh,
                                "snr_dB": snr_db,
                                "power_level": dqn.action[3]
                            })

            # D. V2I 和容量计算
            total_v2i_capacity_bps = 0.0
            for link in V2I_LINK_POSITIONS:
                v2i_tx_pos = link['tx']
                v2i_rx_pos = link['rx']
                v2i_dist = global_channel_model.calculate_3d_distance(v2i_tx_pos, v2i_rx_pos)
                _, _, v2i_signal_power_W = global_channel_model.calculate_snr(
                    V2I_TX_POWER, v2i_dist, bandwidth=SYSTEM_BANDWIDTH)
                total_interference_W = 0.0
                for interferer in active_v2v_interferers:
                    interf_dist = global_channel_model.calculate_3d_distance(interferer['tx_pos'], v2i_rx_pos)
                    pl_db, _, _ = global_channel_model.calculate_path_loss(interf_dist)
                    pl_linear = 10 ** (-pl_db / 10)
                    total_interference_W += interferer['power_W'] * pl_linear
                noise_power_W = global_channel_model._calculate_noise_power(SYSTEM_BANDWIDTH)
                v2i_sinr_linear = v2i_signal_power_W / (total_interference_W + noise_power_W)
                v2i_capacity_bps = SYSTEM_BANDWIDTH * np.log2(1 + v2i_sinr_linear)
                total_v2i_capacity_bps += v2i_capacity_bps
            v2i_sum_capacity_mbps = total_v2i_capacity_bps / 1e6

            # E. 收集当前轮次的指标
            mean_delay, p95_delay, mean_snr_db, v2v_success_rate, _, _ = calculate_mean_metrics(global_dqn_list)
            epoch_v2v_success_rates.append(v2v_success_rate)
            epoch_p95_delays.append(p95_delay)
            epoch_v2i_capacities.append(v2i_sum_capacity_mbps)


            for dqn in global_dqn_list:
                dqn.delay_list, dqn.snr_list, dqn.v2v_success_list = [], [], []
                dqn.v2v_delay_ok_list, dqn.v2v_snr_ok_list = [], []

            # 6. 计算平均值并存储
            avg_v2v_success = np.mean(epoch_v2v_success_rates)
            avg_p95_delay = np.mean(epoch_p95_delays) # <--- 新增
            avg_v2i_capacity = np.mean(epoch_v2i_capacities)
            avg_decision_time_per_agent = np.mean(epoch_decision_times) if epoch_decision_times else 0.0

            results.append({
                "model": model_name,
                "speed_kmh": speed_kmh,
                "v2v_success_rate": avg_v2v_success,
                "p95_delay_ms": avg_p95_delay * 1000,
                "v2i_sum_capacity_mbps": avg_v2i_capacity,
                "decision_time_ms": avg_decision_time_per_agent
            })

            debug_print(f"  Result for {speed_kmh} km/h: "
                        f"V2V Success: {avg_v2v_success:.2%}, "
                        f"P95 Delay: {avg_p95_delay * 1000:.4f} ms, " 
                        f"V2I Capacity: {avg_v2i_capacity:.2f} Mbps, "
                        f"Decision Time/Agent: {avg_decision_time_per_agent:.4f} ms")

    # 7. 保存结果到 CSV
    results_df = pd.DataFrame(results)
    results_csv_path = f"{global_logger.log_dir}/robustness_GAT_Ablation3_Stabilized.csv" # <-- MODIFIED
    results_df.to_csv(results_csv_path, index=False)
    debug_print(f"========== ROBUSTNESS TEST COMPLETE ==========")
    debug_print(f"Results saved to {results_csv_path}")

    # 8. 保存策略数据
    policy_df = pd.DataFrame(policy_data)
    policy_csv_path = f"{global_logger.log_dir}/policy_GAT_Ablation3_Stabilized.csv" # <-- MODIFIED
    policy_df.to_csv(policy_csv_path, index=False)
    debug_print(f"Policy analysis data saved to {policy_csv_path}")


if __name__ == "__main__":
    test_robustness()