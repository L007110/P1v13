# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F
import time
import pandas as pd
from ActionChooser import choose_action, choose_action_from_tensor
from logger import global_logger, debug_print, debug, set_debug_mode
from Parameters import *
from Topology import formulate_global_list_dqn, vehicle_movement
from Classes import Vehicle
from Parameters import USE_PRIORITY_REPLAY, PER_BATCH_SIZE
from Parameters import TARGET_UPDATE_FREQUENCY
from Parameters import (
    N_V2I_LINKS, V2I_TX_POWER, V2I_LINK_POSITIONS, SYSTEM_BANDWIDTH,
    TRANSMITTDE_POWER, USE_UMI_NLOS_MODEL,
    RL_N_STATES_BASE, RL_N_STATES_CSI
)
from GraphBuilder import global_graph_builder
from GNNModel import (
    global_gnn_model, global_target_gnn_model,
    update_target_gnn, update_target_gnn_soft
)
from GNNReplayBuffer import GNNReplayBuffer
from Parameters import (
    GNN_REPLAY_CAPACITY, GNN_BATCH_SIZE,
    GNN_TRAIN_START_SIZE, GNN_SOFT_UPDATE_TAU
)
import torch.optim as optim
import Parameters

def move_graph_to_device(graph_data, device):
    """辅助函数：将图数据字典移动到指定设备"""
    try:
        graph_data['node_features']['features'] = graph_data['node_features']['features'].to(device)
        graph_data['node_features']['types'] = graph_data['node_features']['types'].to(device)

        for edge_type in global_gnn_model.edge_types:
            if graph_data['edge_features'][edge_type] is not None:
                graph_data['edge_features'][edge_type]['edge_index'] = \
                    graph_data['edge_features'][edge_type]['edge_index'].to(device)
                graph_data['edge_features'][edge_type]['edge_attr'] = \
                    graph_data['edge_features'][edge_type]['edge_attr'].to(device)
    except Exception as e:
        debug(f"Error moving graph to device: {e}")
    return graph_data


if USE_UMI_NLOS_MODEL:
    from ChannelModel import global_channel_model
    from NewRewardCalculator import new_reward_calculator

    debug_print("Main.py: Using NewRewardCalculator with UMi NLOS model")
else:

    debug_print("Main.py: Using original RewardCalculator")

def calculate_mean_metrics(dqn_list):
    """安全计算平均指标 (包含 P95 延迟)"""
    delays = []
    snrs = []
    v2v_successes = []

    # 双因子列表
    v2v_delay_ok = []
    v2v_snr_ok = []

    debug("=== Calculating Mean Metrics ===")

    for dqn in dqn_list:
        if hasattr(dqn, 'delay_list') and dqn.delay_list:
            valid_delays = [d for d in dqn.delay_list
                            if d is not None and not np.isnan(d) and d > 0]
            if valid_delays:
                recent_delays = valid_delays[-min(20, len(valid_delays)):]  # <--- 窗口可以调大
                delays.extend(recent_delays)

        if hasattr(dqn, 'snr_list') and dqn.snr_list:
            valid_snrs = [s for s in dqn.snr_list
                          if s is not None and not np.isnan(s) and not np.isinf(s)]
            if valid_snrs:
                recent_snrs = valid_snrs[-min(20, len(valid_snrs)):]
                snrs.extend(recent_snrs)

        # 双因子成功率
        if hasattr(dqn, 'v2v_success_list') and dqn.v2v_success_list:
            recent_successes = dqn.v2v_success_list[-min(20, len(dqn.v2v_success_list)):]
            v2v_successes.extend(recent_successes)

        # 诊断列表
        if hasattr(dqn, 'v2v_delay_ok_list') and dqn.v2v_delay_ok_list:
            v2v_delay_ok.extend(dqn.v2v_delay_ok_list[-min(20, len(dqn.v2v_delay_ok_list)):])
        if hasattr(dqn, 'v2v_snr_ok_list') and dqn.v2v_snr_ok_list:
            v2v_snr_ok.extend(dqn.v2v_snr_ok_list[-min(20, len(dqn.v2v_snr_ok_list)):])

    # 计算所有指标
    mean_delay = np.mean(delays) if delays else 1.0
    # P95 延迟计算
    p95_delay = np.percentile(delays, 95) if delays else 1.0

    mean_snr_linear = np.mean(snrs) if snrs else 1.0
    if mean_snr_linear > 0:
        mean_snr_db = 10 * np.log10(mean_snr_linear)
    else:
        mean_snr_db = -100

    v2v_success_rate = np.mean(v2v_successes) if v2v_successes else 0.0

    # 诊断指标
    v2v_delay_only_rate = np.mean(v2v_delay_ok) if v2v_delay_ok else 0.0
    v2v_snr_only_rate = np.mean(v2v_snr_ok) if v2v_snr_ok else 0.0

    debug(f"=== Mean Metrics Summary ===")
    debug(f"Final mean_delay: {mean_delay:.6f}s")
    debug(f"Final p95_delay: {p95_delay:.6f}s")
    debug(f"Final mean_snr_db: {mean_snr_db:.2f}dB")
    debug(f"Final v2v_success_rate: {v2v_success_rate:.3f}")

    return mean_delay, p95_delay, mean_snr_db, v2v_success_rate, v2v_delay_only_rate, v2v_snr_only_rate



def initialize_enhanced_training():
    """
    初始化增强训练组件
    """
    from PriorityReplayBuffer import initialize_global_per
    from Parameters import USE_PRIORITY_REPLAY, PER_CAPACITY

    if USE_PRIORITY_REPLAY:
        global_per_buffer = initialize_global_per(PER_CAPACITY)
        from logger import debug_print
        debug_print("Priority Experience Replay initialized")
        return global_per_buffer
    else:
        from logger import debug_print
        debug_print("Using standard experience replay")
        return None


def enhanced_training_step(dqn, per_buffer, device):
    """
    PER增强训练步骤 - 使用目标网络
    """
    try:
        batch, indices, weights = per_buffer.sample(PER_BATCH_SIZE)
        if batch is None:
            traditional_training_step(dqn, device)  # Fallback
            return

        rewards = torch.FloatTensor([exp.reward for exp in batch]).to(device)
        states = torch.FloatTensor(np.array([exp.state for exp in batch])).to(device)
        actions = torch.LongTensor([exp.action for exp in batch]).to(device)
        next_states = torch.FloatTensor(np.array([exp.next_state for exp in batch])).to(device)
        weights = torch.FloatTensor(weights).to(device)

        # --- 使用目标网络计算目标 Q 值 ---
        with torch.no_grad():
            # 1. 使用主网络选择下一状态的最佳动作
            next_q_values_online = dqn(next_states)
            best_action_indices = next_q_values_online.argmax(dim=1, keepdim=True)

            # 2. 使用目标网络评估这些动作的 Q 值
            next_q_values_target = dqn.target_network(next_states)
            # 从目标网络中选择被主网络选为最佳动作的 Q 值
            next_q_for_target = next_q_values_target.gather(1, best_action_indices).squeeze(1)

            # 3. 计算 Double DQN 目标
            target_q_values = rewards + RL_GAMMA * next_q_for_target  # (如果需要处理 done 状态: * (1 - dones))

        # 使用主网络计算当前 Q 值
        current_q_values = dqn(states)
        current_action_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 计算 TD 误差（用于优先级更新）
        td_errors = (target_q_values - current_action_q_values).abs().detach().cpu().numpy()

        # 计算损失（应用重要性采样权重）
        dqn.loss = (weights * torch.nn.functional.mse_loss(current_action_q_values, target_q_values.detach(),
                                                           reduction='none')).mean()

        # 更新PER优先级
        per_buffer.update_priorities(indices, td_errors)

        # 反向传播
        dqn.optimizer.zero_grad()
        dqn.loss.backward()
        torch.nn.utils.clip_grad_norm_(dqn.parameters(), max_norm=1.0)
        dqn.optimizer.step()

        # 添加 Epsilon 衰减
        if not FLAG_ADAPTIVE_EPSILON_ADJUSTMENT and dqn.epsilon > RL_EPSILON_MIN:  # 检查是否禁用自适应且大于最小值
            dqn.epsilon *= RL_EPSILON_DECAY

        debug(f"Enhanced training - DQN {dqn.dqn_id}: Loss {dqn.loss.item():.4f}")

    except Exception as e:
        debug(f"Error in enhanced training step (DDQN): {e}")
        import traceback
        traceback.print_exc()
        traditional_training_step(dqn, device)  # Fallback


def traditional_training_step(dqn, device):
    try:
        curr_state_tensor = torch.tensor(dqn.curr_state).float().to(device)  # 确保是 float
        next_state_tensor = torch.tensor(dqn.next_state).float().to(device)  # 确保是 float

        if curr_state_tensor.dim() == 1:
            curr_state_tensor = curr_state_tensor.unsqueeze(0)
        if next_state_tensor.dim() == 1:
            next_state_tensor = next_state_tensor.unsqueeze(0)

        # Double DQN 目标 Q 值计算
        with torch.no_grad():
            # 1. 使用主网络选择下一状态的最佳动作
            next_q_values_online = dqn(next_state_tensor)  # 输出应为 [1, num_actions]

            if next_q_values_online.dim() == 2:
                best_action_indices = next_q_values_online.argmax(dim=1, keepdim=True)
            elif next_q_values_online.dim() == 1:
                best_action_indices = next_q_values_online.argmax(dim=0, keepdim=True).unsqueeze(0)
            else:
                raise ValueError(f"Unexpected shape for next_q_values_online: {next_q_values_online.shape}")

            # 2. 使用目标网络评估这些动作的 Q 值
            next_q_values_target = dqn.target_network(next_state_tensor)
            if next_q_values_target.dim() == 1:
                next_q_values_target = next_q_values_target.unsqueeze(0)
            next_q_for_target = next_q_values_target.gather(1, best_action_indices).squeeze()

            # 确保 q_target 是 Float
            reward_tensor = torch.tensor(dqn.reward, dtype=torch.float32, device=device)
            dqn.q_target = reward_tensor + RL_GAMMA * next_q_for_target

        # 使用主网络计算当前 Q 值
        curr_q_values = dqn(curr_state_tensor)
        if curr_q_values.dim() == 1:
            curr_q_values = curr_q_values.unsqueeze(0)
        action_index = RL_ACTION_SPACE.index(dqn.action) if dqn.action in RL_ACTION_SPACE else 0
        action_index_tensor = torch.tensor([[action_index]], dtype=torch.long, device=device)
        dqn.q_estimate = curr_q_values.gather(1, action_index_tensor).squeeze()  # 应该是 Float

        # 计算 Loss
        dqn.loss = torch.nn.functional.mse_loss(dqn.q_estimate, dqn.q_target.detach())

        # 反向传播
        dqn.optimizer.zero_grad()
        dqn.loss.backward()
        dqn.optimizer.step()

        # 添加 Epsilon 衰减
        if not FLAG_ADAPTIVE_EPSILON_ADJUSTMENT and dqn.epsilon > RL_EPSILON_MIN:  # 检查是否禁用自适应且大于最小值
            dqn.epsilon *= RL_EPSILON_DECAY

        debug(
            f"Traditional training - DQN {dqn.dqn_id}: Loss {dqn.loss.item():.4f}, Epsilon: {dqn.epsilon:.4f}")

    except Exception as e:
        debug(f"Error in traditional training step (DDQN): {e}")
        import traceback
        traceback.print_exc()
        dqn.loss = torch.tensor(1.0, requires_grad=True, device=device)


# 添加 P95 延迟
def rl(mean_loss_across_epochs=None, gnn_optimizer=None):
    epoch = 1
    global_vehicle_id = 0
    overall_vehicle_list = []

    global_per_buffer = None
    global_gnn_buffer = None

    if USE_PRIORITY_REPLAY:
        global_per_buffer = initialize_enhanced_training()

    if USE_GNN_ENHANCEMENT:
        debug_print("Starting GNN-DRL training (Dueling-Double-DQN w/ GNN)")
        global_gnn_buffer = GNNReplayBuffer(capacity=GNN_REPLAY_CAPACITY)
    else:
        debug_print("Starting No-GNN training (Dueling-Double-DQN w/ PER)")
        if USE_PRIORITY_REPLAY:
            global_per_buffer = initialize_enhanced_training()

    # 检查DQN的指标列表初始化 (包含可选的诊断列表)
    for dqn in global_dqn_list:
        if not hasattr(dqn, 'delay_list'):
            dqn.delay_list = []
        if not hasattr(dqn, 'snr_list'):
            dqn.snr_list = []
        if not hasattr(dqn, 'v2v_success_list'):
            dqn.v2v_success_list = []
        if not hasattr(dqn, 'v2v_delay_ok_list'):
            dqn.v2v_delay_ok_list = []
        if not hasattr(dqn, 'v2v_snr_ok_list'):
            dqn.v2v_snr_ok_list = []

    graph_data_t = None
    all_q_values_t = None

    prev_mean_loss = 0.0
    while True:
        # ==================================================================
        # 步骤 1: 车辆移动 (环境演进到 t+1)
        # ==================================================================
        global_vehicle_id, overall_vehicle_list = vehicle_movement(global_vehicle_id, overall_vehicle_list)

        loss_list_per_epoch = []
        mean_loss = 0.0
        cumulative_reward_per_epoch = 0.0
        v2i_sum_capacity_mbps = 0.0

        if len(loss_list_per_epoch) > 0 and mean_loss_across_epochs is not None and len(mean_loss_across_epochs) > 10:
            debug_print(
                f"Epoch {epoch} Prev mean loss {mean_loss} "
                f"Vehicle count {len(overall_vehicle_list)}"
            )
        else:
            debug_print(f"Epoch {epoch}")

        active_v2v_interferers = []

        # ==================================================================
        # 步骤 2: 构建 S_t+1 的图, 并获取 t+1 的 Q 值 (用于训练)
        # ==================================================================
        graph_data_t_plus_1 = None
        all_q_values_t_plus_1_online = None

        if USE_GNN_ENHANCEMENT:
            global_gnn_model.train()
            try:
                graph_data_t_plus_1 = global_graph_builder.build_dynamic_graph(global_dqn_list, overall_vehicle_list,
                                                                               epoch)
            except Exception as e:
                debug(f"GNN S_t+1 graph build/forward pass failed: {e}")

        # ==================================================================
        # 步骤 3: GNN 训练 (使用 S_t, A_t, R_t, S_t+1)
        # ==================================================================

        if (USE_GNN_ENHANCEMENT and
                global_gnn_buffer is not None and
                len(global_gnn_buffer) >= GNN_TRAIN_START_SIZE):

            batch = global_gnn_buffer.sample(GNN_BATCH_SIZE, device)
            if batch is None:
                continue

            total_gnn_loss = torch.tensor(0.0, device=device)
            agents_trained = 0
            entropy_loss = torch.tensor(0.0, device=device)

            for experience in batch:
                graph_t_dev, actions_t, rewards_t, graph_t1_dev = experience

                # --- MODIFIED 1: 调用 forward 并获取 logits (不再需要 torch.no_grad()) ---
                # 移除 torch.no_grad() 意味着 GNN 模型现在是 Joint Training
                all_q_values_t, edge_attention_logits_t = global_gnn_model(graph_t_dev)
                with torch.no_grad():
                    all_q_values_t1_online, _ = global_gnn_model(graph_t1_dev)
                    all_q_values_t1_target, _ = global_target_gnn_model(graph_t1_dev)
                # --- END MODIFIED 1 ---

                # --- ADDED 2: 计算 Attention Entropy ---
                P = F.softmax(edge_attention_logits_t, dim=0)
                # 熵 H = - sum(P * log(P))
                entropy = -torch.sum(P * torch.log(P + 1e-9))
                entropy_loss += entropy
                # --- END ADDED 2 ---

                for dqn_id_str, action_index in actions_t.items():
                    try:
                        dqn_id = int(dqn_id_str)
                        dqn_id_index = dqn_id - 1
                        q_estimate = all_q_values_t[dqn_id_index, action_index]

                        with torch.no_grad():
                            reward = rewards_t[dqn_id_str]
                            best_action_t1 = torch.argmax(all_q_values_t1_online[dqn_id_index])
                            q_target_next = all_q_values_t1_target[dqn_id_index, best_action_t1]
                            q_target = reward + RL_GAMMA * q_target_next

                        loss = torch.nn.functional.mse_loss(q_estimate, q_target.detach())
                        total_gnn_loss += loss
                        agents_trained += 1

                        for dqn in global_dqn_list:
                            if dqn.dqn_id == dqn_id:
                                dqn.loss = loss.item()
                                loss_list_per_epoch.append(dqn.loss)
                                break
                    except Exception as e:
                        debug(f"Error processing agent {dqn_id} in GNN batch: {e}")

            if agents_trained > 0:
                gnn_optimizer.zero_grad()
                mean_batch_loss_td = total_gnn_loss / agents_trained

                # --- Entropy 正则化逻辑 ---
                mean_entropy = entropy_loss / GNN_BATCH_SIZE

                final_loss = mean_batch_loss_td - LAMBDA_ENTROPY * mean_entropy

                final_loss.backward()
                torch.nn.utils.clip_grad_norm_(global_gnn_model.parameters(), max_norm=1.0)
                gnn_optimizer.step()

                update_target_gnn_soft(GNN_SOFT_UPDATE_TAU)

                for dqn in global_dqn_list:
                    if not FLAG_ADAPTIVE_EPSILON_ADJUSTMENT and dqn.epsilon > RL_EPSILON_MIN:
                        dqn.epsilon *= RL_EPSILON_DECAY

        # ==================================================================
        # 步骤 4: 循环 1 (t+1) -> 动作 A_t+1
        # ==================================================================

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

                if USE_GNN_ENHANCEMENT:
                    try:
                        global_gnn_model.eval()  # 设为评估模式

                        # 1. (新) 为这个DQN构建 *局部* 子图
                        #    (graph_data_t_plus_1 对应当前车辆位置)
                        graph_data_local = global_graph_builder.build_spatial_subgraph(
                            dqn, global_dqn_list, overall_vehicle_list, epoch
                        )
                        graph_data_local = move_graph_to_device(graph_data_local, device)

                        # 2. (新) 只为这个DQN运行GNN推理
                        with torch.no_grad():
                            actions_tensor = global_gnn_model(graph_data_local, dqn_id=dqn.dqn_id)

                        global_gnn_model.train()  # 恢复训练模式

                        # 3. (旧) 选择动作
                        choose_action_from_tensor(dqn, actions_tensor, RL_ACTION_SPACE, device)

                    except Exception as e:
                        debug(f"!!! GNN action selection for DQN {dqn.dqn_id} failed: {e}")
                        global_gnn_model.train()  # 确保恢复训练模式
                        choose_action(dqn, RL_ACTION_SPACE, device)  # 回退

                else:
                    # No-GNN 模式 (不变)
                    choose_action(dqn, RL_ACTION_SPACE, device)

                if dqn.action is not None and USE_UMI_NLOS_MODEL:
                    beam_count = dqn.action[0] + 1
                    horizontal_dir = dqn.action[1]
                    vertical_dir = dqn.action[2]
                    power_ratio = (dqn.action[3] + 1) / 10.0
                    directional_gain = new_reward_calculator._calculate_directional_gain(horizontal_dir, vertical_dir)
                    total_power_W = TRANSMITTDE_POWER * power_ratio * beam_count * directional_gain
                    active_v2v_interferers.append({
                        'tx_pos': (dqn.bs_loc[0], dqn.bs_loc[1]),
                        'power_W': total_power_W
                    })
                else:
                    dqn.action = None
            else:
                dqn.curr_state = [0.0] * RL_N_STATES
                dqn.action = None

        # ==================================================================
        # 步骤 5: V2I 容量计算
        # ==================================================================
        total_v2i_capacity_bps = 0.0
        if USE_UMI_NLOS_MODEL:
            for link in V2I_LINK_POSITIONS:
                v2i_tx_pos = link['tx']
                v2i_rx_pos = link['rx']
                v2i_dist = global_channel_model.calculate_3d_distance(v2i_tx_pos, v2i_rx_pos)
                _, _, v2i_signal_power_W = global_channel_model.calculate_snr(
                    V2I_TX_POWER, v2i_dist, bandwidth=SYSTEM_BANDWIDTH)
                total_interference_W = 0.0
                for interferer in active_v2v_interferers:
                    interferer_pos = interferer['tx_pos']
                    interferer_power_W = interferer['power_W']
                    interf_dist = global_channel_model.calculate_3d_distance(interferer_pos, v2i_rx_pos)
                    pl_db, _, _ = global_channel_model.calculate_path_loss(interf_dist)
                    pl_linear = 10 ** (-pl_db / 10)
                    total_interference_W += interferer_power_W * pl_linear
                noise_power_W = global_channel_model._calculate_noise_power(SYSTEM_BANDWIDTH)
                v2i_sinr_linear = v2i_signal_power_W / (total_interference_W + noise_power_W)
                v2i_capacity_bps = SYSTEM_BANDWIDTH * np.log2(1 + v2i_sinr_linear)
                total_v2i_capacity_bps += v2i_capacity_bps
            v2i_sum_capacity_mbps = total_v2i_capacity_bps / 1e6

        # ==================================================================
        # 步骤 6: 循环 2 (t+1) -> 奖励 R_t+1 -> 存储经验
        # ==================================================================
        current_actions_t = {}
        current_rewards_t = {}

        for dqn in global_dqn_list:
            dqn.vehicle_exist_next = False
            base_state_next = []

            if not USE_GNN_ENHANCEMENT:
                dqn.vehicle_in_dqn_range_by_distance = []
                for vehicle in overall_vehicle_list:
                    if (dqn.start[0] <= vehicle.curr_loc[0] <= dqn.end[0] and
                            dqn.start[1] <= vehicle.curr_loc[1] <= dqn.end[1]):
                        dqn.vehicle_exist_next = True
                        vehicle.distance_to_bs = new_reward_calculator.channel_model.calculate_3d_distance(
                            (dqn.bs_loc[0], dqn.bs_loc[1]), vehicle.curr_loc)
                        dqn.vehicle_in_dqn_range_by_distance.append(vehicle)
                dqn.vehicle_in_dqn_range_by_distance.sort(key=lambda x: x.distance_to_bs, reverse=False)

            if dqn.vehicle_exist_curr:
                if USE_GNN_ENHANCEMENT:
                    dqn.reward = new_reward_calculator.calculate_complete_reward(
                        dqn, dqn.vehicle_in_dqn_range_by_distance, dqn.action)
                    cumulative_reward_per_epoch += dqn.reward
                    if dqn.action is not None:
                        action_index = RL_ACTION_SPACE.index(dqn.action)
                        current_actions_t[str(dqn.dqn_id)] = action_index
                        current_rewards_t[str(dqn.dqn_id)] = dqn.reward
                elif dqn.vehicle_exist_next:
                    iState = 0
                    for iVehicle in range(min(RL_N_STATES_BASE // 4, len(dqn.vehicle_in_dqn_range_by_distance))):
                        base_state_next.append(dqn.vehicle_in_dqn_range_by_distance[iVehicle].curr_loc[0])
                        base_state_next.append(dqn.vehicle_in_dqn_range_by_distance[iVehicle].curr_loc[1])
                        base_state_next.append(dqn.vehicle_in_dqn_range_by_distance[iVehicle].curr_dir[0])
                        base_state_next.append(dqn.vehicle_in_dqn_range_by_distance[iVehicle].curr_dir[1])
                        iState += 4
                    if len(base_state_next) < RL_N_STATES_BASE:
                        base_state_next.extend([0.0] * (RL_N_STATES_BASE - len(base_state_next)))
                    else:
                        base_state_next = base_state_next[:RL_N_STATES_BASE]

                    if USE_UMI_NLOS_MODEL and hasattr(dqn, 'update_csi_states'):
                        dqn.update_csi_states(dqn.vehicle_in_dqn_range_by_distance, is_current=False)
                    v2i_interference_penalty_linear = 0.0
                    if dqn.action is not None:
                        beam_count = dqn.action[0] + 1
                        horizontal_dir = dqn.action[1]
                        vertical_dir = dqn.action[2]
                        power_ratio = (dqn.action[3] + 1) / 10.0
                        directional_gain = new_reward_calculator._calculate_directional_gain(horizontal_dir,
                                                                                             vertical_dir)
                        total_power = TRANSMITTDE_POWER * power_ratio * beam_count * directional_gain

                        agent_tx_pos = (dqn.bs_loc[0], dqn.bs_loc[1])
                        for link in V2I_LINK_POSITIONS:
                            v2i_rx_pos = link['rx']
                            interf_dist = new_reward_calculator.channel_model.calculate_3d_distance(agent_tx_pos,
                                                                                                    v2i_rx_pos)
                            pl_db, _, _ = new_reward_calculator.channel_model.calculate_path_loss(interf_dist)
                            pl_linear = 10 ** (-pl_db / 10)
                            v2i_interference_penalty_linear += total_power * pl_linear

                    if not hasattr(dqn, 'prev_v2i_interference'):
                        dqn.prev_v2i_interference = 0.0

                    v2i_state = [dqn.prev_v2i_interference]
                    dqn.prev_v2i_interference = v2i_interference_penalty_linear  # 存储当前干扰供下一轮使用

                    dqn.next_state = base_state_next + dqn.csi_states_next + v2i_state

                    dqn.reward = new_reward_calculator.calculate_complete_reward(
                        dqn, dqn.vehicle_in_dqn_range_by_distance, dqn.action, active_v2v_interferers
                    )
                    cumulative_reward_per_epoch += dqn.reward

                    if global_per_buffer is not None:
                        action_index = RL_ACTION_SPACE.index(dqn.action) if dqn.action in RL_ACTION_SPACE else 0
                        global_per_buffer.add(
                            state=dqn.curr_state, action=action_index, reward=dqn.reward,
                            next_state=dqn.next_state, done=False
                        )
                    if global_per_buffer is not None and len(global_per_buffer) >= PER_BATCH_SIZE:
                        enhanced_training_step(dqn, global_per_buffer, device)
                    else:
                        traditional_training_step(dqn, device)
                    loss_list_per_epoch.append(dqn.loss.item())
                    dqn.loss_list.append(dqn.loss.item())
                else:
                    dqn.loss = torch.tensor(0.0)
                    dqn.reward = 0.0
                    if not USE_GNN_ENHANCEMENT:
                        new_reward_calculator._record_communication_metrics(dqn, 1.0, -100.0)
            else:
                dqn.loss = torch.tensor(0.0)
                dqn.reward = 0.0
                if not USE_GNN_ENHANCEMENT:
                    new_reward_calculator._record_communication_metrics(dqn, 1.0, -100.0)

        # ==================================================================
        # 步骤 7: 存储 S_t+1 的图，供下一个 Epoch 使用
        # ==================================================================
        if (USE_GNN_ENHANCEMENT and global_gnn_buffer is not None
                and graph_data_t is not None):
            if graph_data_t_plus_1 is not None and current_actions_t:
                global_gnn_buffer.add(
                    graph_t=graph_data_t,
                    actions_t=current_actions_t,
                    rewards_t=current_rewards_t,
                    graph_t1=graph_data_t_plus_1
                )
        if USE_GNN_ENHANCEMENT:
            graph_data_t = graph_data_t_plus_1

        # ==================================================================
        # 步骤 8: 周期末尾的日志记录和维护
        # ==================================================================
        mean_delay, p95_delay, mean_snr_db, v2v_success_rate, v2v_delay_only_rate, v2v_snr_only_rate = calculate_mean_metrics(
            global_dqn_list)


        if len(loss_list_per_epoch) > 0:
            if FLAG_EMA_LOSS:
                EMA_WEIGHT = 0.2
                mean_loss = (np.min(loss_list_per_epoch) * EMA_WEIGHT +
                             np.mean(loss_list_per_epoch) * EMA_WEIGHT * 0.1 +
                             prev_mean_loss * (1 - EMA_WEIGHT) * 0.9)
                prev_mean_loss = mean_loss
            else:
                mean_loss = np.mean(loss_list_per_epoch)

        # 将 p95_delay 传递给 logger
        global_logger.log_epoch(
            epoch=epoch, cumulative_reward=cumulative_reward_per_epoch,
            mean_loss=mean_loss,
            mean_delay=mean_delay,
            p95_delay=p95_delay,
            mean_snr=mean_snr_db,
            vehicle_count=len(overall_vehicle_list),
            v2v_success_rate=v2v_success_rate,
            v2i_sum_capacity=v2i_sum_capacity_mbps,
            # 传递诊断指标
            v2v_delay_only_rate=v2v_delay_only_rate,
            v2v_snr_only_rate=v2v_snr_only_rate
        )


        for dqn in global_dqn_list:
            dqn_metrics = {
                'loss': getattr(dqn, 'loss', 0), 'reward': getattr(dqn, 'reward', 0),
                'epsilon': getattr(dqn, 'epsilon', RL_EPSILON),
                'vehicle_count': len(getattr(dqn, 'vehicle_in_dqn_range_by_distance', [])),
                'snr': getattr(dqn, 'prev_snr', 0), 'delay': getattr(dqn, 'prev_delay', 0)
            }
            global_logger.log_dqn_performance(dqn.dqn_id, dqn_metrics)

        if global_per_buffer is not None:
            per_stats = global_per_buffer.get_statistics()
            global_logger.logger.info(
                f"PER Stats - Buffer: {per_stats['buffer_size']}, "
                f"Avg Priority: {per_stats['avg_priority']:.4f}"
            )

        if not USE_GNN_ENHANCEMENT:
            if epoch % TARGET_UPDATE_FREQUENCY == 0:
                debug_print(f"--- Updating Target Networks (Epoch {epoch}) ---")
                for dqn in global_dqn_list:
                    dqn.update_target_network()

        if epoch == 1500:
            global_logger.log_convergence(epoch, mean_loss)
            debug_print(f"Converged at epoch {epoch} with loss {mean_loss}")

            try:
                if USE_GNN_ENHANCEMENT:
                    # 1. 保存 GNN-DRL 模型
                    model_save_path = MODEL_PATH_GNN
                    save_data = global_gnn_model.state_dict()
                    torch.save(save_data, model_save_path)
                    debug_print(f"GNN-DRL Model saved to {model_save_path}")

                else:
                    # 2. 非 GNN 模型：需要进一步区分
                    if USE_DUELING_DQN:
                        # 2a. 保存 No-GNN DRL (Dueling) 模型
                        model_save_path = MODEL_PATH_NO_GNN
                        save_data = {f'dqn_{dqn.dqn_id}': dqn.state_dict() for dqn in global_dqn_list}
                        torch.save(save_data, model_save_path)
                        debug_print(f"No-GNN DRL Model saved to {model_save_path}")

                    else:
                        # 2b. 保存 Standard DQN (非 Dueling) 模型
                        model_save_path = MODEL_PATH_DQN
                        save_data = {f'dqn_{dqn.dqn_id}': dqn.state_dict() for dqn in global_dqn_list}
                        torch.save(save_data, model_save_path)
                        debug_print(f"Standard DQN Model saved to {model_save_path}")

            except Exception as e:
                debug_print(f"Error saving model: {e}")
            break

        epoch += 1

    global_logger.finalize()


def test():
    """
    可扩展性测试的主函数
    """
    debug_print("========== STARTING SCALABILITY TEST MODE (Static Evaluation by Vehicle Count) ==========")
    set_debug_mode(False)  # 关闭详细日志以加速

    # 1. 定义要测试的模型
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

    results = []  # 存储最终的平均结果

    # 确保 GNN 模型 (如果使用) 在评估模式
    global_gnn_model.to(device)
    global_gnn_model.eval()

    # 2. 循环遍历
    for model_name, config in test_scenarios.items():
        debug_print(f"--- Testing Model: {model_name} ---")

        Parameters.USE_GNN_ENHANCEMENT = config["use_gnn"]
        if model_name == "Standard DQN":
            Parameters.USE_DUELING_DQN = False
        else:
            Parameters.USE_DUELING_DQN = True

        is_gnn_model = Parameters.USE_GNN_ENHANCEMENT
        debug_print(f"Model is GNN: {is_gnn_model}, Dueling: {Parameters.USE_DUELING_DQN}")


        # 3. 创建和加载模型权重
        formulate_global_list_dqn(global_dqn_list, device)
        try:
            if is_gnn_model:
                # 加载 GNN 权重
                checkpoint = torch.load(config["model_path"], map_location=device)
                global_gnn_model.load_state_dict(checkpoint)
                global_gnn_model.eval()
                for dqn in global_dqn_list:
                    dqn.eval()
            else:
                # 加载 No-GNN 权重 (每个 DQN)
                checkpoint = torch.load(config["model_path"], map_location=device)
                for dqn in global_dqn_list:
                    dqn.load_state_dict(checkpoint[f'dqn_{dqn.dqn_id}'])
                    dqn.eval()
            debug_print(f"Successfully loaded model from {config['model_path']}")
        except Exception as e:
            debug_print(f"!!! Error loading model {config['model_path']}: {e}")
            debug_print(f"!!! Skipping test for {model_name}")
            continue

        # 4. [外层循环] 遍历所有要测试的车辆数量 (P1v7 逻辑)
        for vehicle_count in TEST_VEHICLE_COUNTS:
            debug_print(f"  Testing with {vehicle_count} vehicles...")

            # 存储 100 次 episode 的瞬时指标
            episode_v2v_success_rates = []
            episode_p95_delays_ms = []
            episode_v2i_capacities = []
            episode_decision_times = []

            global_vehicle_id = 0
            overall_vehicle_list = []

            # 5. [内层循环] 在每个车辆数下运行 N 个测试轮次 (P1v7 逻辑)
            for i_episode in range(TEST_EPISODES_PER_COUNT):

                # A. 移动/生成车辆 (使用固定的 target_count)
                global_vehicle_id, overall_vehicle_list = vehicle_movement(
                    global_vehicle_id,
                    overall_vehicle_list,
                    target_count=vehicle_count  # <<< 核心：静态车辆数
                )

                active_v2v_interferers = []
                all_q_values_gnn = None
                step_decision_times = []  # 存储这一步所有 dqn 的决策时间


                # C. 动作选择 (无训练)
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
                        # 准备状态 (GNN 和 No-GNN 都需要)
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

                        # 测试模式：关闭探索
                        dqn.epsilon = 0.0

                        if is_gnn_model:
                            try:
                                # 1. (新) 为这个DQN构建 *局部* 子图
                                start_time_gnn = time.time()
                                graph_data_local = global_graph_builder.build_spatial_subgraph(
                                    dqn, global_dqn_list, overall_vehicle_list, i_episode
                                )
                                graph_data_local = move_graph_to_device(graph_data_local, device)

                                # 2. (新) 只为这个DQN运行GNN推理
                                with torch.no_grad():
                                    # --- MODIFIED 5: GNN 推理调用只取 Q 值 ---
                                    actions_tensor, _ = global_gnn_model(graph_data_local, dqn_id=dqn.dqn_id)

                                end_time_gnn = time.time()
                                # (记录 GNN 决策时间)
                                dqn.last_decision_time = (end_time_gnn - start_time_gnn) * 1000.0
                                step_decision_times.append(dqn.last_decision_time)

                                # 3. (旧) 选择动作
                                choose_action_from_tensor(dqn, actions_tensor, RL_ACTION_SPACE, device)

                            except Exception as e:
                                debug(f"!!! GNN test forward pass for DQN {dqn.dqn_id} failed: {e}")
                                # 回退到 No-GNN 决策
                                choose_action(dqn, RL_ACTION_SPACE, device)

                        else:
                            # No-GNN 模式 (不变)
                            choose_action(dqn, RL_ACTION_SPACE, device)
                            if not is_gnn_model:
                                step_decision_times.append(dqn.last_decision_time)

                        # D. 收集干扰信息
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
                    else:
                        dqn.action = None  # 确保没有车的DQN不产生干扰

                    # E. V2V 指标计算 (非奖励)
                    # (移到 dqn.vehicle_exist_curr 内部)
                    if dqn.vehicle_exist_curr and dqn.action is not None:
                        closest_vehicle = dqn.vehicle_in_dqn_range_by_distance[0]
                        distance_3d = closest_vehicle.distance_to_bs

                        # (重新获取动作参数)
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

                        # (记录瞬时指标)
                        new_reward_calculator._record_communication_metrics(dqn, delay, snr_db)

                # F. V2I 和容量计算 (在所有干扰源确定后)
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

                # G. 收集当前轮次(Episode)的平均指标
                # (calculate_mean_metrics 会从 dqn.delay_list 等列表中提取数据)
                mean_delay, p95_delay, mean_snr_db, v2v_success_rate, _, _ = calculate_mean_metrics(global_dqn_list)

                episode_v2v_success_rates.append(v2v_success_rate)
                episode_p95_delays_ms.append(p95_delay * 1000)  # 转换为毫秒
                episode_v2i_capacities.append(v2i_sum_capacity_mbps)

                if step_decision_times:
                    if is_gnn_model:
                        episode_decision_times.append(np.mean(step_decision_times))  # GNN 记录全局时间
                    else:
                        episode_decision_times.append(np.sum(step_decision_times))  # No-GNN 记录所有智能体总和

                # H. 清除DQN列表中的瞬时指标，为下一轮 (i_episode + 1) 做准备
                for dqn in global_dqn_list:
                    dqn.delay_list = []
                    dqn.snr_list = []
                    dqn.v2v_success_list = []
                    dqn.v2v_delay_ok_list = []
                    dqn.v2v_snr_ok_list = []

                if i_episode % 20 == 0:
                    debug_print(f"    ... Episode {i_episode}/{TEST_EPISODES_PER_COUNT}")

            # 6. [内层循环结束] 计算 100 轮的平均值并存储
            avg_v2v_success = np.mean(episode_v2v_success_rates)
            avg_p95_delay_ms = np.mean(episode_p95_delays_ms)  # <--- P95
            avg_v2i_capacity = np.mean(episode_v2i_capacities)
            avg_decision_time = np.mean(episode_decision_times) if episode_decision_times else 0.0

            results.append({
                "model": model_name,
                "vehicle_count": vehicle_count,
                "v2v_success_rate": avg_v2v_success,
                "v2i_sum_capacity_mbps": avg_v2i_capacity,
                "p95_delay_ms": avg_p95_delay_ms,  # <--- 添加P95
                "decision_time_ms": avg_decision_time
            })

            debug_print(f"  Result for {vehicle_count} vehicles: "
                        f"V2V Success: {avg_v2v_success:.2%}, "
                        f"P95 Delay: {avg_p95_delay_ms:.4f} ms, "  # <--- P95
                        f"V2I Capacity: {avg_v2i_capacity:.2f} Mbps, "
                        f"Decision Time: {avg_decision_time:.4f} ms")

    # 7. [外层循环结束] 保存结果到 CSV
    results_df = pd.DataFrame(results)
    # --- MODIFIED: Changed output filename ---
    results_csv_path = f"{global_logger.log_dir}/scalability_GAT_Ablation3_Stabilized.csv" #
    results_df.to_csv(results_csv_path, index=False)
    debug_print(f"========== SCALABILITY TEST COMPLETE ==========")
    debug_print(f"Results saved to {results_csv_path}")


if __name__ == "__main__":
    set_debug_mode(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debug_print(f"device is {device}")
    gnn_optimizer = None
    if USE_GNN_ENHANCEMENT:
        global_gnn_model.to(device)
        global_target_gnn_model.to(device)
        gnn_optimizer = optim.Adam(global_gnn_model.parameters(), lr=RL_ALPHA_GNN)
        debug_print(f"GNN 优化器 (Adam, lr={RL_ALPHA_GNN}) 已创建。")
    # 显示当前使用的模型
    if USE_UMI_NLOS_MODEL:
        debug_print("Using UMi NLOS Channel Model with NewRewardCalculator")
    else:
        debug_print("Using Original Channel Model with RewardCalculator")

    print_parameters()

    if RUN_MODE == "TRAIN":
        debug_print("========== STARTING TRAINING MODE ==========")
        formulate_global_list_dqn(global_dqn_list, device)
        for dqn in global_dqn_list:
            debug_print(dqn)
        rl(gnn_optimizer=gnn_optimizer)

    elif RUN_MODE == "TEST":
        test()

    else:
        debug_print(f"!!! Error: Unknown RUN_MODE '{RUN_MODE}' in Parameters.py. Set to 'TRAIN' or 'TEST'.")
