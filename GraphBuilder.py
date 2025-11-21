# -*- coding: utf-8 -*-
import numpy as np
import torch
from collections import defaultdict
from logger import debug, debug_print, set_debug_mode
from Parameters import *
from ChannelModel import global_channel_model
from Parameters import V2V_CHANNEL_BANDWIDTH, TRANSMITTDE_POWER
from Parameters import GNN_INFERENCE_RADIUS # <-- 导入新参数

class GraphBuilder:
    """
    动态图构建器
    负责构建包含RSU和车辆的异构图，为GNN提供输入
    """

    def __init__(self):
        self.edge_types = ['communication', 'interference', 'proximity']
        self.communication_threshold = 500.0  # 通信距离阈值 (米)
        self.interference_threshold = 300.0  # 干扰距离阈值 (米)
        self.proximity_threshold = 200.0  # 邻近距离阈值 (米)

        # 定义固定的特征维度
        self.rsu_feature_dim = 9  # RSU节点特征维度
        self.vehicle_feature_dim = 6  # 车辆节点特征维度
        self.max_feature_dim = max(self.rsu_feature_dim, self.vehicle_feature_dim)
        #定义边特征的维度
        self.comm_edge_feature_dim = 4
        debug("GraphBuilder initialized with dynamic graph construction")

    def build_dynamic_graph(self, dqn_list, vehicle_list, epoch):
        """
        构建动态异构图

        Args:
            dqn_list: DQN列表 (RSU代理)
            vehicle_list: 车辆列表
            epoch: 当前训练轮次

        Returns:
            graph_data: 图数据结构
        """
        debug(f"Building dynamic graph at epoch {epoch}")

        # 1. 创建节点
        nodes = self._create_nodes(dqn_list, vehicle_list)

        # 2. 创建边
        edges = self._create_edges(nodes, dqn_list, vehicle_list, epoch)

        # 3. 构建图数据
        graph_data = {
            'nodes': nodes,
            'edges': edges,
            'node_features': self._extract_node_features(nodes, dqn_list, vehicle_list),
            'edge_features': self._extract_edge_features(edges, nodes),
            'metadata': {
                'epoch': epoch,
                'num_rsu_nodes': len(dqn_list),
                'num_vehicle_nodes': len(vehicle_list),
                'total_nodes': len(nodes['rsu_nodes']) + len(nodes['vehicle_nodes'])
            }
        }

        debug(f"Graph built: {graph_data['metadata']}")
        return graph_data

    def _create_nodes(self, dqn_list, vehicle_list):
        """
        创建图节点

        Returns:
            nodes: 节点字典，包含RSU节点和车辆节点
        """
        nodes = {
            'rsu_nodes': [],
            'vehicle_nodes': []
        }

        # 创建RSU节点
        for i, dqn in enumerate(dqn_list):
            rsu_node = {
                'id': f"rsu_{dqn.dqn_id}",
                'type': 'rsu',
                'original_id': dqn.dqn_id,
                'position': (dqn.bs_loc[0], dqn.bs_loc[1]),  # 2D位置
                'features': self._extract_rsu_features(dqn)
            }
            nodes['rsu_nodes'].append(rsu_node)

        # 创建车辆节点
        for i, vehicle in enumerate(vehicle_list):
            vehicle_node = {
                'id': f"vehicle_{vehicle.id}",
                'type': 'vehicle',
                'original_id': vehicle.id,
                'position': vehicle.curr_loc,
                'direction': vehicle.curr_dir,
                'features': self._extract_vehicle_features(vehicle)
            }
            nodes['vehicle_nodes'].append(vehicle_node)

        debug(f"Created {len(nodes['rsu_nodes'])} RSU nodes and {len(nodes['vehicle_nodes'])} vehicle nodes")
        return nodes

    def _extract_rsu_features(self, dqn):
        """
        提取RSU节点特征 - 固定维度

        Returns:
            features: RSU特征向量 (固定长度7)
        """
        if hasattr(dqn, 'vehicle_in_dqn_range_by_distance'):
            vehicle_count = len(dqn.vehicle_in_dqn_range_by_distance)
        else:
            vehicle_count = 0

        # 基础特征 (固定5个)
        features = [
            dqn.bs_loc[0] / SCENE_SCALE_X,  # 归一化x坐标
            dqn.bs_loc[1] / SCENE_SCALE_Y,  # 归一化y坐标
            float(getattr(dqn, 'vehicle_exist_curr', False)),  # 当前是否有车辆
            vehicle_count / 10.0,  # 车辆数量（归一化）
            getattr(dqn, 'prev_snr', 0.0) / 50.0,  # 历史SNR（归一化）
        ]

        # CSI特征 (固定2个，即使没有CSI信息也填充0)
        csi_distance = 0.0
        csi_snr = 0.0

        if USE_UMI_NLOS_MODEL and hasattr(dqn, 'csi_states_curr') and dqn.csi_states_curr:
            # 使用最近的CSI信息
            csi_distance = dqn.csi_states_curr[0] / 1000.0 if len(dqn.csi_states_curr) > 0 else 0.0
            csi_snr = dqn.csi_states_curr[3] / 50.0 if len(dqn.csi_states_curr) > 3 else 0.0

        features.extend([csi_distance, csi_snr])

        # 添加 V2I 干扰特征
        # (dqn.prev_v2i_interference 是一个很小的线性值 (W), 需要归一化)
        v2i_interference_W = getattr(dqn, 'prev_v2i_interference', 0.0)
        # (我们假设 1e-9 W (=-60dBm) 是一个合理的干扰上限)
        normalized_interference = np.clip(v2i_interference_W / 1e-9, 0.0, 1.0)
        features.append(normalized_interference)

        v2v_interference_W = getattr(dqn, 'prev_v2v_interference', 0.0)
        normalized_v2v_interference = np.clip(v2v_interference_W / 1e-9, 0.0, 1.0)
        features.append(normalized_v2v_interference)

        if len(features) != self.rsu_feature_dim:
            debug(f"Warning: RSU feature dimension mismatch. Expected {self.rsu_feature_dim}, got {len(features)}")
            # 填充或截断到正确长度
            if len(features) < self.rsu_feature_dim:
                features.extend([0.0] * (self.rsu_feature_dim - len(features)))
            else:
                features = features[:self.rsu_feature_dim]

        return features

    def _extract_vehicle_features(self, vehicle):
        """
        提取车辆节点特征 - 固定维度

        Returns:
            features: 车辆特征向量 (固定长度6)
        """
        features = [
            vehicle.curr_loc[0] / SCENE_SCALE_X,  # 归一化x坐标
            vehicle.curr_loc[1] / SCENE_SCALE_Y,  # 归一化y坐标
            (vehicle.curr_dir[0] + 1) / 2.0,  # 水平方向归一化 [-1,0,1] -> [0,0.5,1]
            (vehicle.curr_dir[1] + 1) / 2.0,  # 垂直方向归一化
            float(vehicle.first_occur),  # 是否首次出现
        ]

        # 距离特征
        distance_feature = 0.0
        if hasattr(vehicle, 'distance_to_bs') and vehicle.distance_to_bs is not None:
            distance_feature = vehicle.distance_to_bs / 1000.0  # 距离归一化

        features.append(distance_feature)

        # === 确保特征向量长度为6 ===
        if len(features) != self.vehicle_feature_dim:
            debug(
                f"Warning: Vehicle feature dimension mismatch. Expected {self.vehicle_feature_dim}, got {len(features)}")
            # 填充或截断到正确长度
            if len(features) < self.vehicle_feature_dim:
                features.extend([0.0] * (self.vehicle_feature_dim - len(features)))
            else:
                features = features[:self.vehicle_feature_dim]

        return features

    def _create_edges(self, nodes, dqn_list, vehicle_list, epoch):
        """
        创建多种类型的边

        Returns:
            edges: 边字典，包含不同类型的边
        """
        edges = {
            'communication': [],  # 通信边
            'interference': [],  # 干扰边
            'proximity': []  # 邻近边
        }

        # 1. 通信边: RSU与车辆之间的通信关系
        edges['communication'] = self._calculate_communication_edges(nodes, dqn_list, vehicle_list)

        # 2. 干扰边: 车辆之间的干扰关系
        # --- MODIFIED: 禁用干扰边 (沿用 Ablation 1 的最优发现) ---
        edges['interference'] = []
        # edges['interference'] = self._calculate_interference_edges(nodes, vehicle_list)
        debug_print("[P1v13] Interference edge skipped. Optimal Sparsity.")
        # --- END MODIFIED ---


        # 3. 邻近边: 空间邻近关系
        edges['proximity'] = self._calculate_proximity_edges(nodes, dqn_list, vehicle_list)

        # 统计边数量
        for edge_type in self.edge_types:
            debug(f"{edge_type} edges: {len(edges[edge_type])}")

        return edges

    def _calculate_communication_edges(self, nodes, dqn_list, vehicle_list):
        """
        计算通信边

        Returns:
            communication_edges: 通信边列表
        """
        communication_edges = []

        for rsu_node in nodes['rsu_nodes']:
            # 获取DQN对象
            rsu_dqn = None
            for dqn in dqn_list:
                if dqn.dqn_id == rsu_node['original_id']:
                    rsu_dqn = dqn
                    break

            if rsu_dqn is None:
                continue

            for vehicle_node in nodes['vehicle_nodes']:
                # 安全地获取车辆对象
                vehicle = None
                for v in vehicle_list:
                    if v.id == vehicle_node['original_id']:
                        vehicle = v
                        break

                if vehicle is None:
                    continue

                # 检查车辆是否在RSU范围内
                if (rsu_dqn.start[0] <= vehicle.curr_loc[0] <= rsu_dqn.end[0] and
                        rsu_dqn.start[1] <= vehicle.curr_loc[1] <= rsu_dqn.end[1]):

                    try:
                        # 1. 计算距离
                        distance = global_channel_model.calculate_3d_distance(
                            (rsu_dqn.bs_loc[0], rsu_dqn.bs_loc[1]), vehicle.curr_loc)

                        # 2. 获取路径损耗和预估SNR
                        #    (使用一个标准功率，例如 TRANSMITTDE_POWER 的 10% 作为基准)
                        base_power = TRANSMITTDE_POWER * 0.1
                        csi_info = global_channel_model.get_channel_state_info(
                            (rsu_dqn.bs_loc[0], rsu_dqn.bs_loc[1]),
                            vehicle.curr_loc,
                            tx_power=base_power,
                            bandwidth=V2V_CHANNEL_BANDWIDTH
                        )

                        path_loss_db = csi_info['path_loss_total_db']
                        snr_db = csi_info['snr_db']

                        # 3. 创建边的特征向量 [weight, distance, path_loss, snr]
                        #    进行归一化处理，使GNN易于学习
                        features = [
                            1.0 - (distance / self.communication_threshold),  # weight
                            distance / 1000.0,  # 归一化距离 (km)
                            path_loss_db / 100.0,  # 归一化 PL
                            snr_db / 20.0  # 归一化 SNR
                        ]

                        # 4. 创建边
                        if distance <= self.communication_threshold:
                            communication_edges.append({
                                'source': rsu_node['id'],
                                'target': vehicle_node['id'],
                                'type': 'communication',
                                'distance': distance,
                                'features': features  # <--- 使用新的特征向量
                            })

                    except Exception as e:
                        debug(f"Error calculating edge CSI: {e}")

        return communication_edges

    def _calculate_interference_edges(self, nodes, vehicle_list):
        """
        计算干扰边

        Returns:
            interference_edges: 干扰边列表
        """
        interference_edges = []

        # 车辆之间的干扰（同向车辆更容易相互干扰）
        for i, vehicle_node_i in enumerate(nodes['vehicle_nodes']):
            for j, vehicle_node_j in enumerate(nodes['vehicle_nodes']):
                if i >= j:  # 避免重复计算
                    continue

                # === 修复：安全地获取车辆对象 ===
                vehicle_i = None
                vehicle_j = None
                for v in vehicle_list:
                    if v.id == vehicle_node_i['original_id']:
                        vehicle_i = v
                    if v.id == vehicle_node_j['original_id']:
                        vehicle_j = v

                if vehicle_i is None or vehicle_j is None:
                    continue

                # 计算距离
                distance = np.sqrt(
                    (vehicle_i.curr_loc[0] - vehicle_j.curr_loc[0]) ** 2 +
                    (vehicle_i.curr_loc[1] - vehicle_j.curr_loc[1]) ** 2
                )

                # 如果距离在干扰阈值内，创建干扰边
                if distance <= self.interference_threshold:
                    # 干扰强度与距离和方向相似性相关
                    direction_similarity = self._calculate_direction_similarity(
                        vehicle_i.curr_dir, vehicle_j.curr_dir)

                    interference_strength = (1.0 - (distance / self.interference_threshold)) * direction_similarity

                    interference_edges.append({
                        'source': vehicle_node_i['id'],
                        'target': vehicle_node_j['id'],
                        'type': 'interference',
                        'distance': distance,
                        'direction_similarity': direction_similarity,
                        'weight': interference_strength
                    })

        return interference_edges

    def _calculate_proximity_edges(self, nodes, dqn_list, vehicle_list):
        """
        计算邻近边

        Returns:
            proximity_edges: 邻近边列表
        """
        proximity_edges = []

        # 所有节点之间的空间邻近关系
        all_nodes = nodes['rsu_nodes'] + nodes['vehicle_nodes']

        for i, node_i in enumerate(all_nodes):
            for j, node_j in enumerate(all_nodes):
                if i >= j:  # 避免重复计算
                    continue

                # 计算距离
                distance = np.sqrt(
                    (node_i['position'][0] - node_j['position'][0]) ** 2 +
                    (node_i['position'][1] - node_j['position'][1]) ** 2
                )

                # 如果距离在邻近阈值内，创建邻近边
                if distance <= self.proximity_threshold:
                    proximity_weight = 1.0 - (distance / self.proximity_threshold)

                    proximity_edges.append({
                        'source': node_i['id'],
                        'target': node_j['id'],
                        'type': 'proximity',
                        'distance': distance,
                        'weight': proximity_weight
                    })

        return proximity_edges

    def _calculate_direction_similarity(self, dir1, dir2):
        """
        计算两个方向的相似性

        Returns:
            similarity: 方向相似性 [0,1]
        """
        # 计算方向向量的点积（余弦相似性）
        dot_product = dir1[0] * dir2[0] + dir1[1] * dir2[1]

        # 归一化到[0,1]范围
        similarity = (dot_product + 1) / 2.0

        return similarity

    def _extract_node_features(self, nodes, dqn_list, vehicle_list):
        """
        提取所有节点的特征矩阵
        """
        all_features = []
        node_types = []

        # RSU节点特征
        for rsu_node in nodes['rsu_nodes']:
            all_features.append(rsu_node['features'])
            node_types.append(0)  # 0表示RSU节点

        # 车辆节点特征
        for vehicle_node in nodes['vehicle_nodes']:
            all_features.append(vehicle_node['features'])
            node_types.append(1)  # 1表示车辆节点

        # === 修复：验证所有特征向量长度一致 ===
        feature_lengths = [len(features) for features in all_features]
        if len(set(feature_lengths)) > 1:
            debug(f"Warning: Inconsistent feature dimensions: {feature_lengths}")
            # 统一特征维度到最大值
            max_len = max(feature_lengths)
            for i in range(len(all_features)):
                if len(all_features[i]) < max_len:
                    all_features[i].extend([0.0] * (max_len - len(all_features[i])))
                else:
                    all_features[i] = all_features[i][:max_len]

        # 转换为Tensor
        node_features = torch.FloatTensor(all_features)
        node_types = torch.LongTensor(node_types)

        debug(f"Node features shape: {node_features.shape}, types: {node_types.shape}")

        return {
            'features': node_features,
            'types': node_types
        }

    def _extract_edge_features(self, edges, nodes):
        """
        提取边特征
        """
        edge_features = {}

        for edge_type in self.edge_types:
            edge_list = edges[edge_type]

            if not edge_list:
                edge_features[edge_type] = None
                continue

            # 构建边索引矩阵
            edge_index_list = []
            edge_attr_list = []

            node_id_to_index = {}
            all_nodes = nodes['rsu_nodes'] + nodes['vehicle_nodes']
            for idx, node in enumerate(all_nodes):
                node_id_to_index[node['id']] = idx

            for edge in edge_list:
                src_idx = node_id_to_index[edge['source']]
                tgt_idx = node_id_to_index[edge['target']]

                edge_index_list.append([src_idx, tgt_idx])


                if edge_type == 'communication':
                    edge_attr_list.append(edge['features'])  # 附加 [f1, f2, f3, f4]
                else:
                    # 其他边类型保持原样 (或填充到相同维度)
                    # 为简单起见，我们假设 'interference' 和 'proximity' 边也使用 'weight'
                    # 并填充到 self.comm_edge_feature_dim 维度
                    default_features = [edge['weight']] + [0.0] * (self.comm_edge_feature_dim - 1)
                    edge_attr_list.append(default_features)


            edge_features[edge_type] = {
                'edge_index': torch.LongTensor(edge_index_list).t().contiguous(),
                'edge_attr': torch.FloatTensor(edge_attr_list)
            }

        return edge_features

    def visualize_graph_simple(self, graph_data, filename=None):
        """
        简单可视化图结构（控制台输出）
        """
        nodes = graph_data['nodes']
        edges = graph_data['edges']
        node_features = graph_data['node_features']

        debug("=== Graph Visualization ===")
        debug(f"Total Nodes: {len(nodes['rsu_nodes']) + len(nodes['vehicle_nodes'])}")
        debug(f"  - RSU Nodes: {len(nodes['rsu_nodes'])}")
        debug(f"  - Vehicle Nodes: {len(nodes['vehicle_nodes'])}")
        debug(f"Node Features Shape: {node_features['features'].shape}")

        for edge_type in self.edge_types:
            debug(f"{edge_type.capitalize()} Edges: {len(edges[edge_type])}")

        # 显示部分节点信息
        debug("Sample RSU Nodes:")
        for i, rsu in enumerate(nodes['rsu_nodes'][:3]):  # 只显示前3个
            debug(f"  {rsu['id']} at {rsu['position']}")

        debug("Sample Vehicle Nodes:")
        for i, vehicle in enumerate(nodes['vehicle_nodes'][:3]):  # 只显示前3个
            debug(f"  {vehicle['id']} at {vehicle['position']}, dir: {vehicle['direction']}")

        debug("=== End Graph Visualization ===")

    def _get_distance(self, pos1, pos2):
        """辅助函数：计算两个 (x,y) 坐标之间的 2D 距离"""
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def build_spatial_subgraph(self, center_dqn, all_dqns, all_vehicles, epoch, radius=GNN_INFERENCE_RADIUS):
        """
        构建一个以 center_dqn 为中心，半径为 radius 的局部子图。
        仅用于推理阶段。
        """
        center_pos = center_dqn.bs_loc

        # 1. 过滤 RSU (DQN) 列表
        # 确保中心 DQN 始终在内，并包含所有在半径内的其他 RSU
        filtered_dqns = [center_dqn]
        for dqn in all_dqns:
            if dqn.dqn_id == center_dqn.dqn_id:
                continue  # 已经加过了

            dqn_pos = dqn.bs_loc
            distance = self._get_distance(center_pos, dqn_pos)
            if distance <= radius:
                filtered_dqns.append(dqn)

        # 2. 过滤车辆列表
        # 包含所有在半径内的车辆
        filtered_vehicles = []
        for vehicle in all_vehicles:
            vehicle_pos = vehicle.curr_loc
            distance = self._get_distance(center_pos, vehicle_pos)
            if distance <= radius:
                filtered_vehicles.append(vehicle)

        debug(f"[Subgraph] Center: {center_dqn.dqn_id}. "
              f"Radius: {radius}m. "
              f"RSUs: {len(all_dqns)} -> {len(filtered_dqns)}. "
              f"Vehicles: {len(all_vehicles)} -> {len(filtered_vehicles)}")

        # 3. 重用现有的图构建器，但只传入过滤后的列表
        # 注意：这将返回一个完整的图数据字典，但只包含子图的节点和边
        return self.build_dynamic_graph(filtered_dqns, filtered_vehicles, epoch)

# 全局图构建器实例
global_graph_builder = GraphBuilder()


def test_graph_builder():
    """测试图构建器"""
    from Classes import DQN, Vehicle
    import torch

    debug_print("Testing GraphBuilder...")

    # 创建测试数据
    device = torch.device("cpu")
    test_dqn_list = []
    test_vehicle_list = []

    # 创建测试DQN - 修复：初始化必要的属性
    test_dqn = DQN(
        RL_N_STATES, RL_N_HIDDEN, RL_N_ACTIONS,
        dqn_id=1, start_x=0, start_y=400, end_x=400, end_y=400
    )
    # === 修复：手动设置测试所需的属性 ===
    test_dqn.vehicle_exist_curr = True
    test_dqn.prev_snr = 25.0
    test_dqn.vehicle_in_dqn_range_by_distance = []  # 初始化缺失的属性
    test_dqn_list.append(test_dqn)

    # 创建测试车辆
    test_vehicle = Vehicle(1, 100, 400, 1, 0)  # id, x, y, horizontal, vertical
    test_vehicle_list.append(test_vehicle)

    test_vehicle2 = Vehicle(2, 300, 400, -1, 0)
    test_vehicle_list.append(test_vehicle2)

    # 构建图
    graph_data = global_graph_builder.build_dynamic_graph(
        test_dqn_list, test_vehicle_list, epoch=1)

    # 可视化
    global_graph_builder.visualize_graph_simple(graph_data)

    debug_print("GraphBuilder test completed!")


if __name__ == "__main__":
    set_debug_mode(True)
    test_graph_builder()
