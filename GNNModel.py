# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool  # <-- 确保使用 GATConv
from logger import debug, debug_print, set_debug_mode
from Parameters import *


class EnhancedHeteroGNN(nn.Module):
    """
    [P1v13] 驯服 GAT 架构：Edge Gating 和 Entropy 正则化框架。
    """

    def __init__(self, node_feature_dim=9, hidden_dim=64, num_heads=4, num_layers=2, dropout=0.2):
        super(EnhancedHeteroGNN, self).__init__()

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        from GraphBuilder import global_graph_builder
        comm_edge_dim = global_graph_builder.comm_edge_feature_dim
        self.edge_feature_dim = comm_edge_dim

        # 边类型
        self.edge_types = ['communication', 'interference', 'proximity']

        # --- MODIFIED 1: Edge-Type Gating 参数 (Suggestion A) ---
        # 为每种边类型定义一个可学习的权重，初始化为零 (Sigmoid(0)=0.5)
        self.edge_type_gates = nn.Parameter(torch.zeros(len(self.edge_types)))
        # --- END MODIFIED 1 ---

        # 节点类型编码 (结构保持不变)
        self.node_type_embedding = nn.Embedding(2, hidden_dim // 4)

        # GAT 层 (结构还原为 GATConv)
        self.edge_type_layers = nn.ModuleDict()
        for edge_type in self.edge_types:
            input_dim = node_feature_dim + (hidden_dim // 4)
            edge_layers = nn.ModuleList()
            edge_layers.append(
                GATConv(input_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout,
                        edge_dim=self.edge_feature_dim)
            )
            for _ in range(num_layers - 2):
                edge_layers.append(
                    GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout,
                            edge_dim=self.edge_feature_dim)
                )
            if num_layers > 1:
                edge_layers.append(
                    GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout, edge_dim=self.edge_feature_dim)
                )
            self.edge_type_layers[edge_type] = edge_layers

        # 边类型注意力权重 (融合目标)
        self.edge_type_attention = nn.Parameter(torch.ones(len(self.edge_types)))

        # 为车辆聚合定义注意力层 (保持不变)
        self.attn_pool_linear = nn.Linear(hidden_dim, 1)

        # 输出层 (保持不变)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, RL_N_ACTIONS)
        )

        self._init_weights()
        debug(f"P1v13 Stabilized GAT (w/ Edge Gating) initialized")

    def _init_weights(self):
        """初始化模型权重 (保持不变)"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)

    def forward(self, graph_data, dqn_id=None):
        """
        前向传播
        返回: Q值 和 Edge-Type Attention Logits
        """
        node_features = graph_data['node_features']['features']
        node_types = graph_data['node_features']['types']
        edge_features = graph_data['edge_features']

        batch_size = node_features.size(0)
        type_embedding = self.node_type_embedding(node_types)
        x = torch.cat([node_features, type_embedding], dim=1)

        edge_outputs = []

        # --- MODIFIED 2: 暴露 Edge-Type Attention Logits ---
        edge_attention_logits = self.edge_type_attention
        edge_weights = F.softmax(edge_attention_logits, dim=0)

        # --- MODIFIED 3: Edge-Type Gates (Suggestion A) ---
        edge_gates = torch.sigmoid(self.edge_type_gates)

        for i, edge_type in enumerate(self.edge_types):
            if edge_features[edge_type] is None:
                edge_outputs.append(torch.zeros(batch_size, self.hidden_dim, device=x.device))
                continue

            edge_index = edge_features[edge_type]['edge_index']
            edge_attr = edge_features[edge_type]['edge_attr']

            # --- MODIFIED 4: 应用 Edge-Type Gate 到边特征 ---
            # 门控值 [0, 1] 乘以边特征
            gated_edge_attr = edge_attr * edge_gates[i]

            layers = self.edge_type_layers[edge_type]
            x_edge = x.clone()

            for j, layer in enumerate(layers):
                # 使用门控后的边特征进行 GAT 卷积
                x_edge = layer(x_edge, edge_index, edge_attr=gated_edge_attr)
                if j < len(layers) - 1:
                    x_edge = F.elu(x_edge)
                    x_edge = F.dropout(x_edge, p=self.dropout, training=self.training)

            x_edge = x_edge * edge_weights[i]
            edge_outputs.append(x_edge)

        # 3. 合并不同边类型的输出
        if len(edge_outputs) > 0:
            stacked_outputs = torch.stack(edge_outputs, dim=0)
            x_combined = torch.sum(stacked_outputs, dim=0)
        else:
            x_combined = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        # 4. 提取特征 (逻辑不变)
        if dqn_id is not None:
            q_values = self._extract_local_features(x_combined, graph_data, dqn_id)
        else:
            q_values = self._extract_global_features(x_combined, graph_data)

        # 5. 提取特征
        if dqn_id is not None:
            q_values = self._extract_local_features(x_combined, graph_data, dqn_id)
        else:
            q_values = self._extract_global_features(x_combined, graph_data)

        # --- MODIFIED 5: 返回 Q 值和注意力 logits ---
        return q_values, edge_attention_logits

    def _extract_local_features(self, node_embeddings, graph_data, dqn_id):
        nodes = graph_data['nodes']
        target_rsu_index = -1
        for i, rsu_node in enumerate(nodes['rsu_nodes']):
            if rsu_node['original_id'] == dqn_id:
                target_rsu_index = i
                break
        if target_rsu_index == -1:
            return torch.zeros(RL_N_ACTIONS, device=node_embeddings.device)

        rsu_embedding = node_embeddings[target_rsu_index]
        vehicle_embeddings = []

        for vehicle_node in nodes['vehicle_nodes']:
            for edge in graph_data['edges']['communication']:
                if (edge['source'] == f"rsu_{dqn_id}" and edge['target'] == vehicle_node['id']):
                    vehicle_index = len(nodes['rsu_nodes']) + nodes['vehicle_nodes'].index(vehicle_node)
                    vehicle_embeddings.append(node_embeddings[vehicle_index])
                    break

        if vehicle_embeddings:
            vehicle_stack = torch.stack(vehicle_embeddings)
            attn_scores = self.attn_pool_linear(vehicle_stack)
            attn_weights = F.softmax(attn_scores, dim=0)
            vehicle_embedding = torch.mm(attn_weights.t(), vehicle_stack).squeeze(0)
        else:
            vehicle_embedding = torch.zeros_like(rsu_embedding)

        combined_features = torch.cat([rsu_embedding, vehicle_embedding], dim=0)
        q_values = self.output_layer(combined_features)
        return q_values

    def _extract_global_features(self, node_embeddings, graph_data):
        # ... (保持原样) ...
        nodes = graph_data['nodes']
        num_rsus = len(nodes['rsu_nodes'])
        all_q_values = []
        for dqn_id in range(1, num_rsus + 1):
            q_value = self._extract_local_features(node_embeddings, graph_data, dqn_id)
            all_q_values.append(q_value)
        if all_q_values:
            return torch.stack(all_q_values, dim=0)
        else:
            return torch.zeros(0, RL_N_ACTIONS, device=node_embeddings.device)

    def get_attention_weights(self, graph_data):
        # GCN 没有内部注意力，只返回边类型权重
        attention_info = {
            'edge_type_weights': F.softmax(self.edge_type_attention, dim=0).detach().cpu().numpy(),
            'edge_types': self.edge_types
        }
        return attention_info


# --- MODIFIED: 初始化时不传 num_heads ---
global_gnn_model = EnhancedHeteroGNN(
    node_feature_dim=9,
    hidden_dim=64,
    # num_heads=4,  <-- GCN 不需要
    num_layers=2,
    dropout=0.2
)
global_target_gnn_model = EnhancedHeteroGNN(
    node_feature_dim=9,
    hidden_dim=64,
    # num_heads=4,  <-- GCN 不需要
    num_layers=2,
    dropout=0.2
)


def update_target_gnn():
    global_target_gnn_model.load_state_dict(global_gnn_model.state_dict())
    global_target_gnn_model.eval()
    debug("Global Target GNN (GCN) network updated")


def update_target_gnn_soft(tau):
    try:
        with torch.no_grad():
            for target_param, online_param in zip(global_target_gnn_model.parameters(), global_gnn_model.parameters()):
                target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)
    except Exception as e:
        debug(f"Error during GNN soft update: {e}")


update_target_gnn()
debug_print("Global GCN 和 Target GCN 已初始化并同步。")

if __name__ == "__main__":
    set_debug_mode(True)
    debug_print("GNNModel.py (GCN Version) loaded.")