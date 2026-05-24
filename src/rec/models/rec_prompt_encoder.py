import math
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import RGCNConv
from src.common.graph_encoders import CustomGCNConv
from src.common.graph_fusion import GraphFusionSettings, MultiModalEntityFusion
from src.common.modality_config import ModalityConfig


class InspiredRecommendationPromptEncoder(nn.Module):
    def __init__(
        self, hidden_size, token_hidden_size, n_head, n_layer, n_block,
        n_entity, num_relations, num_bases, edge_index, edge_type,edge_index_c,edge_index_t_s,edge_index_i_s, idx_to_id,
        n_prefix_rec=None, n_prefix_conv=None, modality_config=None,
    ):
        super(InspiredRecommendationPromptEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.n_layer = n_layer
        self.n_block = n_block
        self.n_prefix_rec = n_prefix_rec
        self.n_prefix_conv = n_prefix_conv
        self.modality_config = modality_config or ModalityConfig(add_item_semantic_to_entities=True)
        self.idx_to_id = idx_to_id
        self.idx_to_id_tensor = torch.tensor([self.idx_to_id[i] for i in range(len(self.idx_to_id))], dtype=torch.long)
        self.sorted_ids = sorted(self.idx_to_id.keys())
        self.sorted_indices = torch.tensor([self.idx_to_id[id] for id in self.sorted_ids], dtype=torch.long)
        entity_hidden_size = hidden_size // 2
        self.kg_encoder = RGCNConv(entity_hidden_size, entity_hidden_size, num_relations=num_relations,num_bases=num_bases)
        self.conv_c1 = CustomGCNConv()  # LightGCN
        self.conv_c2 = CustomGCNConv()  # LightGCN
        self.conv_c3 = CustomGCNConv()  # LightGCN
        self.conv_ts1 = CustomGCNConv()  # LightGCN
        self.conv_ts2 = CustomGCNConv()  # LightGCN
        self.conv_ts3 = CustomGCNConv()  # LightGCN
        self.conv_is1 = CustomGCNConv()  # LightGCN
        self.conv_is2 = CustomGCNConv()  # LightGCN
        self.conv_is3 = CustomGCNConv()  # LightGCN

        self.node_embeds = nn.Parameter(torch.empty(n_entity, entity_hidden_size))
        stdv = math.sqrt(6.0 / (self.node_embeds.size(-2) + self.node_embeds.size(-1)))
        self.node_embeds.data.uniform_(-stdv, stdv)
        self.edge_index = nn.Parameter(edge_index, requires_grad=False)
        self.edge_index_c = nn.Parameter(edge_index_c,requires_grad=False)
        self.edge_index_t_s = nn.Parameter(edge_index_t_s,requires_grad=False)
        self.edge_index_i_s = nn.Parameter(edge_index_i_s,requires_grad=False)

        self.edge_type = nn.Parameter(edge_type, requires_grad=False)
        self.entity_proj1 = nn.Sequential(
            nn.Linear(entity_hidden_size, entity_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(entity_hidden_size // 2, entity_hidden_size),
        )
        self.entity_proj2 = nn.Linear(entity_hidden_size, hidden_size)
        self.entity_fusion = MultiModalEntityFusion(
            GraphFusionSettings(
                kg_input_mode="node",
                add_node_to_kg=True,
                collaborative_base="entity",
                semantic_depth=2,
            )
        )
        self.token_proj1 = nn.Sequential(
            nn.Linear(token_hidden_size, token_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(token_hidden_size // 2, token_hidden_size),
        )
        self.token_proj2 = nn.Linear(token_hidden_size, hidden_size)
        self.cross_attn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.prompt_proj1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        self.prompt_proj2 = nn.Linear(hidden_size, n_layer * n_block * hidden_size)
        if self.n_prefix_rec is not None:
            self.rec_prefix_embeds = nn.Parameter(torch.empty(n_prefix_rec, hidden_size))
            nn.init.normal_(self.rec_prefix_embeds)
            self.rec_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )
        if self.n_prefix_conv is not None:
            self.conv_prefix_embeds = nn.Parameter(torch.empty(n_prefix_conv, hidden_size))
            nn.init.normal_(self.conv_prefix_embeds)
            self.conv_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )

    def set_and_fix_node_embed(self, node_embeds: torch.Tensor):
        self.node_embeds.data = node_embeds
        self.node_embeds.requires_grad_(False)

    def get_entity_embeds(self):
        return self.entity_fusion(
            node_embeds=self.node_embeds,
            kg_encoder=self.kg_encoder,
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            collaborative_convs=(self.conv_c1, self.conv_c2, self.conv_c3),
            edge_index_c=self.edge_index_c,
            text_convs=(self.conv_ts1, self.conv_ts2, self.conv_ts3),
            edge_index_t_s=self.edge_index_t_s,
            image_convs=(self.conv_is1, self.conv_is2, self.conv_is3),
            edge_index_i_s=self.edge_index_i_s,
            sorted_indices=self.sorted_indices,
            idx_to_id_tensor=self.idx_to_id_tensor,
            entity_proj1=self.entity_proj1,
            entity_proj2=self.entity_proj2,
            modality_config=self.modality_config,
        )

    def forward(self, entity_ids=None, token_embeds=None, output_entity=False, use_rec_prefix=False,
                use_conv_prefix=False, return_entity_embeds=False):
        batch_size, entity_embeds, entity_len, token_len = None, None, None, None
        all_entity_embeds = None
        if entity_ids is not None:
            batch_size, entity_len = entity_ids.shape[:2]
            all_entity_embeds = self.get_entity_embeds()
            entity_embeds = all_entity_embeds[entity_ids]
        if token_embeds is not None:
            batch_size, token_len = token_embeds.shape[:2]
            token_embeds = self.token_proj1(token_embeds) + token_embeds  
            token_embeds = self.token_proj2(token_embeds)

        if entity_embeds is not None and token_embeds is not None:
            attn_weights = self.cross_attn(token_embeds) @ entity_embeds.permute(0, 2,
                                                                                 1)  
            attn_weights /= self.hidden_size

            if output_entity:
                token_weights = F.softmax(attn_weights, dim=1).permute(0, 2, 1)
                prompt_embeds = token_weights @ token_embeds + entity_embeds
                token_weights_embeds = token_weights @ token_embeds  # 形状为 (batch_size, seq_len, num_entities)
                token_rep = token_weights_embeds.mean(dim=1)  # (batch_size, hidden_size)
                entity_rep = entity_embeds.mean(dim=1)  # (batch_size, hidden_size)
                temperature = 0.07
                logits = F.cosine_similarity(token_rep.unsqueeze(1), entity_rep.unsqueeze(0), dim=-1)
                logits /= temperature
                labels = torch.arange(logits.size(0), device=logits.device)  # (batch_size,)
                loss_cl = F.cross_entropy(logits, labels)
                prompt_len = entity_len
            else:
                entity_weights = F.softmax(attn_weights, dim=2)
                prompt_embeds = entity_weights @ entity_embeds + token_embeds
                prompt_len = token_len
        elif entity_embeds is not None:
            prompt_embeds = entity_embeds
            prompt_len = entity_len
        else:
            prompt_embeds = token_embeds
            prompt_len = token_len

        if self.n_prefix_rec is not None and use_rec_prefix:
            prefix_embeds = self.rec_prefix_proj(self.rec_prefix_embeds) + self.rec_prefix_embeds
            prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
            prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            prompt_len += self.n_prefix_rec
        if self.n_prefix_conv is not None and use_conv_prefix:
            prefix_embeds = self.conv_prefix_proj(self.conv_prefix_embeds) + self.conv_prefix_embeds
            prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
            prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            prompt_len += self.n_prefix_conv

        prompt_embeds = self.prompt_proj1(prompt_embeds) + prompt_embeds
        prompt_embeds = self.prompt_proj2(prompt_embeds)
        prompt_embeds = prompt_embeds.reshape(
            batch_size, prompt_len, self.n_layer, self.n_block, self.n_head, self.head_dim
        ).permute(2, 3, 0, 4, 1, 5)  

        if return_entity_embeds:
            return prompt_embeds,loss_cl,all_entity_embeds
        return prompt_embeds,loss_cl

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        state_dict = {k: v for k, v in self.state_dict().items() if 'edge' not in k}
        save_path = os.path.join(save_dir, 'model.pt')
        torch.save(state_dict, save_path)

    def load(self, load_dir):
        load_path = os.path.join(load_dir, 'model.pt')
        missing_keys, unexpected_keys = self.load_state_dict(
            torch.load(load_path, map_location=torch.device('cpu')), strict=False
        )
        print(missing_keys, unexpected_keys)




class RedialRecommendationPromptEncoder(nn.Module):
    def __init__(
        self, hidden_size, token_hidden_size, n_head, n_layer, n_block,
        n_entity, num_relations, num_bases, edge_index, edge_type,edge_index_c,edge_index_t_s,edge_index_i_s, idx_to_id,
        n_prefix_rec=None, n_prefix_conv=None, modality_config=None,
    ):
        super(RedialRecommendationPromptEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.n_layer = n_layer
        self.n_block = n_block
        self.n_prefix_rec = n_prefix_rec
        self.n_prefix_conv = n_prefix_conv
        self.modality_config = modality_config or ModalityConfig(add_item_semantic_to_entities=False)

        self.idx_to_id = idx_to_id
        self.idx_to_id_tensor = torch.tensor([self.idx_to_id[i] for i in range(len(self.idx_to_id))], dtype=torch.long)
        self.sorted_ids = sorted(self.idx_to_id.keys())
        self.sorted_indices = torch.tensor([self.idx_to_id[id] for id in self.sorted_ids], dtype=torch.long)


        entity_hidden_size = hidden_size // 2
        self.kg_encoder = RGCNConv(entity_hidden_size, entity_hidden_size, num_relations=num_relations,
                                   num_bases=num_bases)
        self.conv_c1 = CustomGCNConv()  # LightGCN
        self.conv_c2 = CustomGCNConv()  # LightGCN
        self.conv_c3 = CustomGCNConv()  # LightGCN
        self.conv_ts1 = CustomGCNConv()  # LightGCN
        self.conv_ts2 = CustomGCNConv()  # LightGCN
        self.conv_ts3 = CustomGCNConv()  # LightGCN
        self.conv_is1 = CustomGCNConv()  # LightGCN
        self.conv_is2 = CustomGCNConv()  # LightGCN
        self.conv_is3 = CustomGCNConv()  # LightGCN


        self.node_embeds = nn.Parameter(torch.empty(n_entity, entity_hidden_size))
        stdv = math.sqrt(6.0 / (self.node_embeds.size(-2) + self.node_embeds.size(-1)))
        self.node_embeds.data.uniform_(-stdv, stdv)
        self.edge_index = nn.Parameter(edge_index, requires_grad=False)
        self.edge_index_c = nn.Parameter(edge_index_c,requires_grad=False)
        self.edge_index_t_s = nn.Parameter(edge_index_t_s,requires_grad=False)
        self.edge_index_i_s = nn.Parameter(edge_index_i_s,requires_grad=False)

        self.edge_type = nn.Parameter(edge_type, requires_grad=False)
        self.entity_proj1 = nn.Sequential(
            nn.Linear(entity_hidden_size, entity_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(entity_hidden_size // 2, entity_hidden_size),
        )
        self.entity_proj2 = nn.Linear(entity_hidden_size, hidden_size)
        self.entity_fusion = MultiModalEntityFusion(
            GraphFusionSettings(
                kg_input_mode="node",
                add_node_to_kg=True,
                collaborative_base="node",
                semantic_depth=2,
            )
        )

        self.token_proj1 = nn.Sequential(
            nn.Linear(token_hidden_size, token_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(token_hidden_size // 2, token_hidden_size),
        )
        self.token_proj2 = nn.Linear(token_hidden_size, hidden_size)

        self.cross_attn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.prompt_proj1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        self.prompt_proj2 = nn.Linear(hidden_size, n_layer * n_block * hidden_size)

        if self.n_prefix_rec is not None:
            self.rec_prefix_embeds = nn.Parameter(torch.empty(n_prefix_rec, hidden_size))
            nn.init.normal_(self.rec_prefix_embeds)
            self.rec_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )
        if self.n_prefix_conv is not None:
            self.conv_prefix_embeds = nn.Parameter(torch.empty(n_prefix_conv, hidden_size))
            nn.init.normal_(self.conv_prefix_embeds)
            self.conv_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )

    def set_and_fix_node_embed(self, node_embeds: torch.Tensor):
        self.node_embeds.data = node_embeds
        self.node_embeds.requires_grad_(False)

    def get_entity_embeds(self):
        return self.entity_fusion(
            node_embeds=self.node_embeds,
            kg_encoder=self.kg_encoder,
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            collaborative_convs=(self.conv_c1, self.conv_c2, self.conv_c3),
            edge_index_c=self.edge_index_c,
            text_convs=(self.conv_ts1, self.conv_ts2, self.conv_ts3),
            edge_index_t_s=self.edge_index_t_s,
            image_convs=(self.conv_is1, self.conv_is2, self.conv_is3),
            edge_index_i_s=self.edge_index_i_s,
            sorted_indices=self.sorted_indices,
            idx_to_id_tensor=self.idx_to_id_tensor,
            entity_proj1=self.entity_proj1,
            entity_proj2=self.entity_proj2,
            modality_config=self.modality_config,
        )

    def forward(self, entity_ids=None, token_embeds=None, output_entity=False, use_rec_prefix=False,
                use_conv_prefix=False, return_entity_embeds=False):
        batch_size, entity_embeds, entity_len, token_len = None, None, None, None
        all_entity_embeds = None
        if entity_ids is not None:
            batch_size, entity_len = entity_ids.shape[:2]
            all_entity_embeds = self.get_entity_embeds()
            entity_embeds = all_entity_embeds[entity_ids]
        if token_embeds is not None:
            batch_size, token_len = token_embeds.shape[:2]
            token_embeds = self.token_proj1(token_embeds) + token_embeds  
            token_embeds = self.token_proj2(token_embeds)

        if entity_embeds is not None and token_embeds is not None:
            attn_weights = self.cross_attn(token_embeds) @ entity_embeds.permute(0, 2,
                                                                                 1)  
            attn_weights /= self.hidden_size

            if output_entity:
                token_weights = F.softmax(attn_weights, dim=1).permute(0, 2, 1)
                prompt_embeds = token_weights @ token_embeds + entity_embeds

                token_weights_embeds = token_weights @ token_embeds  # 形状为 (batch_size, seq_len, num_entities)
                token_rep = token_weights_embeds.mean(dim=1)  # (batch_size, hidden_size)
                entity_rep = entity_embeds.mean(dim=1)  # (batch_size, hidden_size)
                temperature = 0.07
                logits = F.cosine_similarity(token_rep.unsqueeze(1), entity_rep.unsqueeze(0), dim=-1)
                logits /= temperature
                labels = torch.arange(logits.size(0), device=logits.device)  # (batch_size,)
                loss_cl = F.cross_entropy(logits, labels)
                prompt_len = entity_len
            else:
                entity_weights = F.softmax(attn_weights, dim=2)
                prompt_embeds = entity_weights @ entity_embeds + token_embeds
                prompt_len = token_len
        elif entity_embeds is not None:
            prompt_embeds = entity_embeds
            prompt_len = entity_len
        else:
            prompt_embeds = token_embeds
            prompt_len = token_len

        if self.n_prefix_rec is not None and use_rec_prefix:
            prefix_embeds = self.rec_prefix_proj(self.rec_prefix_embeds) + self.rec_prefix_embeds
            prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
            prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            prompt_len += self.n_prefix_rec
        if self.n_prefix_conv is not None and use_conv_prefix:
            prefix_embeds = self.conv_prefix_proj(self.conv_prefix_embeds) + self.conv_prefix_embeds
            prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
            prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            prompt_len += self.n_prefix_conv

        prompt_embeds = self.prompt_proj1(prompt_embeds) + prompt_embeds
        prompt_embeds = self.prompt_proj2(prompt_embeds)
        prompt_embeds = prompt_embeds.reshape(
            batch_size, prompt_len, self.n_layer, self.n_block, self.n_head, self.head_dim
        ).permute(2, 3, 0, 4, 1, 5)  

        if return_entity_embeds:
            return prompt_embeds,loss_cl,all_entity_embeds
        return prompt_embeds,loss_cl

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        state_dict = {k: v for k, v in self.state_dict().items() if 'edge' not in k}
        save_path = os.path.join(save_dir, 'model.pt')
        torch.save(state_dict, save_path)

    def load(self, load_dir):
        load_path = os.path.join(load_dir, 'model.pt')
        missing_keys, unexpected_keys = self.load_state_dict(
            torch.load(load_path, map_location=torch.device('cpu')), strict=False
        )
        print(missing_keys, unexpected_keys)
