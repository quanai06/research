from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from src.common.modality_config import ModalityConfig


@dataclass(frozen=True)
class GraphFusionSettings:
    kg_input_mode: str = "node"
    add_node_to_kg: bool = True
    collaborative_base: str = "entity"
    semantic_depth: int = 2


class MultiModalEntityFusion(nn.Module):
    """Fuse MSCRS KG, collaborative, textual and image semantic graph embeddings.

    This module owns the algorithm, while prompt encoders still own the actual
    parameters/modules. Keeping parameter names in prompt encoders preserves
    checkpoint compatibility with the original scripts.
    """

    def __init__(self, settings: GraphFusionSettings):
        super().__init__()
        self.settings = settings

    def forward(
        self,
        *,
        node_embeds: torch.Tensor,
        kg_encoder: nn.Module,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        collaborative_convs: Sequence[nn.Module],
        edge_index_c: torch.Tensor,
        text_convs: Sequence[nn.Module],
        edge_index_t_s: torch.Tensor,
        image_convs: Sequence[nn.Module],
        edge_index_i_s: torch.Tensor,
        sorted_indices: torch.Tensor,
        idx_to_id_tensor: torch.Tensor,
        entity_proj1: nn.Module,
        entity_proj2: nn.Module,
        modality_config: ModalityConfig,
    ) -> torch.Tensor:
        entity_embeds = self._encode_kg(
            node_embeds=node_embeds,
            kg_encoder=kg_encoder,
            edge_index=edge_index,
            edge_type=edge_type,
            enabled=modality_config.use_kg_graph,
        )

        item_semantic_embeds = self._encode_item_semantics(
            entity_embeds=entity_embeds,
            sorted_indices=sorted_indices,
            text_convs=text_convs,
            edge_index_t_s=edge_index_t_s,
            image_convs=image_convs,
            edge_index_i_s=edge_index_i_s,
            modality_config=modality_config,
        )

        if modality_config.use_collaborative_graph:
            collab_input = entity_embeds if self.settings.collaborative_base == "entity" else node_embeds
            collab_embeds = self._run_graph_stack(collaborative_convs, collab_input, edge_index_c)
            entity_embeds = (sum(collab_embeds) + entity_embeds) / (len(collab_embeds) + 1)

        if modality_config.add_item_semantic_to_entities and item_semantic_embeds:
            movie_embeds = sum(item_semantic_embeds) / len(item_semantic_embeds)
            indices = idx_to_id_tensor.to(movie_embeds.device)[: len(movie_embeds)]
            entity_embeds = entity_embeds.index_add(0, indices, movie_embeds)

        entity_embeds = entity_proj1(entity_embeds) + entity_embeds
        return entity_proj2(entity_embeds)

    def _encode_kg(
        self,
        *,
        node_embeds: torch.Tensor,
        kg_encoder: nn.Module,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        enabled: bool,
    ) -> torch.Tensor:
        if not enabled:
            return node_embeds

        if self.settings.kg_input_mode == "none":
            entity_embeds = kg_encoder(None, edge_index, edge_type)
        else:
            entity_embeds = kg_encoder(node_embeds, edge_index, edge_type)

        if self.settings.add_node_to_kg:
            entity_embeds = entity_embeds + node_embeds
        return entity_embeds

    def _encode_item_semantics(
        self,
        *,
        entity_embeds: torch.Tensor,
        sorted_indices: torch.Tensor,
        text_convs: Sequence[nn.Module],
        edge_index_t_s: torch.Tensor,
        image_convs: Sequence[nn.Module],
        edge_index_i_s: torch.Tensor,
        modality_config: ModalityConfig,
    ) -> list[torch.Tensor]:
        sorted_indices = sorted_indices.to(entity_embeds.device)
        node_features = torch.index_select(entity_embeds, 0, sorted_indices)
        item_semantic_embeds = []

        if modality_config.use_text_graph:
            text_embeds = self._run_graph_stack(text_convs[: self.settings.semantic_depth], node_features, edge_index_t_s)
            item_semantic_embeds.append(sum(text_embeds) / len(text_embeds))

        if modality_config.use_image_graph:
            image_embeds = self._run_graph_stack(image_convs[: self.settings.semantic_depth], node_features, edge_index_i_s)
            item_semantic_embeds.append(sum(image_embeds) / len(image_embeds))

        return item_semantic_embeds

    @staticmethod
    def _run_graph_stack(convs: Sequence[nn.Module], features: torch.Tensor, edge_index: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        current = features
        for conv in convs:
            current = conv(current, edge_index)
            outputs.append(current)
        return outputs
