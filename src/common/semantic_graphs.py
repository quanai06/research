import json

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from src.common.path_utils import get_common_data_dir, require_data_file


class CollaborativeSemanticGraph:
    """Load the precomputed collaborative entity co-occurrence graph."""

    def __init__(self, dataset, split=None, entity_max_length=None, all_items=None, n_entity=None, debug=False):
        self.debug = debug
        self.entity_max_length = entity_max_length
        self.all_items = set(all_items) if all_items is not None else None
        self.n_entity = n_entity
        input_file = require_data_file(get_common_data_dir(dataset) / "edge_index_c.pt")
        self.edge_index_c = torch.load(input_file)

    def get_entity_co_info(self):
        return {"edge_index_c": self.edge_index_c}


class TopKItemSimilarityGraph:
    """Build a top-k item similarity graph from precomputed item embeddings."""

    edge_key = None

    def __init__(self, pad_entity_id, dataset, embedding_file, top_k):
        self.pad_entity_id = pad_entity_id
        self.top_k = top_k
        data_file = require_data_file(get_common_data_dir(dataset) / embedding_file)
        self.prepare_data(data_file)

    def prepare_data(self, data_file):
        with open(data_file, "r", encoding="utf-8") as f:
            id_embeddings = json.load(f)

        new_key = self.pad_entity_id
        id_embeddings[str(new_key)] = [1.0] * 768
        self.keys = [int(key) for key in id_embeddings.keys()]
        self.id_to_idx = {node_id: idx for idx, node_id in enumerate(self.keys)}
        self.idx_to_id = {idx: node_id for node_id, idx in self.id_to_idx.items()}

        embeddings = np.array(list(id_embeddings.values()))
        similarity_matrix = cosine_similarity(embeddings)
        top_k_indices = np.argsort(-similarity_matrix, axis=1)[:, 1 : self.top_k + 1]

        mapped_edges = []
        for i, key in enumerate(self.keys):
            if key == new_key:
                continue
            src_idx = self.id_to_idx[key]
            for target_idx in top_k_indices[i]:
                tgt_idx = self.id_to_idx[self.keys[target_idx]]
                mapped_edges.append([src_idx, tgt_idx])

        pad_idx = self.id_to_idx[new_key]
        mapped_edges.append([pad_idx, pad_idx])
        self.edge_index = torch.as_tensor(mapped_edges, dtype=torch.long).t()

    def get_info(self):
        return {
            self.edge_key: self.edge_index,
            "id_to_idx": self.id_to_idx,
            "idx_to_id": self.idx_to_id,
            "all_movie": self.keys,
        }


class TextSemanticGraph(TopKItemSimilarityGraph):
    edge_key = "edge_index_t_s"

    def __init__(self, pad_entity_id, dataset="inspired", top_k=10):
        super().__init__(
            pad_entity_id=pad_entity_id,
            dataset=dataset,
            embedding_file="id_embeddings_text.json",
            top_k=top_k,
        )

    def get_entity_ts_info(self):
        return self.get_info()


class ImageSemanticGraph(TopKItemSimilarityGraph):
    edge_key = "edge_index_i_s"

    def __init__(self, pad_entity_id, dataset="inspired", top_k=10):
        super().__init__(
            pad_entity_id=pad_entity_id,
            dataset=dataset,
            embedding_file="id_embeddings_image.json",
            top_k=top_k,
        )

    def get_entity_is_info(self):
        return self.get_info()
