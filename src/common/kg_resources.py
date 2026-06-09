import json
from collections import defaultdict
from dataclasses import dataclass

import torch
from loguru import logger

from src.common.path_utils import get_common_data_dir, require_data_file


@dataclass(frozen=True)
class KGProcessingConfig:
    add_self_loop: bool = False
    self_loop_id: int = 185
    filter_min_relation_count: int | None = None


class DBpedia:
    def __init__(self, dataset, debug=False, processing_config: KGProcessingConfig | None = None):
        self.debug = debug
        self.processing_config = processing_config or KGProcessingConfig()
        self.dataset_dir = get_common_data_dir(dataset)
        with open(require_data_file(self.dataset_dir / "dbpedia_subkg.json"), "r", encoding="utf-8") as f:
            self.entity_kg = json.load(f)
        with open(require_data_file(self.dataset_dir / "entity2id.json"), "r", encoding="utf-8") as f:
            self.entity2id = json.load(f)
        with open(require_data_file(self.dataset_dir / "relation2id.json"), "r", encoding="utf-8") as f:
            self.relation2id = json.load(f)
        with open(require_data_file(self.dataset_dir / "item_ids.json"), "r", encoding="utf-8") as f:
            self.item_ids = json.load(f)
        self._process_entity_kg()

    def _process_entity_kg(self):
        cfg = self.processing_config
        if cfg.add_self_loop:
            edge_list = self._build_self_loop_edges(cfg.self_loop_id)
        else:
            edge_list = self._build_bidirectional_edges()

        if cfg.filter_min_relation_count is not None:
            edge_list, num_relations = self._filter_and_reindex_relations(edge_list, cfg.filter_min_relation_count)
            self.num_relations = num_relations
        else:
            self.num_relations = len(self.relation2id)

        edge = torch.as_tensor(edge_list, dtype=torch.long)
        self.edge_index = edge[:, :2].t()
        self.edge_type = edge[:, 2]
        self.pad_entity_id = max(self.entity2id.values()) + 1
        self.num_entities = max(self.entity2id.values()) + 2

        if self.debug:
            logger.debug(
                f"#edge: {len(edge)}, #relation: {self.num_relations}, "
                f"#entity: {self.num_entities}, #item: {len(self.item_ids)}"
            )

    def _build_bidirectional_edges(self):
        edge_list = set()
        for entity in self.entity2id.values():
            if str(entity) not in self.entity_kg:
                continue
            for relation_and_tail in self.entity_kg[str(entity)]:
                edge_list.add((entity, relation_and_tail[1], relation_and_tail[0]))
                edge_list.add((relation_and_tail[1], entity, relation_and_tail[0]))
        return list(edge_list)

    def _build_self_loop_edges(self, self_loop_id):
        edge_list = []
        n_entity = len(self.entity2id)
        for entity in range(n_entity + 1):
            edge_list.append((entity, entity, self_loop_id))
            if str(entity) not in self.entity_kg:
                continue
            for relation_and_tail in self.entity_kg[str(entity)]:
                relation, tail = relation_and_tail
                if entity != tail and relation != self_loop_id:
                    edge_list.append((entity, tail, relation))
                    edge_list.append((tail, entity, relation))
        return edge_list

    @staticmethod
    def _filter_and_reindex_relations(edge_list, min_relation_count):
        relation_cnt = defaultdict(int)
        relation_idx = {}
        for _, _, relation in edge_list:
            relation_cnt[relation] += 1
        for _, _, relation in edge_list:
            if relation_cnt[relation] > min_relation_count and relation not in relation_idx:
                relation_idx[relation] = len(relation_idx) + 1
        edge_list = [
            (head, tail, relation_idx[relation])
            for head, tail, relation in edge_list
            if relation_cnt[relation] > min_relation_count
        ]
        num_relations = max(relation_idx.values(), default=-1) + 1
        return edge_list, num_relations

    def get_entity_kg_info(self):
        return {
            "edge_index": self.edge_index,
            "edge_type": self.edge_type,
            "num_entities": self.num_entities,
            "num_relations": self.num_relations,
            "pad_entity_id": self.pad_entity_id,
            "item_ids": self.item_ids,
        }


def inspired_kg(dataset, debug=False):
    return DBpedia(dataset=dataset, debug=debug, processing_config=KGProcessingConfig())


def redial_kg(dataset, debug=False):
    return DBpedia(
        dataset=dataset,
        debug=debug,
        processing_config=KGProcessingConfig(add_self_loop=True, self_loop_id=185, filter_min_relation_count=1000),
    )
