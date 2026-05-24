from src.common.kg_resources import DBpedia as _DBpedia
from src.common.kg_resources import KGProcessingConfig


class DBpedia(_DBpedia):
    def __init__(self, dataset, debug=False):
        super().__init__(
            dataset=dataset,
            debug=debug,
            processing_config=KGProcessingConfig(
                add_self_loop=True,
                self_loop_id=185,
                filter_min_relation_count=1000,
            ),
        )
