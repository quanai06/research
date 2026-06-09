from src.common.kg_resources import DBpedia as _DBpedia
from src.common.kg_resources import KGProcessingConfig


class DBpedia(_DBpedia):
    def __init__(self, dataset, debug=False):
        super().__init__(
            dataset=dataset,
            debug=debug,
            processing_config=KGProcessingConfig(),
        )
