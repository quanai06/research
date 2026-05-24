from dataclasses import dataclass


@dataclass(frozen=True)
class ModalityConfig:
    use_kg_graph: bool = True
    use_collaborative_graph: bool = True
    use_text_graph: bool = True
    use_image_graph: bool = True
    add_item_semantic_to_entities: bool = True


def add_modality_args(parser):
    parser.add_argument("--disable_kg_graph", action="store_true")
    parser.add_argument("--disable_collaborative_graph", action="store_true")
    parser.add_argument("--disable_text_graph", action="store_true")
    parser.add_argument("--disable_image_graph", action="store_true")
    parser.add_argument("--disable_item_semantic_add", action="store_true")
    return parser


def modality_config_from_args(args, *, add_item_semantic_to_entities: bool = True) -> ModalityConfig:
    return ModalityConfig(
        use_kg_graph=not args.disable_kg_graph,
        use_collaborative_graph=not args.disable_collaborative_graph,
        use_text_graph=not args.disable_text_graph,
        use_image_graph=not args.disable_image_graph,
        add_item_semantic_to_entities=(
            add_item_semantic_to_entities and not args.disable_item_semantic_add
        ),
    )
