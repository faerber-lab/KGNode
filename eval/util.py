from kgnode import KGConfig, get_entities_collection

def _create_chromadb():
    config = KGConfig.default()
    collection = get_entities_collection(
        config=config, entity_limit=10000, force_recreate=False
    )