"""Test that KGNODE_OPENAI_MODEL is correctly loaded from .env file."""

import os
from dotenv import load_dotenv

from kgnode.core.kg_config import KGConfig


def test_openai_model_from_env():
    """Verify that KGConfig loads openai_model from KGNODE_OPENAI_MODEL env var."""
    # Load .env file (should load from project root)
    load_dotenv()

    # Get expected value from environment
    expected_model = os.getenv("KGNODE_OPENAI_MODEL")

    # Create config with defaults
    config = KGConfig.default()

    # Verify
    assert config.openai_model == expected_model, (
        f"Expected openai_model='{expected_model}', "
        f"but got '{config.openai_model}'"
    )

    print(f"✓ SUCCESS: KGConfig.openai_model = '{config.openai_model}'")
    print(f"✓ Loaded from KGNODE_OPENAI_MODEL environment variable")


if __name__ == "__main__":
    test_openai_model_from_env()
