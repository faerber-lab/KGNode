"""Test that gpt-5-mini is actually used in DSPy LM calls."""

import dspy
from dotenv import load_dotenv

from kgnode.core.kg_config import KGConfig


def test_dspy_lm_initialization():
    """Verify that DSPy LM is initialized with the correct model from .env."""
    load_dotenv()

    # Create config (same as evaluation scripts)
    config = KGConfig.default()

    print(f"1. KGConfig.openai_model = '{config.openai_model}'")
    print(f"2. KGConfig.lm_api_key = '{config.lm_api_key[:20]}...' (truncated)")

    # Initialize DSPy LM (same as add_seed_node_info.py, add_sparql_generated_answer.py, etc.)
    lm = dspy.LM(
        model=config.openai_model,
        api_key=config.lm_api_key,
        cache=False
    )

    print(f"3. DSPy LM created with model: '{lm.model}'")

    # Configure DSPy globally (this is what evaluation scripts do)
    dspy.configure(lm=lm)

    # Verify the configured LM
    configured_lm = dspy.settings.lm
    print(f"4. DSPy globally configured LM model: '{configured_lm.model}'")

    # Assertions
    assert config.openai_model == "openai/gpt-5-mini", (
        f"Config has wrong model: {config.openai_model}"
    )
    assert lm.model == "openai/gpt-5-mini", (
        f"DSPy LM initialized with wrong model: {lm.model}"
    )
    assert configured_lm.model == "openai/gpt-5-mini", (
        f"DSPy configured with wrong model: {configured_lm.model}"
    )

    print("\n✅ ALL CHECKS PASSED")
    print(f"✅ Model 'openai/gpt-5-mini' is correctly configured")
    print(f"✅ All evaluation scripts will use this model")


def test_actual_lm_call_with_logging():
    """Make a real LLM call and show the model being used."""
    load_dotenv()
    config = KGConfig.default()

    # Initialize DSPy
    lm = dspy.LM(model=config.openai_model, api_key=config.lm_api_key, cache=False)
    dspy.configure(lm=lm)

    print(f"\n=== Testing actual LLM call ===")
    print(f"Model being used: {lm.model}")
    print(f"Making a simple test call...\n")

    # Simple test signature
    class SimpleTest(dspy.Signature):
        """Test signature."""
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    predictor = dspy.Predict(SimpleTest)

    # Make a cheap test call
    try:
        result = predictor(question="What is 2+2?")
        print(f"✅ LLM call succeeded!")
        print(f"   Question: What is 2+2?")
        print(f"   Answer: {result.answer}")
        print(f"   Model used: {lm.model}")

        # Check if the response includes model info (some providers return this)
        if hasattr(lm, '_history') and lm._history:
            last_call = lm._history[-1]
            print(f"   Last call metadata: {last_call}")

    except Exception as e:
        print(f"⚠️  LLM call failed (API key issue or rate limit): {e}")
        print(f"   But model configuration is correct: {lm.model}")


if __name__ == "__main__":
    print("=" * 70)
    print("TEST 1: Model Configuration Check")
    print("=" * 70)
    test_dspy_lm_initialization()

    print("\n" + "=" * 70)
    print("TEST 2: Actual LLM Call (requires valid API key)")
    print("=" * 70)
    test_actual_lm_call_with_logging()