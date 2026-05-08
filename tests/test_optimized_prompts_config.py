import json
import os
import tempfile
import unittest

from kgnode.core.kg_config import (
    KGConfig,
    _DEFAULT_EXTRACTION_ARTIFACT,
    _DEFAULT_SPARQL_ARTIFACT,
    _load_instruction_from_artifact,
)


def _make_artifact(instructions: str) -> str:
    data = {"predict": {"signature": {"instructions": instructions}}}
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, f)
    f.close()
    return f.name


class TestLoadInstructionFromArtifact(unittest.TestCase):
    def test_reads_instructions_field(self):
        path = _make_artifact("hello world")
        try:
            self.assertEqual(_load_instruction_from_artifact(path), "hello world")
        finally:
            os.unlink(path)

    def test_default_artifacts_exist(self):
        self.assertTrue(os.path.isfile(_DEFAULT_EXTRACTION_ARTIFACT), _DEFAULT_EXTRACTION_ARTIFACT)
        self.assertTrue(os.path.isfile(_DEFAULT_SPARQL_ARTIFACT), _DEFAULT_SPARQL_ARTIFACT)

    def test_default_artifacts_have_instructions(self):
        for path in (_DEFAULT_EXTRACTION_ARTIFACT, _DEFAULT_SPARQL_ARTIFACT):
            instr = _load_instruction_from_artifact(path)
            self.assertIsInstance(instr, str)
            self.assertGreater(len(instr), 50)


class TestKGConfigOptimizedPrompts(unittest.TestCase):
    def test_flag_false_uses_defaults(self):
        config = KGConfig(use_dspy_optimized_prompts=False)
        self.assertEqual(config.entity_extraction_instruction, KGConfig.DEFAULT_ENTITY_EXTRACTION_INSTRUCTION)
        self.assertEqual(config.entity_relation_extraction_instruction, KGConfig.DEFAULT_ENTITY_RELATION_EXTRACTION_INSTRUCTION)
        self.assertEqual(config.sparql_generation_instruction, KGConfig.DEFAULT_SPARQL_GENERATION_INSTRUCTION)

    def test_flag_true_loads_default_artifacts(self):
        config = KGConfig(use_dspy_optimized_prompts=True)
        expected_extraction = _load_instruction_from_artifact(_DEFAULT_EXTRACTION_ARTIFACT)
        expected_sparql = _load_instruction_from_artifact(_DEFAULT_SPARQL_ARTIFACT)
        self.assertEqual(config.entity_extraction_instruction, expected_extraction)
        self.assertEqual(config.entity_relation_extraction_instruction, expected_extraction)
        self.assertEqual(config.sparql_generation_instruction, expected_sparql)

    def test_flag_true_with_custom_artifact_paths(self):
        extraction_path = _make_artifact("custom extraction instruction")
        sparql_path = _make_artifact("custom sparql instruction")
        try:
            config = KGConfig(
                use_dspy_optimized_prompts=True,
                extraction_artifact=extraction_path,
                sparql_artifact=sparql_path,
            )
            self.assertEqual(config.entity_extraction_instruction, "custom extraction instruction")
            self.assertEqual(config.entity_relation_extraction_instruction, "custom extraction instruction")
            self.assertEqual(config.sparql_generation_instruction, "custom sparql instruction")
        finally:
            os.unlink(extraction_path)
            os.unlink(sparql_path)

    def test_explicit_instruction_kwarg_always_wins(self):
        """Explicit instruction kwarg overrides the artifact even when flag is True."""
        config = KGConfig(
            use_dspy_optimized_prompts=True,
            entity_extraction_instruction="explicit extraction",
            entity_relation_extraction_instruction="explicit relation extraction",
            sparql_generation_instruction="explicit sparql",
        )
        self.assertEqual(config.entity_extraction_instruction, "explicit extraction")
        self.assertEqual(config.entity_relation_extraction_instruction, "explicit relation extraction")
        self.assertEqual(config.sparql_generation_instruction, "explicit sparql")

    def test_default_factory_method(self):
        config = KGConfig.default(use_dspy_optimized_prompts=True)
        self.assertEqual(
            config.entity_relation_extraction_instruction,
            _load_instruction_from_artifact(_DEFAULT_EXTRACTION_ARTIFACT),
        )


if __name__ == "__main__":
    unittest.main()
