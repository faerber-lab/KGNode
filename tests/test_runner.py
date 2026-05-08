"""Enhanced test runner for kgnode test suite."""

import unittest
import sys
import time
import os
import argparse
import glob
import requests
from io import StringIO
from dotenv import load_dotenv

# Import global test fixtures
from fixtures import setup_global_fixtures, cleanup_global_fixtures


class EnhancedTestResult(unittest.TextTestResult):
    """Custom test result class to track failures by function."""

    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.test_results = {}  # {module: [(test_name, status, error), ...]}
        self.current_module = None

    def startTest(self, test):
        """Track which test is starting."""
        super().startTest(test)
        module_name = test.__class__.__module__.split(".")[-1]
        test_name = test._testMethodName

        if module_name not in self.test_results:
            self.test_results[module_name] = []

        self.current_module = module_name
        self.current_test = test_name

    def addSuccess(self, test):
        """Record successful test."""
        super().addSuccess(test)
        module_name = test.__class__.__module__.split(".")[-1]
        test_name = test._testMethodName
        func_name = self._extract_function_name(test)
        self.test_results[module_name].append((func_name, "success", None))

    def addError(self, test, err):
        """Record test error."""
        super().addError(test, err)
        module_name = test.__class__.__module__.split(".")[-1]
        test_name = test._testMethodName
        func_name = self._extract_function_name(test)
        self.test_results[module_name].append((func_name, "error", err))

    def addFailure(self, test, err):
        """Record test failure."""
        super().addFailure(test, err)
        module_name = test.__class__.__module__.split(".")[-1]
        test_name = test._testMethodName
        func_name = self._extract_function_name(test)
        self.test_results[module_name].append((func_name, "failure", err))

    def _extract_function_name(self, test):
        """Extract the actual function being tested from test name."""
        test_name = test._testMethodName
        doc = test.shortDescription()

        # Try to extract function name from docstring
        if doc and "Test" in doc:
            # Remove "Test " prefix and extract function name
            parts = doc.replace("Test ", "").split()
            if len(parts) > 0:
                func = parts[0]
                # Remove trailing punctuation
                func = func.rstrip("().,;:")
                return func

        # Fallback: extract from test name (test_function_name_xxx -> function_name)
        if test_name.startswith("test_"):
            parts = test_name[5:].split("_")
            # Find common function patterns
            if "get_seed_nodes" in test_name:
                return "get_seed_nodes()"
            elif "citable" in test_name:
                return "citable()"
            elif "get_subgraphs" in test_name:
                return "get_subgraphs()"
            elif "generate_sparql" in test_name:
                return "generate_sparql()"
            elif "kg_retrieve" in test_name:
                return "kg_retrieve()"
            elif "generate_answer_using_subgraph" in test_name:
                return "generate_answer_using_subgraph()"
            elif "generate_answer" in test_name:
                return "generate_answer()"
            elif "validate_subgraph" in test_name:
                return "validate_subgraph()"
            elif "search_entities_by_keywords" in test_name:
                return "search_entities_by_keywords()"
            elif "compile_entities_chromadb_from_csv" in test_name:
                return "compile_entities_chromadb_from_csv()"
            elif "compile_entities_chromadb" in test_name:
                return "compile_entities_chromadb()"
            elif "semantic_search_entities" in test_name:
                return "semantic_search_entities()"
            elif "get_entities_collection" in test_name:
                return "get_entities_collection()"
            elif "add_or_update_entities" in test_name:
                return "add_or_update_entities()"
            elif "delete_entities" in test_name:
                return "delete_entities()"
            elif "execute_sparql_query" in test_name:
                return "execute_sparql_query()"
            elif "describe_entities_batch" in test_name:
                return "KGConfig.describe_entities_batch()"
            elif "describe_entity" in test_name:
                return "KGConfig.describe_entity()"
            elif "describe_relation" in test_name:
                return "KGConfig.describe_relation()"
            elif "default_config" in test_name:
                return "KGConfig.default()"
            elif "custom_config" in test_name:
                return "KGConfig.__init__()"

        return test_name


class EnhancedTestRunner(unittest.TextTestRunner):
    """Custom test runner with enhanced output."""

    resultclass = EnhancedTestResult

    def run(self, test):
        """Run tests and display enhanced output."""
        result = super().run(test)
        return result


def check_prerequisites():
    """Check if required services are available."""
    issues = []

    # Check SPARQL endpoint
    sparql_endpoint = os.getenv("KGNODE_SPARQL_ENDPOINT", "http://localhost:7878/query")
    try:
        response = requests.get(sparql_endpoint.replace("/query", ""), timeout=2)
        if response.status_code != 200:
            issues.append(f"✗ Oxigraph server not responding at {sparql_endpoint}")
    except requests.exceptions.RequestException:
        issues.append(
            f"✗ Cannot connect to Oxigraph server at {sparql_endpoint}. "
            "Please start: oxigraph_server serve -l ./oxigraph_db"
        )

    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        issues.append("✗ OPENAI_API_KEY environment variable not set")

    # Check ChromaDB directory
    chroma_dir = "_data/vector_db"
    if not os.path.exists(chroma_dir):
        try:
            os.makedirs(chroma_dir, exist_ok=True)
        except Exception as e:
            issues.append(f"✗ Cannot create ChromaDB directory: {e}")

    return issues


def print_enhanced_results(result, duration):
    """Print enhanced test results."""
    print("\n" + "━" * 70)
    print("TEST RESULTS")
    print("━" * 70)

    # Module order (by dependency)
    module_order = [
        "test_config",
        "test_sparql_query",
        "test_keyword_search",
        "test_chromadb",
        "test_seed_finder",
        "test_subgraph_extraction",
        "test_validator",
        "test_generator",
    ]

    # Print results by module
    total_passed = 0
    total_failed = 0
    failed_functions = []

    for module in module_order:
        if module not in result.test_results:
            continue

        tests = result.test_results[module]
        passed = sum(1 for _, status, _ in tests if status == "success")
        failed = sum(1 for _, status, _ in tests if status != "success")

        total_passed += passed
        total_failed += failed

        # Module header
        print(f"\nMODULE: {module}.py")
        print("━" * 70)

        # Print each test
        for func_name, status, error in tests:
            if status == "success":
                print(f"  ✓ {func_name}")
            else:
                print(f"  ✗ {func_name}")
                failed_functions.append((module, func_name, error))

    # Print failures section if any
    if failed_functions:
        print("\n" + "━" * 70)
        print(f"FAILURES ({len(failed_functions)})")
        print("━" * 70)

        for i, (module, func_name, error) in enumerate(failed_functions, 1):
            print(f"\n[{i}] {module}.py :: {func_name}")

            # Format error message
            exc_type, exc_value, exc_tb = error
            print(f"    {exc_type.__name__}: {exc_value}")
            print()

            # Print traceback (last 3 frames)
            import traceback

            tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
            # Print relevant traceback lines
            print("    Traceback:")
            for line in tb_lines[1:]:  # Skip first line (Traceback header)
                for sub_line in line.rstrip().split("\n"):
                    print(f"      {sub_line}")

    # Print summary
    print("\n" + "━" * 70)
    print("SUMMARY")
    print("━" * 70)

    total_tests = total_passed + total_failed
    pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

    print(f"Total Tests:     {total_tests}")
    print(f"Passed:          {total_passed} ({pass_rate:.1f}%)")
    print(f"Failed:          {total_failed}  ({100-pass_rate:.1f}%)")
    print(f"Duration:        {duration:.1f}s")

    if failed_functions:
        print(f"\nFailed Functions:")
        for module, func_name, _ in failed_functions:
            print(f"  • {func_name}")

    print()


def get_available_tests():
    """Get list of available test files.

    Returns:
        List[str]: List of test file names (without test_ prefix and .py extension)
    """
    test_dir = os.path.dirname(__file__)
    test_files = glob.glob(os.path.join(test_dir, "test_*.py"))

    # Extract just the names (e.g., "test_config.py" -> "config")
    test_names = []
    for f in test_files:
        basename = os.path.basename(f)
        if basename == "test_runner.py":
            continue  # Skip the runner itself
        # Remove "test_" prefix and ".py" suffix
        name = basename[5:-3] if basename.startswith("test_") else basename[:-3]
        test_names.append(name)

    return sorted(test_names)


def normalize_test_name(name):
    """Normalize test name to standard format.

    Args:
        name: Can be "chromadb", "test_chromadb", or "test_chromadb.py"

    Returns:
        str: Normalized name without prefix/suffix (e.g., "chromadb")
    """
    # Remove .py extension
    if name.endswith(".py"):
        name = name[:-3]

    # Remove test_ prefix
    if name.startswith("test_"):
        name = name[5:]

    return name


def load_selected_tests(test_names, test_dir):
    """Load test suite for specific test files.

    Args:
        test_names: List of test names to load, or None/empty for all tests
        test_dir: Directory containing test files

    Returns:
        unittest.TestSuite: Test suite with selected tests
    """
    loader = unittest.TestLoader()

    if not test_names:
        # Load all tests
        return loader.discover(test_dir, pattern="test_*.py", top_level_dir=test_dir)

    # Load only specified tests
    suite = unittest.TestSuite()
    available_tests = get_available_tests()

    for name in test_names:
        normalized = normalize_test_name(name)

        if normalized not in available_tests:
            print(f"Error: Test '{name}' not found.")
            print(f"\nAvailable tests:")
            for t in available_tests:
                print(f"  {t}")
            sys.exit(1)

        # Load the specific test file
        test_file = f"test_{normalized}"
        try:
            tests = loader.loadTestsFromName(test_file)
            suite.addTests(tests)
        except Exception as e:
            print(f"Error loading test '{test_file}': {e}")
            sys.exit(1)

    return suite


def main():
    """Main test runner with command-line argument support."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run kgnode test suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                      Run all tests
  %(prog)s chromadb             Run test_chromadb.py
  %(prog)s chromadb seed        Run test_chromadb.py and test_seed_finder.py
  %(prog)s --list               List available tests
        """,
    )
    parser.add_argument(
        "tests",
        nargs="*",
        help="Test files to run (without test_ prefix or .py extension)",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available tests and exit",
    )

    args = parser.parse_args()

    # Load environment variables from .env file
    load_dotenv()

    # Handle --list flag
    if args.list:
        print("Available tests:")
        for test_name in get_available_tests():
            print(f"  {test_name}")
        print(f"\nUsage: python {sys.argv[0]} [test_name ...]")
        sys.exit(0)

    # Determine which tests to run
    test_names = args.tests if args.tests else None
    test_description = (
        f"tests: {', '.join(args.tests)}" if test_names else "all tests"
    )

    print(f"Running kgnode test suite ({test_description})...")
    print()

    # Check prerequisites
    issues = check_prerequisites()
    if issues:
        print("Prerequisites check:")
        for issue in issues:
            print(f"  {issue}")
        print("\nWARNING: Some prerequisites are missing. Tests may fail.\n")

    # Setup global fixtures ONCE before all tests
    setup_global_fixtures()

    try:
        # Load selected tests
        start_dir = os.path.dirname(__file__)
        suite = load_selected_tests(test_names, start_dir)

        # Run with custom runner
        runner = EnhancedTestRunner(verbosity=0, stream=StringIO())

        start_time = time.time()
        result = runner.run(suite)
        duration = time.time() - start_time

        # Print enhanced results
        print_enhanced_results(result, duration)

        # Exit with appropriate code
        exit_code = 0 if result.wasSuccessful() else 1

    finally:
        # Cleanup global fixtures ONCE after all tests
        cleanup_global_fixtures()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
