import json
import os
import socket
from urllib.error import URLError

from SPARQLWrapper import JSON, SPARQLWrapper

from kgnode.core.kg_config import KGConfig


def _execute_raw_sparql(query: str, timeout: int = 30) -> dict | None:
    """Run SPARQL query and return raw JSON response, or None on error."""
    config = KGConfig.default()
    sparql = SPARQLWrapper(config.sparql_endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(timeout)
    try:
        return sparql.query().convert()
    except (socket.timeout, URLError, TimeoutError):
        return None
    except Exception:
        return None


def _answer_from_raw(raw: dict | None, fallback: dict) -> list:
    """Convert raw SPARQL JSON to answer list format, falling back to stored data.

    Args:
        raw: Raw SPARQL JSON response from the endpoint.
        fallback: Stored answer dict from answers.json (used if raw is None/empty).

    Returns:
        Answer list in JSONL binding format.
    """
    if raw is not None:
        if "boolean" in raw:
            return [{"answer": {"type": "boolean", "value": raw["boolean"]}}]
        if "results" in raw:
            bindings = raw["results"].get("bindings", [])
            if bindings:
                return bindings

    # Fallback to stored answer
    if "boolean" in fallback:
        return [{"answer": {"type": "boolean", "value": fallback["boolean"]}}]
    if "results" in fallback:
        return fallback["results"].get("bindings", [])
    return []


def load_first_n_questions(path: str, count: int):
    """Load first N valid questions with answers, save to compact JSONL format.

    Validates SPARQL syntax for each question. Prints warnings for invalid queries.
    Executes golden SPARQL against the KG to get live answers (falls back to stored
    answers on failure). Saves to eval/data/first_{count}.jsonl.

    Args:
        path: Path to questions JSON file
        count: Number of valid questions to return

    Returns:
        List of first N questions with valid SPARQL syntax
    """
    # Determine paths
    questions_path = os.path.join(path, "questions.json")
    answers_path = os.path.join(path, "answers.json")

    # Load questions and answers
    with open(questions_path, "r", encoding="utf-8") as f:
        questions_data = json.load(f)

    with open(answers_path, "r", encoding="utf-8") as f:
        answers_data = json.load(f)

    # Create answer lookup
    answers_by_id = {a["id"]: a["answer"] for a in answers_data["answers"]}

    all_questions = questions_data["questions"]

    # Validate and filter questions
    valid_questions = []
    compact_data = []

    for question in all_questions:
        try:
            sparql_query = question.get("query", {}).get("sparql", "")
            is_valid, error_msg = validate_sparql_syntax(sparql_query)

            if is_valid:
                valid_questions.append(question)

                q_id = question["id"]
                raw = _execute_raw_sparql(sparql_query)
                answer = _answer_from_raw(raw, fallback=answers_by_id.get(q_id, {}))

                compact = {
                    "id": q_id,
                    "query_type": question.get("query_type"),
                    "question": {
                        "1": question.get("question", {}).get("string", ""),
                        "2": question.get("paraphrased_question", {}).get("string", "")
                    },
                    "golden_entities": question.get("entities", []),
                    "golden_relations": question.get("relations", []),
                    "golden_sparql": sparql_query,
                    "answer": answer
                }
                compact_data.append(compact)
            else:
                question_id = question.get("id", "UNKNOWN")
                print(
                    f"Warning: Question {question_id} has invalid SPARQL: "
                    f"{error_msg}"
                )

            # Stop once we have enough valid questions
            if len(valid_questions) >= count:
                break
        except Exception as e:
            question_id = question.get("id", "UNKNOWN")
            print(f"Warning: Question {question_id} validation error: {str(e)}")

    # Save to JSONL file
    output_dir = "./data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"first_{count}.jsonl")

    with open(output_path, "w", encoding="utf-8") as f:
        for item in compact_data[:count]:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(compact_data[:count])} questions to {output_path}")

    return valid_questions[:count]


def load_balanced_questions(path: str, total_count: int):
    """Load balanced questions across query types, save to compact JSONL format.

    Validates SPARQL syntax for each question. Prints warnings for invalid queries.
    Groups valid questions by query_type and returns balanced sample.
    Executes golden SPARQL against the KG to get live answers (falls back to stored
    answers on failure). Saves to ./data/balanced_{count}.jsonl.

    Args:
        path: Path to directory containing questions.json and answers.json
        total_count: Total number of valid questions to return

    Returns:
        List of balanced questions with valid SPARQL syntax
    """
    # Determine paths
    questions_path = os.path.join(path, "questions.json")
    answers_path = os.path.join(path, "answers.json")

    # Load questions and answers
    with open(questions_path, "r", encoding="utf-8") as f:
        questions_data = json.load(f)

    with open(answers_path, "r", encoding="utf-8") as f:
        answers_data = json.load(f)

    # Create answer lookup
    answers_by_id = {a["id"]: a["answer"] for a in answers_data["answers"]}

    all_questions = questions_data["questions"]

    # Validate questions first
    valid_questions = []
    for question in all_questions:
        try:
            sparql_query = question.get("query", {}).get("sparql", "")
            is_valid, error_msg = validate_sparql_syntax(sparql_query)

            if is_valid:
                valid_questions.append(question)
            else:
                question_id = question.get("id", "UNKNOWN")
                print(
                    f"Warning: Question {question_id} has invalid SPARQL: "
                    f"{error_msg}"
                )
        except Exception as e:
            question_id = question.get("id", "UNKNOWN")
            print(f"Warning: Question {question_id} validation error: {str(e)}")

    # Group valid questions by query type
    questions_by_type = {}
    for question in valid_questions:
        query_type = question.get("query_type", "unknown")
        if query_type not in questions_by_type:
            questions_by_type[query_type] = []
        questions_by_type[query_type].append(question)

    # Calculate samples per type
    num_types = len(questions_by_type)
    base_samples = total_count // num_types
    remainder = total_count % num_types

    # Sample from each type
    balanced_questions = []
    compact_data = []

    for i, (query_type, questions) in enumerate(sorted(questions_by_type.items())):
        samples_for_type = base_samples + (1 if i < remainder else 0)
        sampled = questions[:samples_for_type]
        balanced_questions.extend(sampled)

        for question in sampled:
            q_id = question["id"]
            sparql_query = question.get("query", {}).get("sparql", "")

            raw = _execute_raw_sparql(sparql_query)
            answer = _answer_from_raw(raw, fallback=answers_by_id.get(q_id, {}))

            compact = {
                "id": q_id,
                "query_type": question.get("query_type"),
                "question": {
                    "1": question.get("question", {}).get("string", ""),
                    "2": question.get("paraphrased_question", {}).get("string", "")
                },
                "golden_entities": question.get("entities", []),
                "golden_relations": question.get("relations", []),
                "golden_sparql": sparql_query,
                "answer": answer
            }
            compact_data.append(compact)

    # Save to JSONL file
    output_dir = "./data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"balanced_{total_count}.jsonl")

    with open(output_path, "w", encoding="utf-8") as f:
        for item in compact_data:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(compact_data)} balanced questions to {output_path}")
    print(f"Distribution: {[(k, len(v)) for k, v in sorted(questions_by_type.items())]}")

    return balanced_questions


def validate_sparql_syntax(sparql_query: str) -> tuple[bool, str]:
    """Validate SPARQL syntax using rdflib parser (syntax-only, no execution).

    Args:
        sparql_query: SPARQL query string to validate

    Returns:
        Tuple of (is_valid: bool, error_message: str)
        - is_valid: True if syntax is valid, False otherwise
        - error_message: Empty string if valid, error description if invalid
    """
    try:
        from rdflib.plugins.sparql.parser import parseQuery
    except ImportError:
        return False, "rdflib not installed. Install with: pip install rdflib"

    try:
        parseQuery(sparql_query)
        return True, ""
    except Exception as e:
        error_msg = str(e).split('\n')[0]  # First line of error
        return False, error_msg


if __name__ == "__main__":
    load_balanced_questions("../_data/dblp-quad/test", 400)
