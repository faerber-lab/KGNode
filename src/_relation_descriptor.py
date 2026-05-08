"""Relation descriptor logic - create rich descriptions for KG relations/predicates.

Optimized for high cosine similarity with LLM-extracted relation phrases.
"""

import re
from typing import Optional, List

from kgnode._chromadb_shared import uri_to_label
from kgnode.core.kg_config import KGConfig
from kgnode.core.logging_config import get_logger
from kgnode.core.sparql_query import execute_sparql_query


logger = get_logger(__name__)


def _default_relation_descriptor_logic(relation_uri: str) -> str:
    """Default relation descriptor logic.

    Extract the last part of relation URI (e.g., 'authoredBy' from full URI).

    Args:
        relation_uri: URI of the relation/predicate.

    Returns:
        Predicate name extracted from URI.
    """
    clean_uri = relation_uri.strip("<>")

    if "#" in clean_uri:
        return clean_uri.split("#")[-1]
    elif "/" in clean_uri:
        return clean_uri.split("/")[-1]
    else:
        return clean_uri


def _split_camel_case(text: str) -> List[str]:
    """Split camelCase or PascalCase into words.

    Examples:
        authoredBy → ['authored', 'by']
        yearOfPublication → ['year', 'of', 'publication']
        publishedIn → ['published', 'in']

    Args:
        text: camelCase or PascalCase string

    Returns:
        List of lowercase words
    """
    # Insert space before uppercase letters, then split
    spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    words = spaced.split()
    return [w.lower() for w in words]


def _generate_verb_variations(verb: str) -> List[str]:
    """Generate verb tense variations dynamically.

    Examples:
        publish → [publish, published, publishing]
        author → [author, authored, authoring]
        cite → [cite, cited, citing]

    Args:
        verb: Base verb form

    Returns:
        List of verb variations
    """
    variations = [verb]

    # Generate past tense (simple heuristic)
    if verb.endswith('e'):
        variations.append(verb + 'd')  # publish → published
    elif verb.endswith('y') and len(verb) > 2 and verb[-2] not in 'aeiou':
        variations.append(verb[:-1] + 'ied')  # study → studied
    else:
        variations.append(verb + 'ed')  # author → authored

    # Generate -ing form
    if verb.endswith('e') and len(verb) > 2:
        variations.append(verb[:-1] + 'ing')  # publish → publishing
    else:
        variations.append(verb + 'ing')  # author → authoring

    return variations


def _get_verb_base_form(word: str) -> str:
    """Get base form of a verb from past/present participle.

    Examples:
        authored → author
        published → publish
        cited → cite
        creating → create

    Args:
        word: Verb in past tense or present participle

    Returns:
        Base form of verb
    """
    # Handle -ed ending (past tense)
    if word.endswith('ed'):
        # doubled consonant: submitted → submit
        if len(word) > 4 and word[-3] == word[-4] and word[-4] not in 'aeiou':
            return word[:-3]
        # regular: published → publish, authored → author
        elif word.endswith('hed') or word.endswith('red') or word.endswith('ted'):
            return word[:-2]
        else:
            return word[:-1] if word.endswith('ed') else word[:-2]

    # Handle -ing ending (present participle)
    elif word.endswith('ing'):
        # doubling: running → run
        if len(word) > 4 and word[-4] == word[-5]:
            return word[:-4]
        # regular: publishing → publish
        else:
            return word[:-3]

    return word


def _generate_phrase_variations(words: List[str]) -> List[str]:
    """Generate natural language phrase variations from word list.

    ONLY uses grammatical transformations - NO semantic/domain knowledge.

    Examples:
        ['authored', 'by'] → ['authored by', 'authored', 'by', 'author']
        ['published', 'in'] → ['published in', 'published', 'in', 'publish']
        ['year', 'of', 'publication'] → ['year of publication', 'year', 'publication']

    Args:
        words: List of words (typically from splitting camelCase)

    Returns:
        List of phrase variations
    """
    phrases = []

    # Full phrase
    full_phrase = ' '.join(words)
    phrases.append(full_phrase)

    # Individual words (excluding small prepositions in middle positions)
    for i, word in enumerate(words):
        if word not in ['of', 'to', 'from'] or i == 0 or i == len(words) - 1:
            phrases.append(word)

    # Include prepositions when they appear (in, at, by)
    for word in words:
        if word in ['in', 'at', 'by']:
            phrases.append(word)

    # First word grammatical variations (if it looks like a verb)
    if words and len(words[0]) > 3:
        first_word = words[0]

        # Get base form if it's a verb
        base = _get_verb_base_form(first_word)
        if base != first_word:
            phrases.append(base)
            phrases.extend(_generate_verb_variations(base))

    # Generate prepositional phrases with prepositions that appear in the name
    prepositions = ['in', 'at', 'by', 'from', 'during']
    for prep in prepositions:
        if prep in words:
            # Generate "prep word" combinations
            for word in words:
                if word != prep and word not in ['of', 'to']:
                    phrases.append(f"{prep} {word}")

    return list(set(phrases))  # Remove duplicates


def _generate_contextual_terms(domain: Optional[str], range_type: Optional[str]) -> List[str]:
    """Generate contextual terms from domain and range types.

    ONLY extracts terms from actual KG metadata - NO hardcoding.

    Simply splits camelCase domain/range names into words.

    Args:
        domain: Domain type from KG (e.g., "Publication", "Creator")
        range_type: Range type from KG (e.g., "Creator", "Literal")

    Returns:
        List of contextual terms
    """
    terms = []

    # Process domain - just split the actual KG value
    if domain:
        domain_words = _split_camel_case(domain)
        terms.extend(domain_words)

    # Process range - just split the actual KG value
    if range_type:
        range_words = _split_camel_case(range_type)
        terms.extend(range_words)

    return terms


def _extract_terms_from_text(text: str) -> List[str]:
    """Extract meaningful terms from free-text labels/comments.

    Extracts individual words, filtering out noise words and punctuation.
    NO hardcoding - just splits text and filters common stop words.

    Args:
        text: Free-text string (label or comment from KG)

    Returns:
        List of extracted terms
    """
    if not text:
        return []

    # Convert to lowercase and split on whitespace/punctuation
    words = re.findall(r'\b[a-z]+\b', text.lower())

    # Filter out very common stop words (no domain-specific filtering)
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
                   'of', 'to', 'for', 'with', 'as', 'this', 'that', 'these', 'those'}

    meaningful_terms = [w for w in words if w not in stop_words and len(w) > 2]

    return meaningful_terms


def _generate_optimized_description(
    canonical_name: str,
    domain: Optional[str] = None,
    range_type: Optional[str] = None
) -> str:
    """Generate description optimized for cosine similarity with LLM extractions.

    Strategy:
    1. Start with canonical name (highest weight in embeddings)
    2. Add natural language variations from splitting camelCase
    3. Add verb variations for action words
    4. Add contextual terms from domain/range
    5. Use dense keyword packing (space-separated, no punctuation)

    Examples:
        authoredBy → "authoredBy authored by author creator publication"
        yearOfPublication → "yearOfPublication year of publication"
        publishedIn → "publishedIn published in publication"

    Args:
        canonical_name: Canonical relation name (e.g., "authoredBy")
        domain: Optional domain type
        range_type: Optional range type

    Returns:
        Optimized description string
    """
    terms = []

    # 1. Start with canonical name
    terms.append(canonical_name)

    # 2. Split camelCase and generate phrase variations
    words = _split_camel_case(canonical_name)
    phrase_variations = _generate_phrase_variations(words)
    terms.extend(phrase_variations)

    # 3. Add contextual terms from domain/range
    contextual_terms = _generate_contextual_terms(domain, range_type)
    terms.extend(contextual_terms)

    # 4. Remove duplicates while preserving order
    seen = set()
    unique_terms = []
    for term in terms:
        term_lower = term.lower()
        if term_lower not in seen:
            seen.add(term_lower)
            unique_terms.append(term)

    # 5. Join with spaces (dense keyword packing)
    return ' '.join(unique_terms)


def _generate_optimized_description_with_metadata(
    canonical_name: str,
    domain: Optional[str] = None,
    range_type: Optional[str] = None,
    labels: Optional[List[str]] = None,
    comments: Optional[List[str]] = None
) -> str:
    """Generate description with additional terms from KG metadata.

    Extends _generate_optimized_description by extracting terms from
    labels and comments available in the KG.

    Args:
        canonical_name: Canonical relation name (e.g., "authoredBy")
        domain: Optional domain type from KG
        range_type: Optional range type from KG
        labels: Optional list of labels from KG (rdfs:label, skos:prefLabel, etc.)
        comments: Optional list of comments from KG (rdfs:comment)

    Returns:
        Optimized description string with metadata terms
    """
    terms = []

    # 1. Start with canonical name
    terms.append(canonical_name)

    # 2. Split camelCase and generate phrase variations
    words = _split_camel_case(canonical_name)
    phrase_variations = _generate_phrase_variations(words)
    terms.extend(phrase_variations)

    # 3. Extract terms from labels (if available from KG)
    if labels:
        for label in labels:
            label_terms = _extract_terms_from_text(label)
            terms.extend(label_terms)

    # 4. Extract terms from comments (if available from KG)
    if comments:
        for comment in comments:
            # Only use first 100 chars to avoid noise
            comment_short = comment[:100] if len(comment) > 100 else comment
            comment_terms = _extract_terms_from_text(comment_short)
            terms.extend(comment_terms)

    # 5. Add contextual terms from domain/range
    contextual_terms = _generate_contextual_terms(domain, range_type)
    terms.extend(contextual_terms)

    # 6. Remove duplicates while preserving order
    seen = set()
    unique_terms = []
    for term in terms:
        term_lower = term.lower()
        if term_lower not in seen:
            seen.add(term_lower)
            unique_terms.append(term)

    # 7. Join with spaces (dense keyword packing)
    return ' '.join(unique_terms)


def create_relation_description(relation_uri: str, config: Optional[KGConfig] = None) -> str:
    """Create optimized natural language description for a relation/predicate.

    Generates descriptions optimized for HIGH COSINE SIMILARITY with LLM-extracted
    relation phrases. Uses dense keyword packing with natural language variations.

    Strategy:
    1. Query KG for ALL available metadata (label, comment, domain, range)
    2. Extract canonical name from URI
    3. Extract terms from labels and comments
    4. Generate grammatical variations (no hardcoding)
    5. Add contextual terms from domain/range
    6. Use dense keyword packing (space-separated, no noise words)

    Old format (LOW similarity):
        "ObjectProperty: authoredBy - connects Publication to Creator"

    New format (HIGH similarity):
        "authoredBy authored by author publication creator"

    Args:
        relation_uri: URI of the relation/predicate (without angle brackets)
        config: Optional KGConfig instance. If None, uses default.

    Returns:
        Natural language description optimized for semantic search and LLM matching
    """
    # Initialize config if not provided
    if config is None:
        config = KGConfig.default()

    # Remove angle brackets if present
    relation_uri = relation_uri.strip()
    if relation_uri.startswith('<') and relation_uri.endswith('>'):
        relation_uri = relation_uri[1:-1]

    # Query KG for ALL available metadata (including labels and comments)
    sparql_query = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

    SELECT DISTINCT ?predicate ?object
    WHERE {{
      <{relation_uri}> ?predicate ?object .
      FILTER(?predicate IN (
        rdf:type,
        rdfs:label,
        rdfs:comment,
        rdfs:domain,
        rdfs:range,
        skos:prefLabel,
        skos:altLabel
      ))
    }}
    """

    try:
        results = execute_sparql_query(sparql_query, config=config)
    except Exception:
        # Fallback: generate optimized description from URI label only
        canonical_name = _default_relation_descriptor_logic(relation_uri)
        return _generate_optimized_description(canonical_name, None, None)

    # Parse results - collect ALL labels and comments
    domain = None
    range_type = None
    labels = []
    comments = []

    for row in results:
        pred = row.get('predicate', '')
        obj = row.get('object', '')

        if 'domain' in pred:
            domain = uri_to_label(obj)
        elif 'range' in pred:
            range_type = uri_to_label(obj)
        elif 'label' in pred.lower():  # rdfs:label, skos:prefLabel, skos:altLabel
            labels.append(obj)
        elif 'comment' in pred:
            comments.append(obj)

    # Extract canonical name from URI
    canonical_name = _default_relation_descriptor_logic(relation_uri)

    # Generate optimized description with additional terms from labels/comments
    return _generate_optimized_description_with_metadata(
        canonical_name,
        domain,
        range_type,
        labels,
        comments
    )


if __name__ == "__main__":
    # Example usage
    print(create_relation_description("https://dblp.org/rdf/schema#authoredBy"))
    print(create_relation_description("http://www.w3.org/2000/01/rdf-schema#comment"))
