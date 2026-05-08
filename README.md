# kgnode

**Training-Free Subgraph Extraction for Knowledge-Grounded Question Answering**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

kgnode is a Python library that extracts relevant subgraphs from large knowledge graphs using a **path-aware Markov chain traversal algorithm** for question answering tasks. Unlike traditional approaches that require KG-specific training (entity linkers, KG embeddings), kgnode achieves competitive performance through:

1. **Hybrid Seed Discovery**: Combines semantic search (ChromaDB) and keyword search (SPARQL) with type-aware filtering
2. **Path-Aware Traversal**: Priority-queue BFS with exponential probability scoring: `P ∝ exp(cos(path, template))`
3. **Adaptive Stopping**: Quality-based termination that monitors probability distribution
4. **Training-Free**: No KG-specific fine-tuning required - works across different knowledge graphs

## Performance

Evaluated on two benchmarks without domain-specific training:

| Dataset                           | Model       | End-to-End Accuracy | Seed Discovery | Entity Coverage | Relation Coverage |
|-----------------------------------|-------------|---------------------|----------------|-----------------|-------------------|
| DBLP-QuAD (252M triples)          | gpt-4o-mini | 60.0%               | 92.5%          | 85.3%           | 52.7%             |
| DBLP-QuAD                         | gpt-5-mini  | **72.5%**           | **95.2%**      | **91.9%**       | **60.3%**         |
| QALD-10 (Wikidata, 1.65B triples) | gpt-4o-mini | 53.0%               | 87.3%          | 78.1%           | 44.9%             |
| QALD-10                           | gpt-5-mini  | 66.5%               | 90.0%          | 85.0%           | 50.0%             |

**Cross-domain transferability**: 7 percentage point performance drop (DBLP→QALD) with consistent degradation across stages, validating the training-free approach.

## Installation

```bash
pip install kgnode
```

## Quick Start

```python
from kgnode import KGConfig, generate_answer

# Configure for your knowledge graph
config = KGConfig(
    sparql_endpoint="http://localhost:7878/query",
    embedding_model="all-MiniLM-L6-v2",
    openai_model="gpt-4o-mini"
)

# End-to-end question answering
answer = generate_answer(
    query="Did Kamil Zbikowski and Michal Ostapowicz co-author a paper?",
    config=config
)
print(answer)
```

### Step-by-Step Pipeline

```python
from kgnode import get_seed_nodes, get_subgraphs, kg_retrieve

# 1. Find seed nodes (hybrid search: semantic + keyword)
seed_nodes, extracted_entities = get_seed_nodes(
    query="What papers did John Smith publish?",
    config=config
)
# Returns: Tuple of (seed_nodes list, extracted_entities list)

# 2. Extract relevant subgraphs using path-aware traversal
# Note: get_subgraphs processes one seed at a time, so we loop through all seeds
all_subgraphs = []
for seed_node in seed_nodes:
    subgraphs, template_text = get_subgraphs(
        seed_node=seed_node['entity_uri'],
        query="What papers did John Smith publish?",
        config=config,
        seed_nodes=seed_nodes  # Optional: provides context for template generation
    )
    all_subgraphs.extend(subgraphs)
# Returns: List of all subgraphs from all seeds with probability scores

# 3. Full pipeline: query → seed discovery → subgraph → SPARQL → answer
result = kg_retrieve(query="What papers did John Smith publish?", config=config)
```

## Key Features

### 1. Hybrid Seed Discovery
- **Semantic search**: ChromaDB vector database with `all-MiniLM-L6-v2` embeddings
- **Keyword search**: SPARQL text matching with name variations
- **Type-aware filtering**: Fuzzy string matching with adaptive thresholds
- **Performance**: 92-95% accuracy on seed discovery

### 2. Path-Aware Markov Chain Traversal
- **Exponential probability scoring**: Amplifies differences by 7× ratio
- **Template-guided**: LLM generates query templates for semantic alignment
- **Cycle detection**: Prevents infinite loops
- **Deduplication**: Removes redundant subgraphs
- **Adaptive stopping**: Quality-based termination (default: min 3, max 15 subgraphs)

### 3. Training-Free Architecture
- Works across different knowledge graphs (DBLP, Wikidata, etc.)
- No entity linking training required
- No KG embedding training required
- Leverages pre-trained sentence transformers

### 4. Performance Optimizations
- **LRU cache**: 5000-entry path embedding cache (3-5× speedup)
- **Parallel queries**: Concurrent neighbor retrieval
- **Batch encoding**: Efficient embedding computation
- **Schema-aware**: Grounds templates in actual KG vocabulary

## Folder Structure

```
kgnode/
├── src/kgnode/
│   ├── __init__.py              # Public API exports
│   ├── seed_finder.py           # Hybrid seed discovery
│   ├── subgraph_extraction.py   # Path-aware Markov chain algorithm
│   ├── generator.py             # SPARQL and answer generation
│   ├── validator.py             # Subgraph validation
│   ├── keyword_search.py        # SPARQL keyword search
│   ├── chroma_db.py             # ChromaDB vector operations
│   └── core/
│       ├── kg_config.py         # Configuration class
│       ├── sparql_query.py      # SPARQL endpoint communication
│       ├── schema_extractor.py  # Schema extraction
│       ├── schema_chromadb.py   # Schema vector DB
│       └── schema_selector.py   # Query-aware schema selection
├── tests/                        # Unit tests
└── eval/                         # Evaluation scripts
```

## Prerequisites

### 1. SPARQL Endpoint (Required)

kgnode requires a SPARQL endpoint. We recommend **Oxigraph**:

```bash
# Install Oxigraph
# macOS: brew install oxigraph
# Linux: cargo install oxigraph_server
# Windows: Download from https://github.com/oxigraph/oxigraph/releases

# Start server (read-write mode)
oxigraph_server serve -l ./oxigraph_db --cors

# Start server (read-only mode)
oxigraph_server serve-read-only -l ./oxigraph_db --cors

# Load dataset (one-time setup)
oxigraph_server load -l ./oxigraph_db -f _data/dblp.nt

# Custom bind address
oxigraph_server serve -l ~/oxigraph_db --bind 127.0.0.1:7878
```

**Default endpoint:** `http://localhost:7878/query`

### 2. OpenAI API Key (Required for LLM operations)

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 3. ChromaDB (Auto-created on first run)

Vector database for entity and schema embeddings is created automatically.

### 4. Configuration

```python
from kgnode import KGConfig, execute_sparql_query

# Create configuration with custom parameters
config = KGConfig(
    sparql_endpoint="http://localhost:7878/query",
    embedding_model="all-MiniLM-L6-v2",
    openai_model="gpt-4o-mini",
    min_subgraphs=3,              # Adaptive stopping: minimum
    max_subgraphs=15,             # Adaptive stopping: maximum
    quality_threshold_ratio=0.65, # Adaptive stopping: quality threshold
    absolute_prob_threshold=1.5,  # Minimum probability cutoff
)

# Execute SPARQL queries directly
results = execute_sparql_query(
    query="SELECT * WHERE { ?s ?p ?o } LIMIT 10",
    config=config
)
```

## Configuration Options

### Adaptive Stopping Parameters

**Default (Balanced)**:
```python
config = KGConfig(
    min_subgraphs=3,              # Minimum subgraphs to collect
    max_subgraphs=15,             # Maximum hard cap
    quality_threshold_ratio=0.65, # Stop if next prob < 0.65 × median
    absolute_prob_threshold=1.5   # Stop if next prob < 1.5 (exp(0.4))
)
```

**Aggressive (Fewer subgraphs, faster)**:
```python
config = KGConfig(
    min_subgraphs=2,
    max_subgraphs=10,
    quality_threshold_ratio=0.75,
    absolute_prob_threshold=2.0
)
```

**Conservative (More subgraphs, higher coverage)**:
```python
config = KGConfig(
    min_subgraphs=5,
    max_subgraphs=20,
    quality_threshold_ratio=0.4,
    absolute_prob_threshold=1.0
)
```

**Disable Adaptive Stopping**:
```python
config = KGConfig(
    min_subgraphs=25,
    max_subgraphs=25  # Just use hard limit
)
```

### Logging Configuration

**Option 1: Environment Variable**
```bash
# Show debug messages
export KGNODE_LOG_LEVEL=DEBUG
python your_script.py

# Only warnings and errors
export KGNODE_LOG_LEVEL=WARNING
python your_script.py

# Completely silent
export KGNODE_LOG_LEVEL=CRITICAL
python your_script.py
```

**Option 2: In Code**
```python
from kgnode.core import set_log_level, disable_logging

# Show debug messages
set_log_level("DEBUG")

# Only warnings and errors
set_log_level("WARNING")

# Completely silent
disable_logging()
```

## Datasets

### DBLP-QuAD (Primary benchmark)
- **Domain**: Academic publications knowledge graph
- **Source**: https://dblp.org/rdf/
- **Download**: https://zenodo.org/records/7638511
- **Paper**: [DBLP-QuAD (ECIR 2023)](https://www.inf.uni-hamburg.de/en/inst/ab/lt/publications/2023-banerjee-bir-ecir-2023-dblpquad.pdf)
- **Stats**: 252M triples, 92M entities, 62 relations
- **Evaluation**: 400 questions across 10 query types

### QALD-10 (Cross-domain benchmark)
- **Domain**: General knowledge (Wikidata)
- **Stats**: 1.65B triples, 120M entities
- **Evaluation**: 394 test questions
- **Paper**: [QALD-10 (Semantic Web 2024)](https://doi.org/10.3233/SW-233471)

## Testing

### Run All Tests
```bash
python tests/test_runner.py
```

### Run Specific Tests
```bash
# Run single test file
python tests/test_runner.py chromadb

# Run multiple test files
python tests/test_runner.py chromadb seed_finder subgraph_extraction

# List available tests
python tests/test_runner.py --list

# Run standalone test file
python tests/test_chromadb.py
```

### Prerequisites for Testing
- Oxigraph SPARQL server running at `http://localhost:7878/query`
- `OPENAI_API_KEY` environment variable set
- ChromaDB created (happens automatically on first run)

## Development

### Building from Source

```bash
# Clone the repository
git clone <repository-url>
cd kgnode

# Install dependencies using uv
uv sync

# Run tests
python tests/test_runner.py
```

### Building Package for PyPI

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build distribution packages (source + wheel)
uv build

# Check the built packages
twine check dist/*
```

The `uv build` command creates:
- `dist/kgnode-{version}.tar.gz` - Source distribution
- `dist/kgnode-{version}-py3-none-any.whl` - Wheel distribution

## Documentation

For detailed usage, API reference, and examples:
- **Usage Guide**: [docs/USAGE.md](docs/USAGE.md)
- **Research Paper**: See `paper/` directory for the academic paper with full methodology

## Supported Technologies

### Vector Databases
- **ChromaDB** ✅ (implemented)
- Pinecone (planned)
- Qdrant (planned)

### Embedding Models
- **all-MiniLM-L6-v2** ✅ (default, 384 dimensions)
- google/embeddinggemma-300m (alternative)

### LLM Backends
- **GPT-4o-mini** ✅ (default, cost-effective)
- **GPT-5-mini** ✅ (highest performance)

## License

MIT License - see [LICENSE](LICENSE) file for details

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## Acknowledgments

We acknowledge support from NHR Verein for this work.
