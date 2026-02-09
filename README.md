# EVOCOMP

EVOCOMP is an Evolutionary Compiler Architecture for Scalable, Verifiable, and Economical Neural Model Composition.

## Description

EVOCOMP addresses the fundamental challenges in modern AI development: cost, safety, and composability. By treating model composition as a compilation pipeline, it achieves:
- **Cost reduction** via evolutionary model merging.
- **Formal safety verification** with cryptographic proof certificates using Z3.
- **Deterministic deployment** via consensus-verified compilation.
- **Chaos-resilient distributed evolution** with automatic rollback and health monitoring.

## Integration with AI-Scientist

This repository includes integration with an AI-Scientist agent, enabling a meta-scientific loop for automated hypothesis generation and experimental design. The system uses a `MetaScientistNiche` to evolve research strategies alongside model optimization.

## Project Structure

The core logic resides in the `src/` directory. Below is an overview of the modules:

### Core Modules (`src/`)

- **`backend.py`**: Implements the compilation backend, including passes for manifold operation fusion, sparse router fusion, and quantization.
- **`distributed_context.py`**: Manages the distributed evolutionary context, handling experiment broadcasting and result gathering across nodes using `torch.distributed` and Redis.
- **`economics.py`**: Provides utilities for estimating the economic cost of the evolutionary process, including discovery and verification costs.
- **`evolution.py`**: Contains the `HighDimEvolution` class, which implements advanced evolutionary strategies using K-FAC (Kronecker-Factored Approximate Curvature) and TRPO (Trust Region Policy Optimization).
- **`geometry.py`**: Implements `DeterministicGeometryExtractor` for extracting manifold geometry from model populations using randomized SVD.
- **`manifold.py`**: Defines the `EvoManifoldKernel` for embedding models into a lower-dimensional manifold using deterministic random projections, and `ManifoldTheory` for formal reasoning about manifold properties.
- **`meta.py`**: Integrates the `AIScientistAgent` and `AutomatedEvoComp` classes to drive the meta-scientific loop of hypothesis generation and experimentation.
- **`serving.py`**: Implements the `EthicalCircuitBreaker` for run-time model validation, ensuring models meet safety guards before serving.
- **`verification.py`**: Provides formal verification tools (`GuardVerifier`, `SafetySpec`) to mathematically prove safety properties of the evolved models using Z3.

### System Health (`src/evocomp/core/`)

- **`health.py`**: Contains the `HealthChecker` class, a background daemon that monitors system resources (CPU, Memory, Disk, GPU) to ensure the stability of the evolutionary process.

## Installation

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

**Note:** The system requires `torch`, `z3-solver`, `redis`, `psutil`, and `pynvml`. Ensure you have a Redis server running for distributed features.

## Usage

### Basic Evolution

To initialize the evolutionary context and start an experiment:

```python
from src.meta import AutomatedEvoComp

# Initialize the automated system (uses env vars WORLD_SIZE, RANK, REDIS_HOST)
auto_evo = AutomatedEvoComp()

# Start the meta-scientific loop
auto_evo.meta_scientific_loop(iterations=10)
```

### Formal Verification

To verify a safety specification against a manifold:

```python
from src.verification import GuardVerifier, SafetySpec
from src.manifold import EvoManifoldKernel

manifold = EvoManifoldKernel(dim=10)
verifier = GuardVerifier()
spec = SafetySpec(safe_bounds={0: (-1.0, 1.0)})

certificate = verifier.verify(manifold, spec)
if certificate.valid:
    print(f"Verified! Proof Hash: {certificate.proof_hash}")
else:
    print("Verification failed.")
```

## Key Features

- **K-FAC Optimization**: Uses second-order information to guide the evolutionary search.
- **Z3 Formal Verification**: Ensures that evolved models stay within safe bounds defined by `SafetySpec`.
- **Distributed Architecture**: Built on `torch.distributed` and Redis for scalable experiments.
- **Manifold Geometry**: Extracts and utilizes the geometric structure of the model population to inform search.
- **Health Monitoring**: Proactive system health checks to prevent failures during long-running evolution.

## License

(See LICENSE file)
