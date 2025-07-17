# 🔮 auto-embed

<div align="center">
<img src="logo/auto-embed-logo.png" width="256"/>

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Poetry](https://img.shields.io/badge/Poetry-dependency%20management-blue.svg)](https://python-poetry.org/)

*Effortlessly transform your business entities into powerful vector embeddings* ✨

</div>

## 🚀 What is auto-embed?

**auto-embed** is a lightweight, production-ready library that transforms your business entities into mathematical representations. Enable similarity searches, vector operations, and AI-powered insights with just a few lines of code.

## ✨ Features

- 🎯 **Simple Configuration** - YAML-based setup for quick deployment
- 🏠 **Local-First** - ChromaDB integration for vector storage
- 📊 **CSV Support** - Direct filesystem operations
- 🧠 **Deep Learning** - Keras-based autoencoder models
- 🔍 **Similarity Search** - Find related entities instantly
- 📈 **Visualization** - Interactive embeddings exploration
- 🛠️ **CLI & API** - Multiple interfaces for different workflows

## 🏃‍♂️ Quick Start

### Installation

```bash
# Install dependencies
make install-project

# For development
make install-project-dev
```

### Basic Usage

1. **Configure your pipeline** (`config.yaml`):
```yaml
model_name: my_model
id_column: entity_id
vector_store:
  vector_collection_name: my_embeddings
data:
  training:
    type: csv
    path: data/training.csv
```

2. **Train your model**:
```bash
autoembed-cli train --yaml_path config.yaml
```

3. **Generate predictions**:
```bash
autoembed-cli predict --yaml_path config.yaml
```

4. **Find similar entities**:
```bash
autoembed-cli what-is-my-recommendation --id "entity_123"
```

## 🛠️ Available Commands

| Command | Description | Emoji |
|---------|-------------|-------|
| `make run-train-classified-embedding-model-cli` | Train embedding model | 🧠 |
| `make run-predict-model-cli` | Generate predictions | 🔮 |
| `make run-what-is-my-classified-recommendation-cli` | Find recommendations | 💡 |
| `make run-tests` | Run test suite | 🧪 |
| `make run-lint` | Code quality checks | ✅ |

## 🏗️ Architecture

```
📦 auto-embed
├── 🎯 CLI Interface     # Command-line tools
├── 🌐 API Server       # FastAPI endpoints  
├── 🧠 Domain Logic     # Business rules
├── 🏗️ Infrastructure   # ChromaDB, Keras adapters
└── ⚙️ Use Cases        # Training, prediction, queries
```

## 🤝 Contributing

We welcome contributions! Please feel free to:

- 🐛 Report bugs
- 💡 Suggest features
- 📝 Improve documentation
- 🔧 Submit pull requests

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with ❤️ using:
- [TensorFlow/Keras](https://tensorflow.org) for deep learning
- [ChromaDB](https://chromadb.com) for vector storage
- [FastAPI](https://fastapi.tiangolo.com) for API endpoints
- [Poetry](https://python-poetry.org) for dependency management

