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

## TODO

- Add support for textual column with little transformer
- Add support for monitoring tools: `tensorboard`, `mlflow`
- Add other support for business data in datawarehouse or operational databases: `Athena`, `Postgres`, ...
- Add support other vector database: `pgvector`, `milvius`, `s3-vectors`

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
  metadata_columns: 
    - category
    - brand
    - type

data:
  training:
    type: csv
    path: data/training/my_data.csv
  prediction:
    type: csv
    path: data/prediction/new_data.csv

modeling:
  light_mode: false
  light_mode_sample_size: 15000
  bottle_neck_size: 96
  epochs: 5
  batch_size: 256
  hidden_layer_sizes: [512, 256, 128]
  
  modeling_columns:
    categorical_columns:
      - category
      - brand
      - type
      - status
    
    numerical_columns:
      - price
      - age
      - rating
      - quantity

visualisation:
  n_samples: 30000
  visualisation_columns:
    hover_data_columns_name:
      - brand
      - category
      - price
    color_data_column_name: category
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
autoembed-cli visualize --yaml_path config.yaml
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
