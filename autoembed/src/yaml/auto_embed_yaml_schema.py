import json
from dataclasses import asdict, dataclass, field
from typing import Any, List, Literal


@dataclass
class IdColumns:
    columns: List[str]

    def __post_init__(self):
        if self.columns == []:
            raise ValueError("Id columns cannot be empty, please provide at least one column")


@dataclass
class MetadataColumns:
    columns: List[str] | None


class VectorStore:
    def __init__(self, **kwargs):
        self.vector_store_backend = kwargs.get("vector_store_backend", "chromadb")
        if not self._is_backend_supported(self.vector_store_backend):
            raise ValueError(f"Vector store backend {self.vector_store_backend} not supported")

        self.vector_collection_name = kwargs.get("vector_collection_name")
        self.id_columns = IdColumns(kwargs.get("metadata_columns", []))
        self.metadata_columns = MetadataColumns(kwargs.get("metadata_columns"))

    def _is_backend_supported(self, backend: str) -> bool:
        supported_backend = ["chromadb"]
        return backend in supported_backend


@dataclass
class TrainingData:
    type: Literal["csv", "parquet"]
    path: str


@dataclass
class PredictionData:
    type: Literal["csv", "parquet"]
    path: str


@dataclass
class Data:
    def __init__(self, **kwargs):
        self.training = TrainingData(**kwargs.get("training"))
        self.prediction = PredictionData(**kwargs.get("prediction"))


@dataclass
class ModelingColumns:
    categorical_columns: List[str] = field(default_factory=list)
    numerical_columns: List[str] = field(default_factory=list)
    text_column: str | None = None


@dataclass
class Modeling:
    def __init__(self, **kwargs):
        self.model_version = kwargs.get("model_version")
        self.light_mode = kwargs.get("light_mode")
        self.light_mode_sample_size = kwargs.get("light_mode_sample_size")
        self.bottle_neck_size = kwargs.get("bottle_neck_size")
        self.epochs = kwargs.get("epochs")
        self.batch_size = kwargs.get("batch_size")
        self.hidden_layer_sizes = kwargs.get("hidden_layer_sizes")
        self.modeling_columns = ModelingColumns(**kwargs.get("modeling_columns"))


@dataclass
class VisualisationColumns:
    hover_data_columns_name: List[str]
    color_data_column_name: str


@dataclass
class Visualisation:
    def __init__(self, **kwargs):
        self.n_samples = kwargs.get("n_samples")
        self.visualisation_columns = VisualisationColumns(**kwargs.get("visualisation_columns"))


@dataclass
class AutoEmbedByYamlFileSchema:
    def __init__(self, **kwargs):
        self.project_name = kwargs.get("project_name")
        self.vector_store = VectorStore(**kwargs.get("vector_store"))
        self.data = Data(**kwargs.get("data"))
        self.modeling = Modeling(**kwargs.get("modeling"))
        self.visualisation = Visualisation(**kwargs.get("visualisation"))

    @classmethod
    def from_yaml_as_dict(cls, yaml_as_dict: dict[str, Any]) -> "AutoEmbedByYamlFileSchema":
        try:
            auto_embed_yaml_schema = AutoEmbedByYamlFileSchema(**yaml_as_dict)
            return auto_embed_yaml_schema
        except Exception as e:
            raise ValueError(f"Invalid YAML schema: {e}")

    def to_json(self) -> str:
        return json.dumps(asdict(self.data), indent=4)
