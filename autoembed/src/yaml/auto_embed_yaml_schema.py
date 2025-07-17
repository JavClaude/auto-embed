import json
from dataclasses import asdict, dataclass
from typing import Any, List, Literal


@dataclass
class IdColumn:
    columns: List[str]

@dataclass
class MetadataColumns:
    columns: List[str]


@dataclass
class VectorStore:
    def __init__(self, **kwargs):   
        self.vector_collection_name = kwargs.get("vector_collection_name")
        self.metadata_columns = MetadataColumns(kwargs.get("metadata_columns"))


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
    categorical_columns: List[str]
    numerical_columns: List[str]


@dataclass
class Modeling:
    def __init__(self, **kwargs):
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
        self.model_name = kwargs.get("model_name")
        self.id_column = IdColumn(kwargs.get("id_column"))
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
