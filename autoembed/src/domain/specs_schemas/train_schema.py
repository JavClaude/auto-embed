class TrainSchema:
    
    @dataclass



class ModelingColumns:
    categorical_columns: List[str] = field(default_factory=list)
    numerical_columns: List[str] = field(default_factory=list)
    date_columns: List[str] = field(default_factory=list)
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