from dataclasses import dataclass

from autoembed.src.yaml.auto_embed_yaml_schema import VisualisationColumns

@dataclass
class GenerateInteractiveVisualizationCommand:
    n_samples: int
    visualisation_columns: VisualisationColumns
