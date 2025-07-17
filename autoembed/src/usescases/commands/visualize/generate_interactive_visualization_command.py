from dataclasses import dataclass

@dataclass
class GenerateInteractiveVisualizationCommand:
    n_samples: int = 30000
