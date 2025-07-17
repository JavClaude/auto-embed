from logging import Logger

from kink import inject
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go

from recommendations.src.domain.interfaces.embeddings_repository_interface import (
    ClassifiedEmbeddingsRepositoryInterface,
)
from recommendations.src.usescases.commands.visualize.generate_interactive_visualization_command import GenerateInteractiveVisualizationCommand

@inject()
class GenerateInteractiveVisualizationCommandUsecase:
    def __init__(self, classified_embeddings_repository: ClassifiedEmbeddingsRepositoryInterface, logger: Logger):
        self.classified_embeddings_repository = classified_embeddings_repository
        self.logger = logger

    def execute(self, command: GenerateInteractiveVisualizationCommand) -> dict:
        # On récupère tous les embeddings et, on échantillonne après, a voir si on ne peut pas sampler directement dans la requête chromaDb
        all_embeddings = self.classified_embeddings_repository.get_all_embeddings()
        
        self.logger.info(f"Found {len(all_embeddings)} embeddings")

        sampled_embeddings = all_embeddings.sample_batch(command.n_samples)
        embeddings = np.array([embedding.classified_embeddings for embedding in sampled_embeddings.classified_embeddings])

        self.logger.info(f"Fit Reducer (TSNE) on {command.n_samples} embeddings")
        
        reducer = TSNE(n_components=2, random_state=42, n_jobs=-1)

        embeddings_2d = reducer.fit_transform(embeddings)
    
        self.logger.info(f"Generate interactive visualization (Plotly)")

        viz_dataframe = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'label': [embedding.vehicle_make for embedding in sampled_embeddings.classified_embeddings]
        })

        viz_dataframe["classified_ref"] = [embedding.classified_ref for embedding in sampled_embeddings.classified_embeddings]
        viz_dataframe["vehicle_make"] = [embedding.vehicle_make for embedding in sampled_embeddings.classified_embeddings]
        viz_dataframe["vehicle_model"] = [embedding.vehicle_model for embedding in sampled_embeddings.classified_embeddings]
        viz_dataframe["vehicle_commercial_name"] = [embedding.vehicle_commercial_name for embedding in sampled_embeddings.classified_embeddings]
        viz_dataframe["vehicle_version"] = [embedding.vehicle_version for embedding in sampled_embeddings.classified_embeddings]
        viz_dataframe["vehicle_energy"] = [embedding.vehicle_energy for embedding in sampled_embeddings.classified_embeddings]
        viz_dataframe["vehicle_price"] = [embedding.vehicle_price for embedding in sampled_embeddings.classified_embeddings]
        viz_dataframe["vehicle_year"] = [embedding.vehicle_year for embedding in sampled_embeddings.classified_embeddings]

        fig = px.scatter(
            viz_dataframe,
            x="x",
            y="y",
            title="Interactive Visualization",
            template="plotly_white",
            color='label',
            color_discrete_sequence=px.colors.qualitative.Plotly,
            hover_data=["vehicle_make", "vehicle_model", "vehicle_commercial_name", "vehicle_version", "vehicle_energy", "vehicle_price", "vehicle_year"],
        )

        fig.update_layout(
        width=1000,
        height=800,
        hovermode='closest',
        legend=dict(
            title="Vehicle",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01
        )
    )
    
        fig.write_html('test.html')
        print(f"Visualisation interactive sauvegardée dans test.html")

        return sampled_embeddings