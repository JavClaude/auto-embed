from logging import Logger

from kink import inject
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px


from autoembed.src.domain.interfaces.embeddings_repository_interface import (
    EmbeddingsRepositoryInterface,
)
from autoembed.src.usescases.commands.visualize.generate_interactive_visualization_command import GenerateInteractiveVisualizationCommand

@inject()
class GenerateInteractiveVisualizationCommandUsecase:
    def __init__(self, embeddings_repository: EmbeddingsRepositoryInterface, logger: Logger):
        self.embeddings_repository = embeddings_repository
        self.logger = logger

    def execute(self, command: GenerateInteractiveVisualizationCommand) -> dict:
        
        # On récupère tous les embeddings et, on échantillonne après, a voir si on ne peut pas sampler directement dans la requête chromaDb
        all_embeddings = self.embeddings_repository.get_all_embeddings()
        
        self.logger.info(f"Found {len(all_embeddings)} embeddings")

        sampled_embeddings = all_embeddings.sample_batch(command.n_samples)
        embeddings = np.array([business_embedding.embeddings for business_embedding in sampled_embeddings.embeddings])

        self.logger.info(f"Fit Reducer (TSNE) on {command.n_samples} embeddings")
        
        reducer = TSNE(n_components=2, random_state=42, n_jobs=-1)

        embeddings_2d = reducer.fit_transform(embeddings)
    
        self.logger.info("Generate interactive visualization (Plotly)")

        viz_dataframe = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'label': [embedding.metadata[command.visualisation_columns.color_data_column_name] for embedding in sampled_embeddings.embeddings]
        })

        viz_dataframe["id"] = [embedding.id for embedding in sampled_embeddings.embeddings]
        
        for column in command.visualisation_columns.hover_data_columns_name:
            viz_dataframe[column] = [embedding.metadata[column] for embedding in sampled_embeddings.embeddings]

        fig = px.scatter(
            viz_dataframe,
            x="x",
            y="y",
            title="Interactive Visualization",
            template="plotly_white",
            color='label',
            color_discrete_sequence=px.colors.qualitative.Plotly,
            hover_data=command.visualisation_columns.hover_data_columns_name,
        )

        fig.update_layout(
        width=1000,
        height=800,
        hovermode='closest',
        legend=dict(
            title="Embeddings",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01
        )
    )
    
        fig.write_html('interactive_visualization.html')

        return sampled_embeddings