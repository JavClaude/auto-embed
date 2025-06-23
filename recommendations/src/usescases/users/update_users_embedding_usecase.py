from kink import inject

from recommendations.src.domain.entites.classified_events import ClassifiedEventType
from recommendations.src.domain.exceptions.embedding_shape_mismatch_error import EmbeddingShapeMismatchError
from recommendations.src.domain.interfaces.classified_embeddings_repository_interface import ClassifiedEmbeddingsRepositoryInterface
from recommendations.src.domain.interfaces.user_embedding_repository_interface import UserEmbedding, UserEmbeddingRepositoryInterface
from recommendations.src.usescases.users.update_users_embedding_command import UpdateUsersEmbeddingCommand

@inject()
class UpdateUsersEmbeddingUsecase:
    def __init__(self, user_embedding_repository: UserEmbeddingRepositoryInterface, classified_embeddings_repository: ClassifiedEmbeddingsRepositoryInterface):
        self.user_embedding_repository = user_embedding_repository
        self.classified_embeddings_repository = classified_embeddings_repository

    def execute(self, command: UpdateUsersEmbeddingCommand) -> None:
        classified_embeddings = self.classified_embeddings_repository.get_classified_embeddings(command.classified_ref)
        user_embedding = self.user_embedding_repository.get_user_embedding(command.user_id)

        if user_embedding is None:
            self.logger.info(f"User embedding not found for user {command.user_id}, creating new one with classified embeddings: {classified_embeddings.classified_ref}")

            self.user_embedding_repository.update_user_embedding(UserEmbedding(
                user_ref=command.user_id,
                embedding=classified_embeddings.classified_embeddings
            ))

        if classified_embeddings.classified_embeddings.shape != user_embedding.embedding.shape:
            self.logger.error(f"Classified embeddings shape {classified_embeddings.classified_embeddings.shape} does not match user embedding shape {user_embedding.embedding.shape}")
            raise EmbeddingShapeMismatchError(f"Classified embeddings shape {classified_embeddings.classified_embeddings.shape} does not match user embedding shape {user_embedding.embedding.shape}")

        # Move to domain layer

        if command.classified_event_type == ClassifiedEventType.CONTACT:
            weight = 0.6
        
        elif command.classified_event_type == ClassifiedEventType.DETAIL:
            weight = 0.3
        
        else:
            weight = 0.1

        user_embedding.embedding = user_embedding.embedding + classified_embeddings.classified_embeddings * weight
        
        # End move to domain layer
        self.logger.info(f"Updating user embedding for user {command.user_id} with classified embeddings: {classified_embeddings.classified_ref} and weight: {weight}")
        self.user_embedding_repository.update_user_embedding(user_embedding)

        return user_embedding