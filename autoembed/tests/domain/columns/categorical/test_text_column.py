import pandas as pd
import numpy as np
from autoembed.src.domain.columns.categorical.text_column import CLS_TOKEN, TextColumn


def test_when_i_create_a_text_column_from_a_series_i_should_get_a_trained_tokenizer():
    # Given
    series = pd.Series(["Bonjour le monde", "Comment allez-vous ?", "Ceci est un test", "Python est génial"], name="description")

    # When
    text_col = TextColumn.from_series(series, max_length=128, vocab_size=1000)

    # Then

    assert text_col.tokenizer is not None
    assert text_col.max_length == 128
    assert text_col.word_embedding == 64
    assert text_col.vocab_size != 0
    assert text_col.tokenizer.get_vocab_size() != 0


def test_when_i_transform_a_series_i_should_get_a_correct_transformed_text_column():
    # Given
    series = pd.Series(["Bonjour le monde", "   Bonjour    le      monde    "])
    text_col = TextColumn.from_series(series, max_length=32, vocab_size=100)

    # When
    transformed_col = text_col.transform(series)

    # Then
    assert transformed_col.input_ids.shape == (2, 32)
    assert np.array_equal(transformed_col.input_ids[0], transformed_col.input_ids[1])
    assert transformed_col.input_ids[0][0] == text_col.tokenizer.token_to_id(CLS_TOKEN)
    assert transformed_col.input_ids[0][-1] == text_col.tokenizer.token_to_id("[PAD]")
