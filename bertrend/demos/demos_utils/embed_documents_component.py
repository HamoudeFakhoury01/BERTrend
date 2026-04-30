#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import streamlit as st

from bertrend.demos.demos_utils.i18n import translate
from bertrend.demos.demos_utils.icons import SUCCESS_ICON
from bertrend.demos.demos_utils.state_utils import SessionStateManager
from bertrend.services.embedding_service import EmbeddingService
from bertrend.utils.data_loading import TEXT_COLUMN


def display_embed_documents_component():
    """
    Common Streamlit UI component for embedding documents.
    """
    # Embed documents
    if st.button(translate("embed_documents")):
        with st.spinner(translate("embedding_documents")):
            embedding_dtype = SessionStateManager.get("embedding_dtype")
            embedding_model_name = SessionStateManager.get("embedding_model_name")
            if SessionStateManager.get("embedding_service_type", "remote") == "local":
                embedding_service = EmbeddingService(
                    local=True,
                    model_name=embedding_model_name,
                    embedding_dtype=embedding_dtype,
                )
            else:
                embedding_service = EmbeddingService(
                    local=False,
                    url=SessionStateManager.get("embedding_service_url"),
                )
                SessionStateManager.set(
                    "embedding_model_name", embedding_service.embedding_model_name
                )

            df = SessionStateManager.get_dataframe("time_filtered_df")
            mask = df[TEXT_COLUMN].notna() & df[TEXT_COLUMN].astype(str).str.strip().ne("")
            n_dropped = int((~mask).sum())
            if n_dropped > 0:
                df = df[mask].reset_index(drop=True)
                SessionStateManager.set("time_filtered_df", df)
                st.warning(f"Skipped {n_dropped} rows with empty or NaN text before embedding")
            texts = df[TEXT_COLUMN].tolist()

            embeddings, token_strings, token_embeddings = embedding_service.embed(
                texts=texts,
            )

            SessionStateManager.set(
                "embedding_model", embedding_service.embedding_model
            )
            SessionStateManager.set("embeddings", embeddings)
            SessionStateManager.set("token_strings", token_strings)
            SessionStateManager.set("token_embeddings", token_embeddings)

            SessionStateManager.set("data_embedded", True)

            st.success(translate("embeddings_calculated_message"), icon=SUCCESS_ICON)
