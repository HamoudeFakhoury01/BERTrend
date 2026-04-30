#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import json
import os
import time
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv
from joblib import Parallel, delayed
from langchain_core.embeddings import Embeddings
from loguru import logger

from bertrend.services.authentication import SecureAPIClient

MAX_N_JOBS = 2
BATCH_DOCUMENT_SIZE = 1000
MAX_DOCS_PER_REQUEST_PER_WORKER = 100000

load_dotenv(override=True)


class EmbeddingAPIClient(SecureAPIClient, Embeddings):
    """
    Custom Embedding API client, can integrate seamlessly with langchain
    """

    def __init__(
        self,
        url: str,
        client_id: str = "bertrend",
        client_secret: str = os.getenv("BERTREND_CLIENT_SECRET", None),
    ):
        super().__init__(url, client_id, client_secret)
        self.model_name = self.get_api_model_name()
        self.num_workers = self.get_num_workers()

    def get_api_model_name(self) -> str:
        """
        Return currently loaded model name in Embedding API.
        """
        with requests.get(
            self.url + "/model_name",
            verify=False,
        ) as response:
            if response.status_code == 200:
                model_name = response.json()
                logger.debug(f"Model name: {model_name}")
                return model_name
            else:
                logger.error(f"Error: {response.status_code}")
                raise Exception(f"Error: {response.status_code}")

    def get_num_workers(self) -> int:
        """
        Return currently loaded number of workers in Embedding API.
        """
        with requests.get(
            self.url + "/num_workers",
            verify=False,
        ) as response:
            if response.status_code == 200:
                num_workers = response.json()
                logger.debug(f"Number of workers: {num_workers}")
                return num_workers
            else:
                logger.error(f"Error: {response.status_code}")
                raise Exception(f"Error: {response.status_code}")

    def embed_query(
        self, text: str | list[str], show_progress_bar: bool = False
    ) -> list[float]:
        if isinstance(text, str):
            text = [text]
        logger.debug(f"Calling EmbeddingAPI using model: {self.model_name}")
        logger.debug("Computing embeddings...")
        with requests.post(
            self.url + "/encode",
            json={"text": text, "show_progress_bar": show_progress_bar},
            verify=False,
            headers=self._get_headers(),
        ) as response:
            if response.status_code == 200:
                embeddings = np.array(response.json()["embeddings"])
                logger.debug("Computing embeddings done")
                return embeddings.tolist()[0]
            else:
                logger.error(f"Error: {response.status_code} | body={response.text[:300]}")

    def embed_batch(
        self,
        texts: list[str],
        show_progress_bar: bool = True,
        max_retries: int = 5,
    ) -> list[list[float]]:
        last_error = None
        for attempt in range(max_retries):
            try:
                with requests.post(
                    self.url + "/encode",
                    json={"text": texts, "show_progress_bar": show_progress_bar},
                    verify=False,
                    headers=self._get_headers(),
                    timeout=600,
                ) as response:
                    if response.status_code == 200:
                        embeddings = np.array(response.json()["embeddings"])
                        return embeddings.tolist()

                    # 401 may indicate the request was dispatched to a worker
                    # whose SECRET_KEY does not match the one that signed our
                    # token. Force a token refresh before the next attempt.
                    if response.status_code == 401:
                        self.token = None
                        self.token_expiry = 0

                    last_error = (
                        f"HTTP {response.status_code} | body={response.text[:300]}"
                    )
            except (requests.RequestException, ConnectionError) as e:
                last_error = f"Network error: {type(e).__name__}: {e}"

            if attempt < max_retries - 1:
                wait = 2 ** attempt  # 1, 2, 4, 8, 16
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed ({last_error}). "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)

        logger.error(
            f"All {max_retries} attempts failed for batch of {len(texts)} texts. "
            f"Last error: {last_error}"
        )
        return []

    def embed_documents(
        self,
        texts: list[str],
        show_progress_bar: bool = True,
        batch_size: int = BATCH_DOCUMENT_SIZE,
    ) -> list[list[float]]:
        if len(texts) > MAX_DOCS_PER_REQUEST_PER_WORKER * self.num_workers:
            # Too many documents to embed in one request, refuse it
            logger.error(
                f"Error: Too many documents to be embedded ({len(texts)} chunks, max {MAX_DOCS_PER_REQUEST_PER_WORKER * self.num_workers})"
            )
            raise ValueError(
                f"Error: Too many documents to be embedded ({len(texts)} chunks, max {MAX_DOCS_PER_REQUEST_PER_WORKER * self.num_workers})"
            )

        logger.debug(f"Calling EmbeddingAPI using model: {self.model_name}")

        # Pre-warm auth token to avoid first-call concurrency edge cases.
        self._get_headers()

        # Split texts into chunks
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        n_batches = len(batches)
        logger.info(
            f"Computing embeddings on {len(texts)} documents using {n_batches} batches..."
        )

        # Per-batch incremental cache so a crash mid-run does not waste all the
        # work that already succeeded. Each successful batch is dumped to
        # `<BERTREND_BASE_DIR>/cache/embeddings_partial/batch_<NNNN>.npy` and
        # reloaded on a subsequent call (idempotent resume).
        base_dir = os.getenv(
            "BERTREND_BASE_DIR", os.path.expanduser("~/.bertrend")
        )
        partial_dir = Path(base_dir) / "cache" / "embeddings_partial"
        partial_dir.mkdir(parents=True, exist_ok=True)

        results: list[list[list[float]]] = []
        n_resumed = 0
        for i, batch in enumerate(batches):
            batch_file = partial_dir / f"batch_{i:04d}.npy"
            if batch_file.exists():
                try:
                    cached = np.load(batch_file)
                    if len(cached) == len(batch):
                        results.append(cached.tolist())
                        n_resumed += 1
                        continue
                except Exception as e:
                    logger.warning(f"Could not reuse cached batch {i}: {e}")

            batch_emb = self.embed_batch(batch, show_progress_bar)
            if not batch_emb:
                results.append([])
                continue

            np.save(batch_file, np.array(batch_emb))
            results.append(batch_emb)

            if (i + 1) % 5 == 0 or i == n_batches - 1:
                done = sum(1 for r in results if r)
                logger.info(f"Progress: {done}/{n_batches} batches OK")

        if n_resumed > 0:
            logger.info(
                f"Resumed {n_resumed}/{n_batches} batches from {partial_dir}"
            )

        failed = [i for i, r in enumerate(results) if not r]
        if failed:
            raise ValueError(
                f"{len(failed)}/{n_batches} batches failed permanently after retries "
                f"(indices: {failed[:10]}{'...' if len(failed) > 10 else ''}). "
                f"Successful batches are cached at {partial_dir} — re-running this "
                f"call will skip them and only retry the failed ones."
            )

        embeddings = [emb for result in results for emb in result]
        assert len(embeddings) == len(texts)

        # Full success → wipe the partial cache.
        try:
            for f in partial_dir.glob("batch_*.npy"):
                f.unlink()
            partial_dir.rmdir()
        except OSError as e:
            logger.warning(f"Could not clean up partial cache: {e}")

        return embeddings

    async def aembed_query(self, text: str) -> list[float]:
        # FIXME!
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        # FIXME!
        return self.embed_documents(texts)
