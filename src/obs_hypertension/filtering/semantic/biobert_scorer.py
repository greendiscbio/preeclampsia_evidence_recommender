import torch
from sentence_transformers import SentenceTransformer


class BioBERTScorer:
    def __init__(
        self,
        model_name,
        semantic_query,
        sim_threshold,
        batch_size=16,
        return_all_scores=False,   # 👈 NUEVO
    ):
        self.model_name = model_name
        self.semantic_query = semantic_query
        self.sim_threshold = sim_threshold
        self.batch_size = batch_size
        self.return_all_scores = return_all_scores

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        self.model = SentenceTransformer(model_name, device=str(self.device))

        self.query_emb = self.model.encode(
            semantic_query,
            normalize_embeddings=True
        )

    def score_partition(self, rows):
        buffer_texts = []
        buffer_ids = []

        for r in rows:
            buffer_ids.append(r["corpusid"])
            buffer_texts.append(r["text"])

            if len(buffer_texts) == self.batch_size:
                yield from self._score_batch(buffer_ids, buffer_texts)
                buffer_ids, buffer_texts = [], []

        if buffer_texts:
            yield from self._score_batch(buffer_ids, buffer_texts)

    def _score_batch(self, ids, texts):
        emb = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=self.batch_size,
            show_progress_bar=False
        )

        scores = emb @ self.query_emb

        for cid, text, score in zip(ids, texts, scores):
            score = float(score)

            if self.return_all_scores:
                yield (cid, score, text)
            else:
                if score >= self.sim_threshold:
                    yield (cid, score, text)

