#!/usr/bin/env python3
"""
LogRobust: BiLSTM + attention on word-mean embeddings (Zhang et al., WWW 2019).
Adapted to unsupervised setting: BiLSTM autoencoder, cosine distance as anomaly score.
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import LogRecord, load_sequences, get_parser, vocab_size as _vocab_size_fn

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH = True
except ImportError:
    _TORCH = False

_SKIP = re.compile(r"^(<[^>]+>|\{[^}]+\}|\[[^\]]*\]|[^\w]*)$")


def _tokenise_template(template: str) -> list[str]:
    tokens = []
    for tok in template.split():
        if not _SKIP.match(tok):
            tok = tok.strip(".:,;()[]{}\"'")
            if tok and not _SKIP.match(tok):
                tokens.append(tok.lower())
    return tokens


def _build_word_vocab(templates: dict[int, str]) -> dict[str, int]:
    words: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    for tmpl in templates.values():
        for w in _tokenise_template(tmpl):
            if w not in words:
                words[w] = len(words)
    return words


class _LogRobustNet(nn.Module if _TORCH else object):
    def __init__(self, word_vocab: int, word_dim: int, lstm_hidden: int,
                 n_layers: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.word_embed = nn.Embedding(word_vocab, word_dim, padding_idx=0)
        self.bilstm     = nn.LSTM(
            word_dim, lstm_hidden // 2, n_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.attn_w  = nn.Linear(lstm_hidden, 1, bias=False)
        self.fc      = nn.Linear(lstm_hidden, out_dim)

    def forward(self, template_words: "torch.Tensor",
                word_mask: "torch.Tensor") -> "torch.Tensor":
        B, W, Mw = template_words.shape
        emb   = self.word_embed(template_words.view(B * W, Mw))
        wm    = word_mask.view(B * W, Mw, 1)
        count = wm.sum(1).clamp(min=1)
        templ_emb = (emb * wm).sum(1) / count
        templ_emb = templ_emb.view(B, W, -1)
        h, _  = self.bilstm(templ_emb)
        attn  = torch.softmax(self.attn_w(h), dim=1)
        ctx   = (attn * h).sum(dim=1)
        return self.fc(ctx)


class LogRobustDetector:
    def __init__(self, window: int = 10, step: int = 1,
                 word_embed_dim: int = 32, lstm_hidden: int = 64,
                 n_layers: int = 2, dropout: float = 0.1,
                 max_words: int = 16, threshold_k: float = 3.0):
        if not _TORCH:
            raise RuntimeError("PyTorch required. pip install torch")
        self.window         = window
        self.step           = step
        self.word_embed_dim = word_embed_dim
        self.lstm_hidden    = lstm_hidden
        self.n_layers       = n_layers
        self.dropout        = dropout
        self.max_words      = max_words
        self.threshold_k    = threshold_k
        self._word_vocab:  Optional[dict] = None
        self._tmpl_words:  Optional[dict] = None
        self._model:       Optional[_LogRobustNet] = None
        self._threshold:   float = 0.0
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_vocab(self) -> None:
        parser            = get_parser()
        self._word_vocab  = _build_word_vocab(parser.templates)
        self._tmpl_words  = {}
        for tid, tmpl_str in parser.templates.items():
            words = [self._word_vocab.get(w, self._word_vocab["<UNK>"])
                     for w in _tokenise_template(tmpl_str)]
            self._tmpl_words[int(tid)] = words[:self.max_words]

    def _seqs_to_tensors(self, seqs) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        Xs, Ls = [], []
        for ids, lbl in seqs:
            if len(ids) < self.window:
                continue
            Xs.append(ids[:self.window])
            Ls.append(lbl)
        if not Xs:
            return (torch.zeros(0, self.window, self.max_words, dtype=torch.long),
                    torch.zeros(0, self.window, self.max_words),
                    torch.zeros(0))
        B  = len(Xs)
        tw = torch.zeros(B, self.window, self.max_words, dtype=torch.long)
        wm = torch.zeros(B, self.window, self.max_words)
        for i, ids_w in enumerate(Xs):
            for j, tid in enumerate(ids_w[:self.window]):
                word_ids = self._tmpl_words.get(int(tid), [])
                n        = min(len(word_ids), self.max_words)
                if n > 0:
                    tw[i, j, :n] = torch.tensor(word_ids[:n], dtype=torch.long)
                    wm[i, j, :n] = 1.0
        return tw, wm, torch.tensor(Ls, dtype=torch.float32)

    def _target_mean(self, tw: "torch.Tensor", wm: "torch.Tensor") -> "torch.Tensor":
        """Mean of per-template mean-word-embeddings across the window."""
        B, W, Mw = tw.shape
        emb  = self._model.word_embed(tw.view(B * W, Mw))
        mask = wm.view(B * W, Mw, 1)
        per_t = (emb * mask).sum(1) / mask.sum(1).clamp(min=1)
        return per_t.view(B, W, -1).mean(1)

    def fit(self, records: list[LogRecord],
            epochs: int = 20, batch_size: int = 256) -> "LogRobustDetector":
        self._build_vocab()
        seqs = load_sequences(records, window=self.window, step=self.step,
                              group_by_slug=True)
        tw, wm, _ = self._seqs_to_tensors(seqs)
        if len(tw) == 0:
            raise ValueError("No training sequences.")

        self._model = _LogRobustNet(
            word_vocab=len(self._word_vocab), word_dim=self.word_embed_dim,
            lstm_hidden=self.lstm_hidden, n_layers=self.n_layers,
            out_dim=self.word_embed_dim, dropout=self.dropout,
        ).to(self._device)
        opt = torch.optim.Adam(self._model.parameters(), lr=1e-3)

        loader = DataLoader(TensorDataset(tw, wm), batch_size=batch_size, shuffle=True)
        print(f"[LogRobust] Training on {len(tw):,} windows")
        self._model.train()
        for ep in range(1, epochs + 1):
            total = 0.0
            for tw_b, wm_b in loader:
                tw_b, wm_b = tw_b.to(self._device), wm_b.to(self._device)
                pred   = self._model(tw_b, wm_b)
                target = self._target_mean(tw_b, wm_b).detach()
                loss   = (1 - nn.functional.cosine_similarity(pred, target, dim=-1)).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item() * len(tw_b)
            if ep % max(1, epochs // 5) == 0 or ep == epochs:
                print(f"  Epoch {ep:3d}/{epochs}  loss={total/len(tw):.4f}")

        scores, _ = self.score(records)
        self._threshold = float(scores.mean() + self.threshold_k * scores.std()) \
                          if len(scores) else 0.0
        print(f"[LogRobust] Fitted, threshold={self._threshold:.4f}")
        return self

    def score(self, records: list[LogRecord]) -> tuple[np.ndarray, np.ndarray]:
        seqs = load_sequences(records, window=self.window, step=self.step,
                              group_by_slug=True)
        tw, wm, L = self._seqs_to_tensors(seqs)
        if len(tw) == 0:
            return np.array([]), np.array([])
        self._model.eval()
        all_scores = []
        with torch.no_grad():
            for i in range(0, len(tw), 256):
                tw_b = tw[i: i + 256].to(self._device)
                wm_b = wm[i: i + 256].to(self._device)
                pred   = self._model(tw_b, wm_b)
                target = self._target_mean(tw_b, wm_b)
                s = (1 - nn.functional.cosine_similarity(pred, target, dim=-1)).cpu().numpy()
                all_scores.append(s)
        return np.concatenate(all_scores), L.numpy().astype(int)

    def predict(self, records: list[LogRecord]
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scores, labels = self.score(records)
        if len(scores) == 0:
            return np.array([]), np.array([]), np.array([])
        return (scores > self._threshold).astype(int), scores, labels

    def save(self, path: Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), str(path) + "_weights.pt")
        meta = {
            "window": self.window, "step": self.step,
            "word_embed_dim": self.word_embed_dim, "lstm_hidden": self.lstm_hidden,
            "n_layers": self.n_layers, "dropout": self.dropout,
            "max_words": self.max_words, "threshold_k": self.threshold_k,
            "threshold": self._threshold,
            "word_vocab": self._word_vocab,
        }
        Path(str(path) + "_meta.json").write_text(json.dumps(meta))
        Path(str(path) + "_tmpl.json").write_text(
            json.dumps({str(k): v for k, v in self._tmpl_words.items()}))
        print(f"[LogRobust] Saved -> {path}*")

    @classmethod
    def load(cls, path: Path) -> "LogRobustDetector":
        path = Path(path)
        meta = json.loads(Path(str(path) + "_meta.json").read_text())
        obj  = cls(window=meta["window"], step=meta["step"],
                   word_embed_dim=meta["word_embed_dim"],
                   lstm_hidden=meta["lstm_hidden"], n_layers=meta["n_layers"],
                   dropout=meta["dropout"], max_words=meta["max_words"],
                   threshold_k=meta["threshold_k"])
        obj._threshold  = meta["threshold"]
        obj._word_vocab = meta["word_vocab"]
        obj._tmpl_words = {int(k): v for k, v in
                           json.loads(Path(str(path) + "_tmpl.json").read_text()).items()}
        obj._model = _LogRobustNet(
            word_vocab=len(obj._word_vocab), word_dim=obj.word_embed_dim,
            lstm_hidden=obj.lstm_hidden, n_layers=obj.n_layers,
            out_dim=obj.word_embed_dim, dropout=obj.dropout)
        obj._model.load_state_dict(
            torch.load(str(path) + "_weights.pt", map_location=obj._device))
        obj._model.to(obj._device)
        return obj


if __name__ == "__main__":
    from data_loader import load_all
    data  = load_all()
    model = LogRobustDetector(window=10, step=1, word_embed_dim=32, lstm_hidden=64)
    model.fit(data["train"], epochs=20)
    preds, scores, labels = model.predict(data["test"])
    tp = int(((preds==1)&(labels==1)).sum()); fp = int(((preds==1)&(labels==0)).sum())
    fn = int(((preds==0)&(labels==1)).sum())
    prec = tp/max(tp+fp,1); rec = tp/max(tp+fn,1)
    f1   = 2*prec*rec/max(prec+rec,1e-9)
    print(f"[LogRobust] P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")