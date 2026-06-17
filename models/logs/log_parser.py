#!/usr/bin/env python3
"""
Drain-lite log template extractor for Open5GS/UERANSIM logs.
Self-contained prefix-tree implementation (no external drain3 dependency).
"""

import re
from dataclasses import dataclass, field
from typing import Optional


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(line: str) -> str:
    return _ANSI.sub("", line)


_MASKS: list[tuple[re.Pattern, str]] = [
    # Open5GS timestamp prefix:  "05/16 17:56:42.507: [amf] ERROR: "
    (re.compile(r"^\d{2}/\d{2} [\d:.]+:\s*\[\w+\]\s*\w+:\s*"), ""),
    # UERANSIM timestamp prefix: "[2026-05-16 17:56:42.507] [nas] [debug] "
    (re.compile(r"^\[\d{4}-\d{2}-\d{2} [\d:.]+\]\s*\[\w+\]\s*\[\w+\]\s*"), ""),
    # ISO 8601 date-time boundary: ddTHH inside MongoDB/system JSON logs.
    # "\b\d+\b" misses these because T is \w, removing the word boundary
    # between the day digits and T, and between T and the hour digits.
    # e.g.  16T23  /  19T03  ->  <N>T<N>  so all timestamps normalise.
    (re.compile(r"\d{1,2}T\d{2}"), "<N>T<N>"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "<IP>"),
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"), "<UUID>"),
    (re.compile(r"imsi-\d+"), "<IMSI>"),
    (re.compile(r"0x[0-9a-fA-F]+"), "<HEX>"),
    # port numbers after colon
    (re.compile(r":\d{2,5}\b"), ":<PORT>"),
    # file path in parens:  (../src/amf/ngap-handler.c:690)
    (re.compile(r"\(\.\./[^\)]+\)"), "(<SRC>)"),
    # generic integers (last, to avoid eating hex/IP/port already masked)
    (re.compile(r"\b\d+\b"), "<N>"),
]


def mask_tokens(line: str) -> str:
    line = strip_ansi(line).strip()
    for pat, repl in _MASKS:
        line = pat.sub(repl, line)
    return line


def tokenize(line: str) -> list[str]:
    return line.split()


WILDCARD = "<*>"


@dataclass
class LogCluster:
    tid: int
    template_tokens: list[str]
    count: int = 1

    @property
    def template(self) -> str:
        return " ".join(self.template_tokens)


@dataclass
class PrefixNode:
    children: dict = field(default_factory=dict)
    clusters: list = field(default_factory=list)  # leaf clusters


class LogParser:
    """
    Drain log parser.

    depth               — levels of the prefix tree (first depth-1 tokens used as keys)
    similarity_threshold — minimum token-overlap ratio to match an existing cluster
    max_children        — max branches per prefix node before collapsing to wildcard
    """

    def __init__(self, depth: int = 4, similarity_threshold: float = 0.5,
                 max_children: int = 128):
        self.depth                = depth
        self.similarity_threshold = similarity_threshold
        self.max_children         = max_children
        self._root: PrefixNode    = PrefixNode()
        self._clusters: dict[int, LogCluster] = {}
        self._next_id: int        = 1
        self.templates: dict[int, str] = {}   # tid -> template string
        self.vocab:     dict[str, int] = {}   # template string -> tid

    def _preprocess(self, line: str) -> list[str]:
        return tokenize(mask_tokens(line))

    def _seq_similarity(self, tokens: list[str], cluster: LogCluster) -> float:
        tmpl = cluster.template_tokens
        if len(tokens) != len(tmpl):
            return 0.0
        match = sum(1 for t, tt in zip(tokens, tmpl) if tt == WILDCARD or t == tt)
        return match / len(tmpl)

    def _get_leaf(self, tokens: list[str]) -> PrefixNode:
        """Navigate or create prefix-tree path for these tokens, return leaf node."""
        node = self._root
        n = len(tokens)
        if n not in node.children:
            node.children[n] = PrefixNode()
        node = node.children[n]
        for tok in tokens[:self.depth - 1]:
            key = tok if not tok.startswith("<") else WILDCARD
            if key not in node.children:
                if len(node.children) >= self.max_children:
                    key = WILDCARD
                if key not in node.children:
                    node.children[key] = PrefixNode()
            node = node.children[key]
        return node

    def _best_match(self, tokens: list[str],
                    node: PrefixNode) -> Optional[LogCluster]:
        best_sim, best = -1.0, None
        for clust in node.clusters:
            sim = self._seq_similarity(tokens, clust)
            if sim > best_sim:
                best_sim, best = sim, clust
        return best if best_sim >= self.similarity_threshold else None

    def _merge(self, tokens: list[str], cluster: LogCluster) -> None:
        """Wildcard any token that differs from the existing template."""
        old_str = cluster.template
        cluster.template_tokens = [
            tt if (t == tt or tt == WILDCARD) else WILDCARD
            for t, tt in zip(tokens, cluster.template_tokens)
        ]
        cluster.count += 1
        new_str = cluster.template
        if old_str != new_str:
            self.templates[cluster.tid] = new_str
            self.vocab.pop(old_str, None)
            self.vocab[new_str] = cluster.tid

    def dedup_templates(self) -> dict[int, int]:
        # returns {old_tid: canonical_tid} when sequential runs assign duplicate template strings
        seen: dict[str, int] = {}   # template_str -> canonical tid
        remap: dict[int, int] = {}
        for tid, tmpl_str in sorted(self.templates.items()):
            if tmpl_str in seen:
                remap[tid] = seen[tmpl_str]
            else:
                seen[tmpl_str] = tid
        return remap

    def parse(self, line: str) -> tuple[int, str]:
        tokens = self._preprocess(line)
        if not tokens:
            tokens = ["<EMPTY>"]

        node  = self._get_leaf(tokens)
        match = self._best_match(tokens, node)

        if match is not None:
            self._merge(tokens, match)
            return match.tid, match.template

        tid   = self._next_id
        self._next_id += 1
        clust = LogCluster(tid=tid, template_tokens=list(tokens))
        node.clusters.append(clust)
        self._clusters[tid]     = clust
        self.templates[tid]     = clust.template
        self.vocab[clust.template] = tid
        return tid, clust.template

    def save(self, path) -> None:
        import json
        from pathlib import Path
        data = {
            "depth": self.depth,
            "similarity_threshold": self.similarity_threshold,
            "max_children": self.max_children,
            "next_id": self._next_id,
            "clusters": [
                {"tid": c.tid, "tokens": c.template_tokens, "count": c.count}
                for c in self._clusters.values()
            ],
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data))

    @classmethod
    def load(cls, path) -> "LogParser":
        import json
        from pathlib import Path
        data = json.loads(Path(path).read_text())
        obj  = cls(
            depth=data["depth"],
            similarity_threshold=data["similarity_threshold"],
            max_children=data["max_children"],
        )
        obj._next_id = data["next_id"]
        for item in data["clusters"]:
            c = LogCluster(tid=item["tid"], template_tokens=item["tokens"],
                           count=item["count"])
            obj._clusters[item["tid"]]  = c
            obj.templates[item["tid"]]  = c.template
            obj.vocab[c.template]       = item["tid"]
            node = obj._get_leaf(item["tokens"])
            node.clusters.append(c)
        return obj