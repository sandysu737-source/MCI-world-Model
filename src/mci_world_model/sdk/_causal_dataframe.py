"""CausalDataFrame — pandas DataFrame with causal discovery built-in.

Usage:
    >>> from mci_world_model import CausalDataFrame
    >>> cdf = CausalDataFrame(df)
    >>> graph = cdf.causal.discover(method="pc")
    >>> graph.show()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CausalGraphResult:
    """Causal discovery result with visualization support."""

    nodes: list[str]
    edges: list[tuple[str, str]]
    adj_matrix: np.ndarray
    confidence: float = 0.0
    method: str = "pc"

    def __repr__(self) -> str:
        return (
            f"CausalGraph({len(self.nodes)} nodes, "
            f"{len(self.edges)} edges, "
            f"conf={self.confidence:.3f}, "
            f"method={self.method})"
        )

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"CausalGraph: {len(self.nodes)} variables, {len(self.edges)} edges",
            f"  Method: {self.method}, Confidence: {self.confidence:.3f}",
            "  Edges:",
        ]
        for src, dst in sorted(self.edges):
            lines.append(f"    {src} → {dst}")
        return "\n".join(lines)

    def show(self, title: str | None = None):
        """Render the causal graph using matplotlib + networkx.

        Requires: pip install mci-world-model[full]
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError:
            print("Visualization requires: pip install mci-world-model[full]")
            print("  (installs matplotlib + networkx)")
            return

        G = nx.DiGraph()
        G.add_nodes_from(self.nodes)
        for src, dst in self.edges:
            # Edge thickness proportional to confidence
            conf = self.confidence
            G.add_edge(src, dst, weight=max(conf, 0.1))

        pos = nx.spring_layout(G, seed=42, k=2.0)

        _fig, ax = plt.subplots(figsize=(max(8, len(self.nodes) * 1.2), 6))
        nx.draw_networkx_nodes(G, pos, node_color="lightblue",
                               node_size=1200, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)

        # Draw edges with arrowheads
        edge_weights = [G[u][v].get("weight", 0.5) * 3 for u, v in G.edges()]
        nx.draw_networkx_edges(
            G, pos, edge_color="steelblue",
            width=edge_weights,
            arrowstyle="->", arrowsize=15,
            connectionstyle="arc3,rad=0.1",
            ax=ax,
        )

        ax.set_title(title or f"Causal Graph ({self.method})", fontsize=14)
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    def to_networkx(self):
        """Convert to networkx DiGraph."""
        import networkx as nx
        G = nx.DiGraph()
        G.add_nodes_from(self.nodes)
        for src, dst in self.edges:
            G.add_edge(src, dst)
        return G


class _CausalAccessor:
    """Accessor for df.causal.discover() / df.causal.do() etc."""

    def __init__(self, data: np.ndarray, columns: list[str]):
        self._data = data
        self._columns = columns

    def discover(self, method: str = "pc", **kwargs) -> CausalGraphResult:
        """Discover causal graph from data.

        Args:
            method: one of 'pc', 'fci', 'notears', 'cam', 'camgolem', 'ges'
            **kwargs: passed to the discoverer constructor

        Returns:
            CausalGraphResult with .show() method
        """
        from mci_world_model.sdk._autonomous_law_discoverer_v2 import (
            CAMDiscoverer,
            CAMGOLEMDiscoverer,
            FCIDiscoverer,
            NOTEARSDiscoverer,
            PCSkeletonDiscoverer,
        )

        discoverers = {
            "pc": lambda: PCSkeletonDiscoverer(alpha=kwargs.get("alpha", 0.05)),
            "fci": lambda: FCIDiscoverer(alpha=kwargs.get("alpha", 0.05)),
            "notears": lambda: NOTEARSDiscoverer(
                lambda1=kwargs.get("lambda1", 0.1),
                max_iter=kwargs.get("max_iter", 500),
            ),
            "cam": lambda: CAMDiscoverer(
                alpha=kwargs.get("alpha", 0.05),
                n_subsamples=kwargs.get("n_subsamples", 30),
                stability_threshold=kwargs.get("stability_threshold", 0.5),
            ),
            "camgolem": lambda: CAMGOLEMDiscoverer(
                alpha=kwargs.get("alpha", 0.05),
                lambda1=kwargs.get("lambda1", 0.01),
                max_iter=kwargs.get("max_iter", 300),
            ),
        }

        if method not in discoverers:
            raise ValueError(
                f"Unknown method: {method}. "
                f"Choose from: {list(discoverers.keys())}"
            )

        discoverer = discoverers[method]()
        skel = discoverer.discover(self._data, self._columns)

        return CausalGraphResult(
            nodes=skel.nodes,
            edges=skel.edges,
            adj_matrix=skel.adj_matrix.copy(),
            confidence=skel.confidence,
            method=method,
        )

    def summary(self) -> str:
        """Quick data summary."""
        n, p = self._data.shape
        return (
            f"CausalDataFrame: {n} samples × {p} variables\n"
            f"  Variables: {', '.join(self._columns[:10])}"
            + ("..." if p > 10 else "")
        )


class CausalDataFrame:
    """pandas DataFrame wrapper with built-in causal discovery.

    Usage:
        >>> import pandas as pd
        >>> from mci_world_model import CausalDataFrame
        >>> df = pd.DataFrame({"X": [...], "Y": [...]})
        >>> cdf = CausalDataFrame(df)
        >>> g = cdf.causal.discover(method="pc")
        >>> g.show()
    """

    def __init__(self, data):
        """Wrap a pandas DataFrame or numpy array.

        Args:
            data: pandas DataFrame or numpy ndarray
        """
        # Accept pandas DataFrame
        try:
            import pandas as pd
            if isinstance(data, pd.DataFrame):
                self._columns = list(data.columns)
                self._data = data.to_numpy(dtype=float)
            else:
                raise TypeError
        except (ImportError, TypeError):
            # Fallback: numpy array
            arr = np.asarray(data, dtype=float)
            if arr.ndim != 2:
                raise ValueError("Data must be 2-dimensional") from None
            self._data = arr
            self._columns = [f"X{i}" for i in range(arr.shape[1])]

        self.causal = _CausalAccessor(self._data, self._columns)

    def __repr__(self) -> str:
        return self.causal.summary()
