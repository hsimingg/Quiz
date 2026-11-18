"""Template 7: Union-Find (disjoint sets) to group variant pairs into one lexical item.
kompanya~kumpanya and kumpanya~compania -> one group {compania, kompanya, kumpanya}.
Path halving + union by rank: near O(1) per operation."""
from collections import defaultdict


class UnionFind:
    def __init__(self):
        self.parent, self.rank = {}, {}

    def find(self, x):
        self.parent.setdefault(x, x)
        self.rank.setdefault(x, 0)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def groups(self):
        g = defaultdict(list)
        for x in list(self.parent):
            g[self.find(x)].append(x)
        return list(g.values())


def group_pairs(pairs, counts=None):
    """pairs: iterable of (a, b). Returns groups, each sorted by frequency (desc) if counts
    are given, so the first word is the dominant spelling. Largest groups first."""
    uf = UnionFind()
    for a, b in pairs:
        uf.union(a, b)
    key = (lambda w: (-counts.get(w, 0), w)) if counts else (lambda w: w)
    groups = [sorted(g, key=key) for g in uf.groups()]
    return sorted(groups, key=lambda g: (-len(g), g[0]))
