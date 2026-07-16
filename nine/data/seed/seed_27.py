
class Grafo:
    """Grafo simples usando lista de adjacencia."""

    def __init__(self, direcionado=False):
        self.adjacencias = {}
        self.direcionado = direcionado

    def adicionar_vertice(self, v):
        if v not in self.adjacencias:
            self.adjacencias[v] = []

    def adicionar_aresta(self, u, v):
        self.adicionar_vertice(u)
        self.adicionar_vertice(v)
        self.adjacencias[u].append(v)
        if not self.direcionado:
            self.adjacencias[v].append(u)

    def bfs(self, inicio):
        """Busca em largura (BFS)."""
        visitados = set()
        fila = [inicio]
        ordem = []
        while fila:
            v = fila.pop(0)
            if v not in visitados:
                visitados.add(v)
                ordem.append(v)
                fila.extend(self.adjacencias.get(v, []))
        return ordem

    def dfs(self, inicio):
        """Busca em profundidade (DFS)."""
        visitados = set()
        ordem = []

        def _dfs(v):
            visitados.add(v)
            ordem.append(v)
            for vizinho in self.adjacencias.get(v, []):
                if vizinho not in visitados:
                    _dfs(vizinho)

        _dfs(inicio)
        return ordem
