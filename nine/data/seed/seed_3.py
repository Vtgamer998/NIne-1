
class Pilha:
    """Implementacao de uma pilha (LIFO) em Python."""

    def __init__(self):
        self.items = []

    def empilhar(self, item):
        self.items.append(item)

    def desempilhar(self):
        if self.vazia():
            return None
        return self.items.pop()

    def topo(self):
        if self.vazia():
            return None
        return self.items[-1]

    def vazia(self):
        return len(self.items) == 0

    def tamanho(self):
        return len(self.items)


class Fila:
    """Implementacao de uma fila (FIFO) em Python."""

    def __init__(self):
        self.items = []

    def enfileirar(self, item):
        self.items.append(item)

    def desenfileirar(self):
        if self.vazia():
            return None
        return self.items.pop(0)

    def vazia(self):
        return len(self.items) == 0

    def tamanho(self):
        return len(self.items)
