
import time
from contextlib import contextmanager


@contextmanager
def temporizador(nome="bloco"):
    """Context manager para medir tempo de execucao."""
    inicio = time.time()
    yield
    fim = time.time()
    print(f"{nome}: {fim-inicio:.4f}s")


class TabelaHash:
    """Implementacao simples de tabela hash."""
    def __init__(self, tamanho=100):
        self.tamanho = tamanho
        self.tabela = [[] for _ in range(tamanho)]

    def _hash(self, chave):
        return hash(chave) % self.tamanho

    def inserir(self, chave, valor):
        indice = self._hash(chave)
        for par in self.tabela[indice]:
            if par[0] == chave:
                par[1] = valor
                return
        self.tabela[indice].append([chave, valor])

    def buscar(self, chave):
        indice = self._hash(chave)
        for par in self.tabela[indice]:
            if par[0] == chave:
                return par[1]
        return None

    def remover(self, chave):
        indice = self._hash(chave)
        for i, par in enumerate(self.tabela[indice]):
            if par[0] == chave:
                self.tabela[indice].pop(i)
                return True
        return False
