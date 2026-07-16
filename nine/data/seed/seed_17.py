
class NoLista:
    def __init__(self, valor):
        self.valor = valor
        self.proximo = None


class ListaLigada:
    """Lista ligada simples."""

    def __init__(self):
        self.cabeca = None
        self._tamanho = 0

    def inserir(self, valor):
        novo = NoLista(valor)
        novo.proximo = self.cabeca
        self.cabeca = novo
        self._tamanho += 1

    def remover(self, valor):
        atual = self.cabeca
        anterior = None
        while atual and atual.valor != valor:
            anterior = atual
            atual = atual.proximo
        if atual is None:
            return False
        if anterior is None:
            self.cabeca = atual.proximo
        else:
            anterior.proximo = atual.proximo
        self._tamanho -= 1
        return True

    def buscar(self, valor):
        atual = self.cabeca
        while atual:
            if atual.valor == valor:
                return True
            atual = atual.proximo
        return False

    def tamanho(self):
        return self._tamanho

    def para_lista(self):
        resultado = []
        atual = self.cabeca
        while atual:
            resultado.append(atual.valor)
            atual = atual.proximo
        return resultado
