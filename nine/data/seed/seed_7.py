
def contar_palavras(texto):
    """Retorna um dicionario com a frequencia de cada palavra."""
    palavras = texto.lower().split()
    frequencia = {}
    for palavra in palavras:
        palavra = palavra.strip(".,!?;:()[]{}")
        if palavra:
            frequencia[palavra] = frequencia.get(palavra, 0) + 1
    return frequencia


def top_n_palavras(texto, n=10):
    """Retorna as N palavras mais frequentes em um texto."""
    freq = contar_palavras(texto)
    return sorted(freq.items(), key=lambda x: x[1], reverse=True)[:n]


def remover_duplicatas(lista):
    """Remove elementos duplicados mantendo a ordem original."""
    vistos = set()
    resultado = []
    for item in lista:
        if item not in vistos:
            vistos.add(item)
            resultado.append(item)
    return resultado
