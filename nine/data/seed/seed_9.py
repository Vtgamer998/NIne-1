
def ler_arquivo(caminho):
    """Le um arquivo de texto e retorna seu conteudo."""
    with open(caminho, "r", encoding="utf-8") as f:
        return f.read()


def escrever_arquivo(caminho, conteudo):
    """Escreve conteudo em um arquivo de texto."""
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(conteudo)


def listar_arquivos(diretorio, extensao=".py"):
    """Lista todos os arquivos com determinada extensao em um diretorio."""
    import os
    arquivos = []
    for root, _, files in os.walk(diretorio):
        for f in files:
            if f.endswith(extensao):
                arquivos.append(os.path.join(root, f))
    return arquivos
