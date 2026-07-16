
import re


def extrair_urls(texto):
    """Extrai URLs de um texto usando regex."""
    padrao = r"https?://[\w./?=&%-]+"
    return re.findall(padrao, texto)


def extrair_emails(texto):
    """Extrai enderecos de email de um texto."""
    padrao = r"[\w.-]+@[\w.-]+\.[a-zA-Z]{2,}"
    return re.findall(padrao, texto)


def limpar_texto(texto):
    """Remove pontuacao e normaliza espacos."""
    texto = re.sub(r"[^\w\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto.lower()


def dividir_em_frases(texto):
    """Divide texto em frases."""
    return re.split(r"[.!?]+", texto)
