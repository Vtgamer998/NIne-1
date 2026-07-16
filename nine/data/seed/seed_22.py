
def formatar_numero(n, casas=2):
    """Formata numero com separador de milhares."""
    return f"{n:,.{casas}f}"


def por_extenso(n):
    """Converte numero (0-99) para portugues."""
    unidades = ["zero", "um", "dois", "tres", "quatro", "cinco",
                "seis", "sete", "oito", "nove"]
    especiais = ["dez", "onze", "doze", "treze", "quatorze", "quinze",
                 "dezesseis", "dezessete", "dezoito", "dezenove"]
    dezenas = ["", "", "vinte", "trinta", "quarenta", "cinquenta",
               "sessenta", "setenta", "oitenta", "noventa"]
    if n < 10:
        return unidades[n]
    if n < 20:
        return especiais[n - 10]
    d = n // 10
    u = n % 10
    if u == 0:
        return dezenas[d]
    return f"{dezenas[d]} e {unidades[u]}"
