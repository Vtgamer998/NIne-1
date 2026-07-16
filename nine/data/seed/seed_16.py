
def calcular_imc(peso, altura):
    """Calcula o Indice de Massa Corporal."""
    imc = peso / (altura ** 2)
    if imc < 18.5:
        classificacao = "Abaixo do peso"
    elif imc < 25:
        classificacao = "Peso normal"
    elif imc < 30:
        classificacao = "Sobrepeso"
    else:
        classificacao = "Obesidade"
    return imc, classificacao


def juros_compostos(principal, taxa, meses):
    """Calcula montante com juros compostos."""
    return principal * (1 + taxa) ** meses


def parcelamento(valor, parcelas, juros=0.02):
    """Calcula valor de parcelas com juros."""
    return valor * (juros * (1 + juros) ** parcelas) / ((1 + juros) ** parcelas - 1)
