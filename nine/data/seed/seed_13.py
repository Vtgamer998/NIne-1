
def celsius_para_fahrenheit(c):
    return (c * 9/5) + 32


def fahrenheit_para_celsius(f):
    return (f - 32) * 5/9


def km_para_milhas(km):
    return km * 0.621371


def milhas_para_km(milhas):
    return milhas / 0.621371


def segundos_para_hms(segundos):
    """Converte segundos para horas, minutos e segundos."""
    h = segundos // 3600
    m = (segundos % 3600) // 60
    s = segundos % 60
    return h, m, s


def hms_para_segundos(h, m, s):
    return h * 3600 + m * 60 + s
