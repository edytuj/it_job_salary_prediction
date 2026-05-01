def format_salary(x):
    value = int(round(x))
    return f"{value:,}".replace(",", " ") + " PLN"
