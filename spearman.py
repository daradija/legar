from itertools import permutations
#from scipy.stats import spearmanr

def spearmanr(x, y):
    """
    Calculate Spearman's rank correlation coefficient.
    """
    rx= sorted(range(len(x)), key=lambda i: x[i])
    ry= sorted(range(len(y)), key=lambda i: y[i])
    n = len(x)
    d_squared = sum((rank_x - rank_y) ** 2 for rank_x, rank_y in zip(rx, ry))
    return 1 - (6 * d_squared) / (n * (n**2 - 1)), None


for size in range(3,9):
    # x fijo
    x = list(range(size))

    # Todas las permutaciones de y
    perms = list(permutations(x))
    coeficientes = []

    min_coef = 1.0
    max_coef = -1.0
    min_y = None
    max_y = None
    
    for y in perms:
        coef, _ = spearmanr(x, y)
        coeficientes.append(coef)
        
        if coef < min_coef:
            min_coef = coef
            min_y = y
        if coef > max_coef:
            max_coef = coef
            max_y = y
        # print(f"y = {y}, Spearman = {coef:.3f}")

    # Promedio de los coeficientes
    promedio = sum(coeficientes) / len(coeficientes)
    print(f"\nPromedio de Spearman para {size} : {promedio:.3f}")
    print(f"Mínimo: {min_coef:.3f} para y = {min_y}")
    print(f"Máximo: {max_coef:.3f} para y = {max_y}")

    
