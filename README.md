# Logic Estimator Guided by Array Regresion (LEGAR)

Este indicador tiene similitudes con el indicador de Spearman.

## Spearman

El coeficiente de correlación de Spearman es una medida no paramétrica de la correlación entre dos variables. Se basa en el rango de los datos en lugar de los valores absolutos, lo que lo hace menos sensible a los valores atípicos.
El coeficiente de correlación de Spearman varía entre -1 y 1, donde:
- 1 indica una correlación positiva perfecta (a medida que una variable aumenta, la otra también lo hace).
- -1 indica una correlación negativa perfecta (a medida que una variable aumenta, la otra disminuye).
- 0 indica que no hay correlación.

La fórmula para calcular el coeficiente de correlación de Spearman es:

$$
\rho = 1 - \frac{6 \displaystyle\sum d_i^2}{n(n^2 - 1)}
$$

donde:
- $d_i$ es la diferencia entre los rangos de las dos variables para cada observación.
- $n$ es el número de observaciones.
- $\sum d_i^2$ es la suma de los cuadrados de las diferencias de rango.

Una prueba curiosa es comprobar si la media de todas las permutaciones da 0. Lo que que no está sesgado. 

En spearman.py se realiza diche prueba. 
