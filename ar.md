# Array Regression Coeficient (AR)

$$
\rho = 1 - \frac{\displaystyle\sum d_i/((R_i^2+R_i)/n+n/2-R_i-1/2)}{n}
$$

donde:
- $R_i$ es el rango de la variable real.
- $P_i$ es el rango de la variable predicha.
- $d_i=|Ri-Pi|$ es la diferencia entre los rangos de las dos variables para cada observación.
- $n$ es el número de observaciones.


El denominador es la distancia promedio dado $R_i$ y $n$: 

$$
D(R_i,n) = \frac{R_i^2 + R_i}{n}\;+\;\frac{n}{2}-R_i-\frac{1}{2}
$$

Luego

$$
\rho =1 -\frac{1}{n}\sum_{i=1}^n\frac{d_i}{D(R_i,n)}
$$

### Ejemplo
Vamos a hacer numericamente un caso simple, donde $n=2$ y veremos que en los dos casos posibles se obtiene el 1 y -1.

Se debe interpretar como que 1 es una predicción perfecta y -1 es una predicción totalmente errónea.

Para $n=2$ y $R=(0,1)$:

$$
D(0,2) =\frac{0^2+0}{2} + \frac{2}{2} - 0 - \frac12
=0 + 1 - 0.5
=0.5
$$

$$
D(1,2) =\frac{1^2+1}{2} + \frac{2}{2} - 1 - \frac12
=1 + 1 - 1 - 0.5
=0.5
$$


Si $P=(0,1)$ significa que la predicción es perfecta en términos de orden.

$$
\rho = 1 -\frac{0/0.5+0/0.5}{2} = 1
$$


Si $P=(1,0)$

$$
\rho = 1 -\frac{1/0.5+1/0.5}{2} = 1 -\frac{2+2}{2} = 1 - 2 = -1
$$

¿Qué significa el denominador $D(R_i,n)$?

Es dos veces la distancia promedio ponderada por la probabilidad.
 
Supongamos $D(0,2)=0.5$

Si está en la posición 0: 0 de distancia por 0.5 de probabilidad.

Si está en la posición 1: 1 de distancia por 0.5 de probabilidad.

$D(0,2)= 2 * (0*0.5 + 1*0.5)/2 = 0.5$

Eso significa que ponderamos el error de i por 2 veces el promedio. 

La fómula está generalizada:

Centremosnó en la parte entre paréntesis de un tamaño n y $R_i$ 

Hasta llegar a la posición $R_i$ en este caso 0, la parte izquierda y la parte derecha se desgolosa como:

$$
D(R_i,n)=(R_i+R_i-1+...+0+1+...+(R_i-n))/n
$$

Son dos sumatorios uno hasta $R_i$ y otro desde $R_i-n$:

$$
D(R_i,n)
=\frac{\displaystyle\sum_{j=0}^{R_i}j \;+\;\sum_{j=0}^{R_i-n}j}{n}.
$$

1. **Fórmulas cerradas de las sumas**

$$
\sum_{j=0}^{R_i}j = \frac{R_i(R_i+1)}2,
\qquad
\sum_{j=0}^{R_i-n}j = \frac{(R_i-n)(R_i-n+1)}2.
$$

2. **Sustitución en $D(R_i,n)$**

$$
D(R_i,n)
=\frac{\displaystyle\frac{R_i(R_i+1)}2 + \frac{(R_i-n)(R_i-n+1)}2}{n}
=\frac{R_i(R_i+1)+(R_i-n)(R_i-n+1)}{2\,n}.
$$

3. **Expansión del numerador**

$$
\begin{aligned}
R_i(R_i+1)&=R_i^2+R_i,\\
(R_i-n)(R_i-n+1)&=R_i^2 + R_i -2nR_i + n^2 - n,\\
\Rightarrow\;R_i(R_i+1)+(R_i-n)(R_i-n+1)
&=2R_i^2 +2R_i -2nR_i + n^2 - n.
\end{aligned}
$$

4. **Dividir por $2n$**

$$
D(R_i,n)
=\frac{2R_i^2 +2R_i -2nR_i + n^2 - n}{2\,n}.
$$

5. **Separar y simplificar**

$$
\begin{aligned}
D(R_i,n)
&=\frac{2R_i^2+2R_i}{2n} \;-\;\frac{2nR_i}{2n}\;+\;\frac{n^2}{2n}\;-\;\frac{n}{2n}\\
&=\frac{R_i^2+R_i}{n}\;-\;R_i\;+\;\frac{n}{2}\;-\;\frac12.
\end{aligned}
$$

Por tanto, la forma cerrada final es

$$
\boxed{D(R_i,n)
=\frac{R_i^2+R_i}{n} \;-\; R_i \;+\;\frac{n}{2}\;-\;\frac12.}
$$
