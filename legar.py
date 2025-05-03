import pandas as pd
import numpy as np
import math
import random
import itertools
#import tensorflow as tf


class ARCoeficient:
	def __init__(self):
		self.Y = []
		self.YP = []
		self.index0 = []

	def add(self, Y, YP, index):
		if math.isfinite(Y) and math.isfinite(YP):
			self.Y.append(Y)
			self.YP.append(YP)
			self.index0.append(index)

	def calculateByAdd(self):
		self.calculateByNumpy(np.array(self.Y), np.array(
			self.YP), np.array(self.index0, dtype=np.int64))

	def calculateByNumpy(self, Y, YP, idex):
		# Verificar que Y y YP tienen el mismo shape
		if Y.shape != YP.shape:
			raise ValueError("Y e YP deben tener el mismo shape")
		

		# Crear un array de 0 a N-1, donde N es la longitud de Y (o YP)
		N = len(Y)  # O YP, ya que tienen el mismo tamaño
		index_array = np.arange(N)

		# Unir las columnas Y, YP, el índice y las columnas vacías en una nueva matriz
		self.result_matrix1 = np.column_stack((Y, YP, index_array))
		self.id0 = idex
		# Crear una máscara para filas sin NaN en Y o YP
		self.mask = ~np.isnan(Y) & ~np.isnan(YP)
		self.remask()

	def top(self,n,ar1,yp,rentabilidadMinimaDiaria,rentabilidadMaximaDiaria,prediccionMinimaDiaria,prediccionMaximaDiaria):
		join=[(a,b,c) for a,b,c in zip(ar1,yp,self.ar)]
		join.sort(key=lambda x: x[0],reverse=0<n) #a
		n=abs(n)
		s=0
		n2=0
		for a,b,c in join:
			if a>prediccionMaximaDiaria:
				continue
			if a<prediccionMinimaDiaria:
				continue
			if b>=rentabilidadMinimaDiaria and b<=rentabilidadMaximaDiaria:
				s+=c
				n2+=1
				if n==n2:
					break
		if n2==0:
			return 0.5
		return s/n2
		

	def calculateByTensorFlow(self, Y, YP, idex):
		# Verificar que Y y YP tienen el mismo shape
		if Y.shape != YP.shape:
			raise ValueError("Y e YP deben tener el mismo shape")

		# Crear un tensor de 0 a N-1, donde N es la longitud de Y (o YP)
		N = tf.shape(Y)[0]
		index_array = tf.range(N)

		# Unir las columnas Y, YP, el índice y las columnas vacías en una nueva matriz
		self.result_matrix1 = tf.stack([Y, YP, tf.cast(index_array, Y.dtype)], axis=1)
		self.id0 = idex

		# Crear una máscara para filas sin NaN en Y o YP
		mask_Y = tf.math.logical_not(tf.math.is_nan(Y))
		mask_YP = tf.math.logical_not(tf.math.is_nan(YP))
		self.mask = tf.math.logical_and(mask_Y, mask_YP)

		# Llamar a la función remask, asumiendo que ya está definida en tu clase
		self.remaskTF()

	def remask(self):
		result_matrix2 = self.result_matrix1[self.mask]
		if self.id0 is not None:
			self.id = self.id0[self.mask]

		N = len(result_matrix2)

		if N < 2:
			self.mean = 0
			return

		# Crear dos columnas más con valores "vacíos" (NaN por ejemplo)
		empty_columns = np.empty((N, 2))
		result_matrix = np.column_stack((result_matrix2, empty_columns))

		# ordenar por Y, y en caso de igualdad, por index_array
		# Nota: index_array ya está incluido en result_matrix, por lo que se usa directamente para ordenar
		ordenY = np.lexsort((result_matrix[:, 2], result_matrix[:, 0]))
		result_matrix[:, 3] = ordenY
		ordenYP = np.lexsort((result_matrix[:, 2], result_matrix[:, 1]))
		result_matrix[:, 4] = ordenYP

		self.ordenYP = ordenYP

		# diferencia_abs = np.abs(result_matrix[:, 3] - result_matrix[:, 4]) # este paso nos lo podemos ahorrar

		# Calcular el total acumulado de las diferencias
		# total_acumulado = np.sum(diferencia_abs)

		# locPrepadador = 1.0-total_acumulado/(N*(N/2))

		np.seterr(all='raise')

		# if N == 3:
		# 	denominador = [2, 4/3, 2]
		# 	# Media de distancias x 2 1/3+0/3 +1/3
		# if N == 4:
		# 	denominador = [3, 2, 2, 3]
		# if N == 5:
		# 	denominador = [4, (1/5+0/5+1/5+2/5+3/5)*2, (2/5+1/5+0/5+1/5+2/5)*2, (1/5+0/5+1/5+2/5+3/5)*2, 4]
            
		# denominador = []
		# #for i in range(N):
		# for i in ordenY:
		# 	value = sum(abs(i - j) for j in range(N)) / N * 2
		# 	denominador.append(value)
               
		#indices = np.arange(N)  # Crea un array de 0 a N-1
		i = ordenY
		# distancias = np.abs(i - i[:, np.newaxis])  # Calcula la matriz de distancias absolutas
		# denominador = np.sum(distancias, axis=1) / N * 2  # Suma las distancias por fila, promedia y multiplica por 2
  
		denominador=((2*i*i)+(2*i))/N+N-(2*i)-1

		numerador = np.abs(result_matrix[:, 3] - result_matrix[:, 4])
		# denominador=np.where(ordenY==ordenYP,1, np.where(ordenY > ordenYP ,ordenY,N-1-ordenY))
		self.ar = np.subtract(1, numerador/denominador).flatten()
		# except e:
		# 	print("Error",e)
		self.mean = np.mean(self.ar)



	def remaskTF(self):
		# Filtrar result_matrix1 usando la máscara
		result_matrix2 = tf.boolean_mask(self.result_matrix1, self.mask)
		
		if self.id0 is not None:
			self.id = tf.boolean_mask(self.id0, self.mask)

		N = tf.shape(result_matrix2)[0]

		if N < 2:
			self.mean = 0
			return

		# Crear dos columnas más con valores "vacíos" (NaN por ejemplo)
		empty_columns = tf.fill([N, 2], tf.constant(float('nan'), dtype=result_matrix2.dtype))
		result_matrix = tf.concat([result_matrix2, empty_columns], axis=1)

		# Ordenar por Y y, en caso de igualdad, por index_array
		ordenY = tf.argsort(tf.argsort(result_matrix[:, 0], stable=True) + tf.argsort(result_matrix[:, 2], stable=True))
		result_matrix = tf.tensor_scatter_nd_update(result_matrix, tf.expand_dims(tf.range(N), axis=1), tf.expand_dims(tf.cast(ordenY, result_matrix.dtype), axis=1))
		result_matrix = tf.tensor_scatter_nd_update(result_matrix, tf.expand_dims(tf.range(N), axis=1), tf.expand_dims(tf.cast(tf.argsort(tf.argsort(result_matrix[:, 1], stable=True) + tf.argsort(result_matrix[:, 2], stable=True)), result_matrix.dtype), axis=1))

		i = ordenY

		# Calcular el denominador
		denominador = ((2 * tf.cast(i, tf.float32) * tf.cast(i, tf.float32)) + (2 * tf.cast(i, tf.float32))) / tf.cast(N, tf.float32) + tf.cast(N, tf.float32) - (2 * tf.cast(i, tf.float32)) - 1

		numerador = tf.abs(result_matrix[:, 3] - result_matrix[:, 4])

		self.ar = tf.subtract(1.0, numerador / denominador).numpy().flatten()

		self.mean = tf.reduce_mean(self.ar)

class AutoAR:
	def __init__(self, save=True):
		self.fila = {}
		self.tabla = []
		self.save = save

	def param(self, label, value):
		if self.fila.get(label) != None:
			raise Exception("Error")
		self.fila[label] = value

	def tick(self):
		self.tabla.append(self.fila)
		# print(self.fila)
		self.fila = {}

	def think(self, target):
		if self.save:
			df = pd.DataFrame(self.tabla)
			df.to_csv('autoar.csv', index=False)
		if len(self.tabla) == 0:
			return
		xlabel = self.tabla[0].keys()
		for xl in xlabel:
			if xl == target:
					continue
			
			mean=self.predict(xl,target)
			print(f"{xl}: {mean*100-50:.2f}%")
      
	def predict(self, source, target,limite=0):
		x = []
		y = []
		tabla=self.tabla
		if 0<limite:
			tabla=tabla[-limite:]
		for f in tabla:
			x.append(f[source])
			y.append(f[target])
		arc = ARCoeficient()
		arc.calculateByNumpy(np.array(y), np.array(
			x), np.array(range(len(x)), dtype=np.int64))
		return arc.mean	


class Measurer:
	def __init__(self):
		self.data = []

	def add(self, value):
		self.data.append(value)

	def mean(self, label="Random:"):
		m= np.mean(self.data)
		print(f"{label} {m * 100-50:.2f}%")
		return m



def switchRandom(x):
	for _ in range(len(x)*5):
		i = random.randint(0, len(x)-1)
		j = random.randint(0, len(x)-1)
		x[i], x[j] = x[j], x[i]
	return x


def rndARV2(tam):
      x=[i for i in range(tam)]
      y=[i for i in range(tam)]
      x=switchRandom(x)
      y=switchRandom(y)
      arc = ARCoeficient()
      arc.calculateByNumpy(np.array(y), np.array(
			 x), np.array(range(len(x)), dtype=np.int64))
      return arc.mean

def rndAR(tam):
    x = [-1 for i in range(tam)]
    y = [-1 for i in range(tam)]
    for i in range(tam):
        while True:
            x[i] = random.randint(0, tam-1)
            if y[x[i]] == -1:
                y[x[i]] = x[i]
                break

    arc = ARCoeficient()
    arc.calculateByNumpy(np.array(y), np.array(
        x), np.array(range(len(x)), dtype=np.int64))
    # print(f"Random: {arc.mean * 100-50:.2f}%")
    return arc.mean


def permAR(permutacion):
    tam = len(permutacion)
    x = [i for i in range(tam)]
    y = permutacion
    arc = ARCoeficient()
    arc.calculateByNumpy(np.array(y), np.array(
        x), np.array(range(len(x)), dtype=np.int64))
    return arc.mean

def legar(x,y):
    arc = ARCoeficient()
    arc.calculateByNumpy(np.array(y), np.array(
        x), np.array(range(len(x)), dtype=np.int64))
    return arc.mean*2-1

def elAzar():
    muestras = 100000
    tam = 2
    for _ in range(10):
        m = Measurer()
        if tam <= 9:
            nums = list(range(tam))
            for perm in itertools.permutations(nums):
                m.add(permAR(list(perm)))
        else:
            for i in range(muestras):
                m.add(rndARV2(tam))
        m.mean(label=str(tam))
        tam += 1

def explorar():
    df = pd.read_csv('autoar.csv')
    # df=df[df['Ventana'] > df["Ventana"].mean()]
    #df = df[df["Filtro"] == 0.55]

    aar = AutoAR(False)
    aar.tabla = df.to_dict("records")
    print(aar.predict("Filtro","AR"))
    aar.think("Simulation")

if __name__ == "__main__":
    #explorar() 
    elAzar()
