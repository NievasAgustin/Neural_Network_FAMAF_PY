import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 



celsius = np.array([-80, -40, 0, 7, 13, 19, 29], dtype=float)

fahrenheit = np.array([-112, -40, 32, 44.6, 55.4, 66.2, 84.2], dtype=float)

print(celsius)
print(fahrenheit)

entrada = tf.keras.layers.Dense(units=1, input_shape=[1])
oculta = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)
Mi_modelo = tf.keras.Sequential([entrada, oculta, salida])

print(entrada,oculta,salida)

print(Mi_modelo)


Mi_modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.05), 
    loss='mean_squared_error'
)


Mi_modelo.summary()

print("Comenzando entrenamiento...")
Mi_historial = Mi_modelo.fit(fahrenheit, celsius , epochs=1000, verbose=False)
print("Modelo entrenado!")


print("Printing loss figure")
plt.xlabel("# Epocas")
plt.ylabel("Magnitud de pérdida")
plt.grid()
plt.plot(Mi_historial.history["loss"], label='Loss')
plt.legend()
plt.show()
print("Did it work?")

print(entrada.get_weights())
print(oculta.get_weights())
print(salida.get_weights())


print("Hagamos unas predicciones!")
resultado = Mi_modelo.predict([37.0])
print("El primer resultado es " + str(resultado) + " Celcius!")
resultado = Mi_modelo.predict([100.0])
print("El segundo resultado es " + str(resultado) + " Celcius!")