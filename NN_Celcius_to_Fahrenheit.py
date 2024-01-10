import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 



celsius = np.array([-80, -40, 0, 7, 13, 19, 29], dtype=float)

fahrenheit = np.array([-112, -40, 32, 44.6, 55.4, 66.2, 84.2], dtype=float)

print(celsius)
print(fahrenheit)


Mi_capa = tf.keras.layers.Dense(units=1, input_shape=[1])

print(Mi_capa)

Mi_modelo = tf.keras.Sequential([Mi_capa])

print(Mi_modelo)


Mi_modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.05), 
    loss='mean_squared_error'
)


Mi_modelo.summary()

print("Comenzando entrenamiento...")
Mi_historial = Mi_modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False)
print("Modelo entrenado!")



plt.xlabel("# Epocas")
plt.ylabel("Magnitud de pérdida")
plt.grid()
plt.plot(Mi_historial.history["loss"], label='Loss')
plt.legend()


print("Variables internas del modelo")
print(Mi_capa.get_weights())


print("Hagamos unas predicciones!")
resultado = Mi_modelo.predict([37.0])
print("El primer resultado es " + str(resultado) + " fahrenheit!")
resultado = Mi_modelo.predict([100.0])
print("El segundo resultado es " + str(resultado) + " fahrenheit!")