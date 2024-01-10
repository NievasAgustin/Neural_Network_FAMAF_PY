import tensorflow as tf
import matplotlib.pyplot as plt
import time

"""
ciudad (0), m^2, nº hab, nº planta, ascensor (0-1), exterior (0-1),
estado (0 no rehabilitado, 1 rehab, 2 nuevo), céntrico (0, 1)
"""

caracteristicas = [(0, 54, 2, 4, 0, 1, 0, 0),
                   (0, 152, 2, 4, 1, 1, 2, 1),
                   (0, 64, 3, 4, 0, 1, 0, 0),
                   (0, 154, 5, 4, 1, 1, 1, 1),
                   (0, 100, 1, 5, 1, 1, 1, 0),
                   (0, 140, 5, 2, 1, 1, 2, 0),
                   (0, 120, 3, 2, 1, 1, 1, 1),
                   (0, 70, 2, 3, 1, 1, 1, 0),
                   (0, 60, 2, 2, 0, 1, 1, 1),
                   (0, 129, 3, 18, 1, 1, 2, 1),
                   (0, 93, 1, 3, 1, 1, 2, 0),
                   (0, 52, 2, 2, 0, 1, 1, 1),
                   (0, 110, 3, 5, 1, 1, 1, 1),
                   (0, 63, 3, 2, 1, 1, 1, 0),
                   (0, 160, 1, 4, 1, 1, 2, 0)
                    ]

precios = [750, 2000, 650, 1500, 900, 1000, 1300, 750, 900, 1800, 975, 880, 1400, 750, 1050]


capaEntrada = tf.keras.layers.Dense(units=8, input_shape=[8])
capaOcultaUno  = tf.keras.layers.Dense(units=8)
capaSalida  = tf.keras.layers.Dense(units=1)


Mi_modelo = tf.keras.Sequential([capaEntrada, capaOcultaUno, capaSalida])

Mi_modelo.compile(
    optimizer = tf.keras.optimizers.Adam(0.01),
    loss = 'mse'
)

print('Inicio de entrenamiento...')
Mi_historial = Mi_modelo.fit(caracteristicas, precios, epochs=800, verbose=0) #False)
print('Modelo entrenado!')

plt.xlabel('# Época')
plt.ylabel('Mágnitud de pérdida')
plt.plot(Mi_historial.history['loss'])
plt.grid()
plt.xlim(0, 800)
plt.show()  


Mi_modelo.save('pisos_alquiler.keras')


for idx in range (len(precios)):
  print(' ')
  print('==========================================')
  solicitud = [caracteristicas[idx]]
  print(solicitud, precios[idx])
  print('-----------------------------------------')
  valor = Mi_modelo.predict(solicitud)
  print("El precio del piso %i solicitado tiene un valor de %.2f euros" % (idx,valor))
  print("Loss: ", precios[idx]-valor, ", with a porcentaje of : ", (precios[idx]-valor)/precios[idx])