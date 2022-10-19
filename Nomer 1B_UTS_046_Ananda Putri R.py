#AnandaPutriRahmadani_21091397046
#Multi neuron mengggunakan numpy

#inisialisasi numpy
import numpy as np

#Inisialisasi variabel
#memasukan nilai variabel layer feature 10
inputs = [3.0, 8.0, 2.0, 9.0, 4.0, 1.0, 7.0, 5.0, 6.0, 10.0]

#Memberikan nilai bobot pada variabel sesuai dengan jumlah input
#memasukan jumlah weight sesuai dengan jumlah neuron
weights = [[0.2, 0.4, 0.6, 0.8, 0.9, 0.2, 0.1, 0.3, 0.5, 0.4],
[0.3, 0.2, 0.1, 0.2, 0.8, 0.3, 0.4, 0.7, 0.9, 0.6],
[0.2, 0.6, 0.7, 0.1, 0.4, 0.9, 0.6, 0.8, 0.9, 0.1],
[1, 0.5, 0.2, 0.3, 0.2, 4, 8, 0.7, 0.35, 0.27],
[9, 0.3, 0.4, 0.2, 0.3, 0.44, 7, 0.1, 0.21, 0.30]]

#inisialisasi bias sesuai dengan neuron yang ditentukan
biases = [4.0, 2.0, 3.0, 7.0, 8.0]

#ouput
layer_outputs = np.dot(weights, inputs) + biases

#print output
print(layer_outputs)