#Ananda Putri Rahmadani_21091397046
#Single Neuron Menggunakan numpy

#inisialisasi numpy
import numpy as np

#inisialisasi variabel
#memasukan nilai variabel layer feature 10
inputs = [3.0, 2.0, 2.2, 6.3, 4.1, 4.5, 6.8, 9.7, 3.4, 6.5]

#memberikan nilai bobot pada variabel sesuai dengan jumlah input
weights = [2.6, 4.2, 9.3, 1.6, 1.3, 9.2, 2.0, 1.0, 8.6, 3.4]

#inisialisasi bias
bias = 6.0

#output
outputs = np.dot(weights, inputs) + bias

#print outputs
print(outputs)
