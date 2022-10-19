#AnandaPutriRahmadani_21091397046
#Multi Neuron Batch Input

#inisialisasi numpy
import numpy as np

#inisialisasi variabel
#memasukan nilai variabel layer feature 10 dengan batch sejumlah 6
inputs = [
    [2.0, 3.7, 8.0, 2.7, 0.0, 31.5, 5.9, 3.5, 5.0, 5.5],
    [0.3, 0.1, 2.2, 2.6, 3.2, 3.4, 0.2, 2.4, 9.2, 7.4],
    [1.4, 9.5, 18.0, 20.5, 32.1, 60.12, 33.7, 67.1, 76.0, 50.5],
    [6.0, 3.4, 2.6, 7.8, 3.6, 3.8, 4.6, 4.8, 5.6, 5.8],
    [1.4, 0.3, 7.2, 5.0, 8.2, 6.1, 9.2, 9.4, 27.3, 0.4],
    [13.2, 17.3, 14.5, 10.5, 38.1, 12.6, 11.7, 3.23, 59.2, 82.4],
]

#Memberi nilai bobot pada variabel sesuai dengan jumlah input
#memasukan jumlah weight sesuai dengan jumlah neuron dengan batch sejumlah 5
weights = [
    [6.0, 4.8, 8.4, 2.5, 0.1, 3.5, 9.7, 4.5, 6.2, 15.5],
    [7.4, 9.7, 4.10, 2.84, 3.52, 38.4, 45.2, 4.4, 5.2, 5.4],
    [3.3, 6.1, 2.3, 10.9, 31.6, 3.82, 4.26, 4.8, 56.6, 55.8],
    [5.8, 4.3, 4.2, 7.8, 0.2, 7.4, 3.5, 0.7, 40.3, 71.1],
    [5.1, 13.7, 30.6, 42.7, 95.1, 12.3, 29.0, 40.7, 28.1, 93.11]
]
#inisialisasi bias sesuai dengan neuron yang ditentukan 
biases = [4.7, 2.8, 1.0, 9.6, 3.1]

#outputs
outputs = np.dot(inputs, np.array(weights) . T) + biases

#print output
print(outputs)