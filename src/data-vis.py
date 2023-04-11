"""
Matthew Keefer, Leroy Schaigorodsky, Alan Sourek, Ceara Zhang
DS 4400 / Hockey Game Analysis
Final Project
Date Created: 3/29/23
Last Updated:
"""
import matplotlib as plt
# plot the weight vector
plt.bar(range(len(w)), w)
plt.xlabel('Attribute Index')
plt.ylabel('Weight')
plt.title('Perceptron Weight Vector')
plt.show()