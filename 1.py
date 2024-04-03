import numpy as np
import matplotlib.pyplot as plt

# Sigmoid Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ReLU Activation Function
def relu(x):
    return np.maximum(0, x)

# Leaky ReLU Activation Function
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# Tanh Activation Function
def tanh(x):
    return np.tanh(x)

# Generate input values
x = np.linspace(-5, 5, 100)

# Generate output values for each activation function
y_sigmoid = sigmoid(x)
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)
y_tanh = tanh(x)


# Plotting
plt.figure(figsize=(10, 6))

# Sigmoid plot
plt.subplot(2, 2, 1)
plt.plot(x, y_sigmoid, label='Sigmoid')
plt.title('Sigmoid Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()

# ReLU plot
plt.subplot(2, 2, 2)
plt.plot(x, y_relu, label='ReLU', color='orange')
plt.title('ReLU Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()

# Leaky ReLU plot
plt.subplot(2, 2, 3)
plt.plot(x, y_leaky_relu, label='Leaky ReLU', color='green')
plt.title('Leaky ReLU Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()

# Tanh plot
plt.subplot(2, 2, 4)
plt.plot(x, y_tanh, label='Tanh', color='red')
plt.title('Tanh Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()

plt.tight_layout()
plt.show()
