import numpy as np

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x >= 0, x, alpha * x) # 0 included in the function domain``

def tanh(x):
    return np.tanh(x)

def main():
    random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]

    random_values = np.array(random_values)
    
    print("Original values:", random_values)
    print("ReLU:", relu(random_values))
    print("Leaky ReLU:", leaky_relu(random_values))
    print("Tanh:", tanh(random_values))

if __name__ == "__main__":
    main()
