import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.array(x)))

def main():
    random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]
    
    print("Original values:", random_values)
    print("Sigmoid:", sigmoid(random_values))

if __name__ == "__main__":
    main()
