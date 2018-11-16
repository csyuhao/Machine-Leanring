import numpy as np
import matplotlib.pyplot as plt

# size of the points dataset
m = 20

# Points x-coordinate and dummy value(x0, x1)
x0 = np.ones((m,1))
x1 = np.arange(1, m + 1).reshape(m, 1)
X = np.hstack((x0, x1))

# Points y-coordinate
y = np.array([
    3,4,5,5,2,4,7,8,11,8,12,
    11,13,13,16,17,18,17,19,21
]).reshape(m, 1)

# The Learning Rate alpha.
alpha = 0.01

def error_function(theta, X, y):
    '''Error function J definition.'''
    diff = np.dot(X, theta) - y
    return (1./(2 * m)) * np.dot(np.transpose(diff), diff)

def gradient_function(theta, X, y):
    '''Gradient of the function J definition.'''
    diff = np.dot(X, theta) - y
    return (1./m) * np.dot(np.transpose(X), diff)

def gradient_descent(X, y, aplha):
    '''Perform gradient descent.'''
    theta = np.array([1,1]).reshape(2,1)
    gradient = gradient_function(theta, X, y)
    while not np.all(np.absolute(gradient) <= 1e-5):
        theta = theta - gradient * aplha
        gradient = gradient_function(theta, X, y)
    
    return theta

optimal = gradient_descent(X, y, alpha)
print("Optimal:", optimal)
print("error function:", error_function(optimal, X, y)[0,0])

z = np.dot(X, optimal)
plt.scatter(x1,y,s=10)
plt.plot(x1,z,'b')
plt.show()