from sklearn import  linear_model
import numpy as np
import matplotlib.pyplot as plt

# Create linear regression object
liner_regression = linear_model.LinearRegression()

N = 50
#X  = np.linspace(2.0, 10.0, num=N)
X = np.arange(2, N, 3)

# make the y variable as 2 time of the X value
y_2X = map(lambda x: x * 2, X)

# add some random guassian noise to y. 
mu, sigma = 0, 5 # mean and standard deviation
noise = np.random.normal(mu, sigma, len(y_2X))
y_with_noise = y_2X + noise

#plot x and y with and without noise
plt.plot(X, y_with_noise, 'o', color='black')
plt.plot(X, y_2X, 'x', color='blue')
plt.show()
print X.shape
X = np.reshape(X, (len(X),1))
y_with_noise = np.reshape(y_with_noise, (len(y_with_noise), 1))

# now fit the X and y_with_noise data
liner_regression.fit(X, y_with_noise)

print "Predicting X=100", str(liner_regression.predict(100))
# The coefficients
print('Coefficients: \n', liner_regression.coef_)

# The mean square error
error = np.mean((liner_regression.predict(X) - y_2X) ** 2)
print("Residual sum of squares: %.2f"
      % error)
z = np.sqrt(error)
print z
print('Variance score: %.2f' % liner_regression.score(X, y_with_noise))

plt.plot(X, y_2X, '+', color='blue')
plt.plot(X, y_with_noise, 'o',  color='green')
plt.plot(X, liner_regression.predict(X), color='red',
         linewidth=3)

plt.xticks(())
plt.yticks(())

#plt.show()
