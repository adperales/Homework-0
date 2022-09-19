# Homework-0

# --------------------------- 1st Linear Regression with Gradient Descent --------------------------------------------------

import pandas
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn.datasets as dt
from sklearn.model_selection import train_test_split
df = pandas.read_csv('D3.csv', header = 0)

x1_list = df['X1']
Y_list = df['Y']
m = len(x1_list)
h_theta = []
j_theta = []
theta_new = []
theta_old = []
theta_0 = 0
theta_1 = 1
a = 0.01
g = []
deriv_theta_0 = 0
deriv_theta_1 = 0

# Linear Regression
for i in range(m):
    h = theta_0 + (theta_1 * x1_list[i])
    h_theta.append(h)
    j = (((h_theta[i]*x1_list[i]) - Y_list[i])**2)
    j_theta.append(j/(2*m))

# Gradient Descent
for j in range(m):
    g = ((h_theta[j] - Y_list[j])*x1_list[j])/len(x1_list)
    deriv_theta_0 = ((1/m)*(x1_list[j])*((h_theta[j] * x1_list[j]) + h_theta[j] + (Y_list[j] - g)))
    theta_old.append(deriv_theta_0)
    deriv_theta_1 = theta_old[j] - (a*deriv_theta_0)
    theta_new.append(deriv_theta_1)
    
z = []
for i in range(len(x1_list)):
    a = (theta_new[i] - theta_old[i]) - h_theta[i]
    z.append(a)

# 1st Regression with Gradient Descent learning rate 0.000000015 
plt.scatter(x1_list,Y_list)
plt.title('X1 Dataset')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
#plt.show()
plt.plot(h_theta,theta_new)    
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('1st Regression')
#plt.show()
plt.plot(h_theta,j_theta)
plt.plot(h_theta,z)
plt.show()

# --------------------------- 2nd Linear Regression with Gradient Descent --------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn.datasets as dt
from sklearn.model_selection import train_test_split
df = pandas.read_csv('D3.csv', header = 0)

x2_list = df['X2']
x2_list = sorted(x2_list)
h_theta = []
j_theta = []
theta_new = []
theta_old = []
m = len(x2_list)
theta_0 = 0
theta_1 = 1
a = 0.09
g = []
deriv_theta_0 = 0
deriv_theta_1 = 0

for i in range(m):
    h = theta_0 + (theta_1 * x2_list[i])
    h_theta.append(h)
    j = (((h_theta[i]*x2_list[i]) - Y_list[i])**2)
    j_theta.append(j/(2*m))

# Gradient Descent
for j in range(m):
    g = ((h_theta[j] - Y_list[j]) * x2_list[j])/len(x2_list)
    deriv_theta_0 = ((1/(m)) * (x2_list[j])*((h_theta[j] * x2_list[j]) + h_theta[j] + (Y_list[j] - g)))
    theta_old.append(deriv_theta_0)
    deriv_theta_1 = theta_old[j] - (a * deriv_theta_0)
    theta_new.append(deriv_theta_1)
    
z = []
for i in range(len(x1_list)):
    a = (theta_new[i] - theta_old[i]) - h_theta[i]
    z.append(a)

# 2st Regression with Gradient Descent learning rate 0.01
plt.scatter(x2_list,Y_list)
plt.title('X2 Dataset')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
#plt.show()
plt.plot(h_theta,theta_new)    
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('2nd Regression')
#plt.show()
plt.plot(h_theta,j_theta)
plt.plot(h_theta,z)
plt.show()

# --------------------------- 3rd Linear Regression with Gradient Descent --------------------------------------------------

import pandas
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn.datasets as dt
from sklearn.model_selection import train_test_split
df = pandas.read_csv('D3.csv', header = 0)

x3_list = df['X3']
x3_list = sorted(x3_list)
h_theta = []
j_theta = []
theta_new = []
theta_old = []
m = len(x3_list)
theta_0 = 0
theta_1 = 1
a = 0.1
g = []
deriv_theta_0 = 0
deriv_theta_1 = 0

for i in range(m):
    h = theta_0 + (theta_1 * x3_list[i])
    h_theta.append(h)
    j = (((h_theta[i]*x3_list[i]) - Y_list[i])**2)
    j_theta.append(j/(2*m))

# Gradient Descent
for j in range(m):
    g = ((h_theta[j] - Y_list[j]) * x3_list[j])/len(x3_list)
    deriv_theta_0 = (((1/m)) * (x3_list[j] )* ((h_theta[j] * x3_list[j]) + x3_list[j] + (Y_list[j] - g)))
    theta_old.append(deriv_theta_0)
    deriv_theta_1 = theta_old[j] - (a * deriv_theta_0)
    theta_new.append(deriv_theta_1)
    
    
z = []
for i in range(len(x1_list)):
    a = (theta_new[i] - theta_old[i]) - h_theta[i]
    z.append(a)

# 3st Regression with Gradient Descent
plt.scatter(x3_list,Y_list)
plt.title('X3 Dataset')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
#plt.show()
plt.plot(h_theta,theta_new)    
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('3rd Regression')
#plt.show()
plt.plot(h_theta,j_theta)
plt.plot(h_theta,z)
plt.show()

# --------------------------- Linear Regression with Gradient Descent with multiple variables --------------------------------------------------

import pandas
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn.datasets as dt
from sklearn.model_selection import train_test_split
df = pandas.read_csv('D3.csv', header = 0)

x1_list = df['X1']
x2_list = df['X2']
x2_list = sorted(x2_list)
x3_list = df['X3']
x3_list = sorted(x3_list)
Y_list = df['Y']

m = len(x1_list)
h_theta_1 = []
j_theta_1 = []
theta_new_1 = []
theta_old_1 = []
theta_0_1 = 0
theta_1_1 = 1
g_1 = []
deriv_theta_0_1 = 0
deriv_theta_1_1 = 0

h_theta_2 = []
j_theta_2 = []
theta_new_2 = []
theta_old_2 = []
theta_0_2 = 0
theta_1_2 = 1
g_2 = []
deriv_theta_0_2 = 0
deriv_theta_1_2 = 0

h_theta_3 = []
j_theta_3 = []
theta_new_3 = []
theta_old_3 = []
theta_0_3 = 0
theta_1_3 = 1
g_3 = []
deriv_theta_0_3 = 0
deriv_theta_1_3 = 0

x_combined_list = []
h_theta_0 = []
h_theta_1 = []
j_theta = []
theta_new = []
theta_old = []
theta_0 = 0
theta_1 = 1
g = []
deriv_theta_0 = 0
deriv_theta_1 = 0

a = 0.01

for i in range(m):
    h_1 = theta_0_1 + (theta_1_1 * x1_list[i])
    h_theta_1.append(h_1)
    j_1 = (((h_theta_1[i]*x1_list[i]) - Y_list[i])**2)
    j_theta_1.append(j_1/(2*m))

for i in range(m):
    h_2 = theta_0_2 + (theta_1_2 * x2_list[i])
    h_theta_2.append(h_2)
    j_2 = (((h_theta_2[i]*x2_list[i]) - Y_list[i])**2)
    j_theta_2.append(j_2/(2*m))

for i in range(m):
    h_3 = theta_0_3 + (theta_1_3 * x3_list[i])
    h_theta_3.append(h)
    j_3 = (((h_theta_3[i]*x3_list[i]) - Y_list[i])**2)
    j_theta_3.append(j_3/(2*m))

# Combining Datasets    
for i in range(m):
    x_combined_list.append(x1_list[i] + x2_list[i] + x3_list[i])
    h_theta  = theta_0 + (theta_1 * x_combined_list[i])
    h_theta_1.append(h_theta)
    j = (((h_theta_1[i] * x_combined_list[i]) - Y_list[i])**2)
    j_theta.append(j/(2*m))
    h_theta_0.append(h_theta)
    
    
# Gradient Descent
for j in range(len(x_combined_list)):
    g = ((h_theta_1[j] - Y_list[j]) * x_combined_list[j])/(len(x_combined_list))
    deriv_theta_0 = (-2/m) * (((h_theta_0[j] * x_combined_list[j]) + h_theta_1[j-1] - (Y_list[j] - g))*x_combined_list[j])
    theta_old.append(deriv_theta_0)
    deriv_theta_1 = theta_old[j] - (a * deriv_theta_0)
    theta_new.append(deriv_theta_1)
    
z = []
for i in range(len(x_combined_list)):
    a = (theta_new[i] - theta_old[i]) - h_theta_0[i]
    z.append(a)
    
#plt.scatter(j_theta,Y_list)
plt.title('Dataset')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
#plt.show()
#plt.plot(h_theta_0,theta_new)
plt.title('Dataset')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
#plt.show()
#plt.plot(h_theta_0,j_theta)
plt.plot(h_theta_0,z)
plt.show()







