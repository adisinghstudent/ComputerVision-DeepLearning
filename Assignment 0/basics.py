import numpy as np

###############################
# Question 1: Differentiation #
###############################
# We'll approximate the derivative using a central difference:
def derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

# Define the functions (single-variable)
def f1(x):
    return 3*x**2 - 2*x + 5

def f2(x):
    return 5*x**3 - x**2 + 4*x - 1

def f3(x):
    return 7*x**4 - 3*x**3 + 2*x

def f4(x):
    return np.exp(x) + x**3

def f5(x):
    return x**2 * np.log(x)

# Compute derivatives at the specified points:
df1_at_1 = derivative(f1, 1)
df2_at_2 = derivative(f2, 2)
df3_at_1 = derivative(f3, 1)
df4_at_0 = derivative(f4, 0)
df5_at_1 = derivative(f5, 1)

print("Question 1: Differentiation (approximate using central differences)")
print("d/dx f1(x) at x=1 =", df1_at_1)
print("d/dx f2(x) at x=2 =", df2_at_2)
print("d/dx f3(x) at x=1 =", df3_at_1)
print("d/dx f4(x) at x=0 =", df4_at_0)
print("d/dx f5(x) at x=1 =", df5_at_1)
print("")

######################################################
# Question 2: First-Order Partial Derivatives (2 vars) #
######################################################
# We'll approximate partial derivatives using central differences.
def partial_derivative(f, x, y, var='x', h=1e-5):
    if var == 'x':
        return (f(x + h, y) - f(x - h, y)) / (2 * h)
    elif var == 'y':
        return (f(x, y + h) - f(x, y - h)) / (2 * h)
    else:
        raise ValueError("var must be 'x' or 'y'")

# Define the functions of two variables
def f1_xy(x, y):
    return x**2 + y**2 + 2*x*y

def f2_xy(x, y):
    return x**3 + 3*x*y + y**2

def f3_xy(x, y):
    return np.log(x) + x**2 * y

def f4_xy(x, y):
    return x**2 * y + y**3

# Compute the partial derivatives at the given points:
# For f1 at (1,1)
df1_dx_11 = partial_derivative(f1_xy, 1, 1, var='x')
df1_dy_11 = partial_derivative(f1_xy, 1, 1, var='y')

# For f2 at (1,2)
df2_dx_12 = partial_derivative(f2_xy, 1, 2, var='x')
df2_dy_12 = partial_derivative(f2_xy, 1, 2, var='y')

# For f3 at (2,3)
df3_dx_23 = partial_derivative(f3_xy, 2, 3, var='x')
df3_dy_23 = partial_derivative(f3_xy, 2, 3, var='y')

# For f4 at (1,2)
df4_dx_12 = partial_derivative(f4_xy, 1, 2, var='x')
df4_dy_12 = partial_derivative(f4_xy, 1, 2, var='y')

print("Question 2: First-Order Partial Derivatives (approximate)")
print("∂f1/∂x at (1,1) =", df1_dx_11)
print("∂f1/∂y at (1,1) =", df1_dy_11)
print("∂f2/∂x at (1,2) =", df2_dx_12)
print("∂f2/∂y at (1,2) =", df2_dy_12)
print("∂f3/∂x at (2,3) =", df3_dx_23)
print("∂f3/∂y at (2,3) =", df3_dy_23)
print("∂f4/∂x at (1,2) =", df4_dx_12)
print("∂f4/∂y at (1,2) =", df4_dy_12)
print("")

#############################
# Question 3: Vector Operations #
#############################
# Define the vectors:
v = np.array([2, -1, 3])
w = np.array([4, 0, -2])

# Vector addition:
v_plus_w = v + w

# Dot product:
dot_product = np.dot(v, w)

# Magnitudes (Euclidean norms):
v_norm = np.linalg.norm(v)
w_norm = np.linalg.norm(w)

print("Question 3: Vector Operations")
print("v + w =", v_plus_w)
print("v ⋅ w =", dot_product)
print("||v|| =", v_norm)
print("||w|| =", w_norm)
print("")

###############################
# Question 4: Matrix Multiplication #
###############################
# Define the matrices using the given numbers:
# A is given as [1 2 3 4] which we interpret as a 2x2 matrix:
A = np.array([[1, 2],
              [3, 4]])
# B is given as [2 0 1 -1] which we interpret as a 2x2 matrix:
B = np.array([[2, 0],
              [1, -1]])

# Compute A ⋅ B
A_dot_B = np.dot(A, B)

# Compute B ⋅ A
B_dot_A = np.dot(B, A)

# Compute the transpose of A ⋅ B
transpose_A_dot_B = A_dot_B.T

print("Question 4: Matrix Multiplication")
print("A ⋅ B =\n", A_dot_B)
print("B ⋅ A =\n", B_dot_A)
print("Transpose of A ⋅ B =\n", transpose_A_dot_B)
print("")

#########################################
# Question 5: Mean, Variance, Std Dev   #
#########################################
# Losses for 5 batches during training:
L = np.array([0.5, 0.7, 0.4, 0.6, 0.8])

# Compute the mean:
mean_L = np.mean(L)

# Compute the variance (population variance):
variance_L = np.var(L)

# Compute the standard deviation:
std_L = np.std(L)

print("Question 5: Statistics")
print("Mean of L =", mean_L)
print("Variance of L =", variance_L)
print("Standard Deviation of L =", std_L)
