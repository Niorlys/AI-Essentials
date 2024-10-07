import sympy
import numpy as np
import matplotlib.pyplot as plt
from _0_base import MLBase

class SymbGradientDescendent:
    def __init__(self, function, variables):
        """
        Initialize the GradientDescendent class.

        Parameters:
        - function: sympy expression representing the function to minimize.
        - variables: list of sympy.Symbol objects representing the variables.
        """
        self.function = function
        self.variables = variables
        self.function_numeric = sympy.lambdify(variables, function, 'numpy')

        # Compute the symbolic gradient (partial derivatives for each variable)
        self.gradient = [sympy.diff(function, var) for var in variables]

    def run(self, learning_rate, initial_params, max_iter, plot_history=False):
        """
        Run the gradient descent algorithm according to the following procedure:
        1. Initialize parameters θ with some values (often set to 0 or random small values)
        2. Repeat until convergence or max_iter:
            a. Compute the gradient of the cost function with respect to θ:
               ∇J(θ) = [∂J/∂θ_0, ∂J/∂θ_1, ..., ∂J/∂θ_n]
            
            b. Update each parameter θ using the gradient and the learning rate:
               θ = θ - η * ∇J(θ)    
            
        Parameters:
        - learning_rate: float, the step size for each iteration.
        - initial_params: list or array-like, the initial values for each variable.
        - max_iter: int, the maximum number of iterations.

        Returns:
        - history: list of lists, the history of parameter values at each iteration.
        """

        grad_numeric = sympy.lambdify(self.variables, self.gradient, 'numpy')

        # Initializing parameters
        params = np.array(initial_params, dtype=float)
        
        history = [params.copy()]

        # Perform gradient descent
        for _ in range(max_iter):
            # Compute the gradient at the current parameters
            grad_values = np.array(grad_numeric(*params))

            # Update parameters using gradient descent
            params -= learning_rate * grad_values

            
            history.append(params.copy())

        if plot_history:
            self._plot_history(history)

        return np.array(history)

    def _plot_history(self, history):
        """
        Plot the history of the parameter values for 2D functions.

        Parameters:
        - history: list of lists, the history of parameter values at each iteration.
        """
        if len(self.variables) != 2:
            raise ValueError("Plotting is only supported for 2D functions.")

        x_vals = [params[0] for params in history]
        y_vals = [params[1] for params in history]

        x = np.linspace(-10, 10, 400)
        y = np.linspace(-10, 10, 400)
        X, Y = np.meshgrid(x, y)
        Z = sympy.lambdify(self.variables, self.function, 'numpy')(X, Y)

        plt.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.6)
        plt.plot(x_vals, y_vals, 'ro-', markersize=5, label='Gradient Descent Path')
        plt.xlabel(f'{self.variables[0]}')
        plt.ylabel(f'{self.variables[1]}')
        plt.title('Gradient Descent Path')
        plt.legend()
        plt.show()

    @staticmethod
    def non_symb_gradient_descendent(gradient, learning_rate, initial_params, max_iter):
        """
        Apply gradient descendet using an ussual callable as gradient
        """
        params = np.array(initial_params, dtype=float)

        for _ in range(max_iter):
            grad_values = np.array([pd(*params) for pd in gradient])

            params -= learning_rate * grad_values

        return params

def linear_regression_gradient_descent(learning_rate=0.02, max_iter=1000):
    # Generating synthetic data
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10
    y = 2 * X + 1 + np.random.randn(100, 1)*0.7 
    m = len(X)

    gradient = [lambda theta_0, theta_1: (1/m)*np.sum((theta_0*X + theta_1 - y)*X),
                lambda theta_0, theta_1: (1/m)*np.sum((theta_0*X + theta_1 - y))]
    
    initial_params = [0, 0]
    opt = SymbGradientDescendent.non_symb_gradient_descendent(gradient, learning_rate, initial_params, max_iter)
    print("theta_0, theta_1:", *opt)

    plt.scatter(X, y)
    plt.title("Synthetic Data and Regression Line")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(X, opt[0]*X + opt[1], 'r-', label='Regression Line')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    x, y = sympy.symbols('x y')
    f = (x-2)**2 + y**2

    gd = SymbGradientDescendent(f, [x, y])

    learning_rate = 0.1
    initial_params = [-8.0, 8.0]
    max_iter = 100
    opt = gd.run(learning_rate, initial_params, max_iter, plot_history=True)
    print("Optimized parameters:", opt[-1])
