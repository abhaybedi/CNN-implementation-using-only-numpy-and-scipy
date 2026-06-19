import numpy as np

class Sgd:
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        return weight_tensor - self.learning_rate * gradient_tensor

class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.velocity = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.velocity = (self.momentum_rate * self.velocity - self.learning_rate * gradient_tensor) # previous velocity * momentum + current negative gradient step

        return weight_tensor + self.velocity # update weights using velocity step


class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho

        self.v = 0  # first moment
        self.r = 0  # second moment
        self.k = 0  # number times updated
        self.eps = 1e-8

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.k += 1

        self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor # v = old gradient memory, mu = keep old part, (1-mu) = add current gradient
        self.r = self.rho * self.r + (1 - self.rho) * (gradient_tensor ** 2) # old squared-gradient memory + current gradient squared, weighted by rho

        v_hat = self.v / (1 - self.mu ** self.k) # corrected gradient memory, because v starts at 0
        r_hat = self.r / (1 - self.rho ** self.k)

        return weight_tensor - self.learning_rate * v_hat / (np.sqrt(r_hat) + self.eps) # momentum direction / adaptive gradient scale
