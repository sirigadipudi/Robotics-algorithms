""" Written by Brian Hou for CSE571: Probabilistic Robotics
"""

import numpy as np

from utils import minimized_angle


class ExtendedKalmanFilter:
    def __init__(self, mean, cov, alphas, beta):
        self.alphas = alphas
        self.beta = beta

        self._init_mean = mean
        self._init_cov = cov
        self.reset()

    def reset(self):
        self.mu = self._init_mean
        self.sigma = self._init_cov

    def update(self, env, u, z, marker_id):
        """Update the state estimate after taking an action and receiving a landmark
        observation.

        u: action
        z: landmark observation
        marker_id: landmark ID
        """
        # YOUR IMPLEMENTATION HERE
    
        #Linearize dynamics and Prediction steps
        M = env.noise_from_motion(u, self.alphas)
        V = env.V(self.mu, u)
        mu_pred, G = env.G(self.mu, u)
        sigma_pred = (G @ self.sigma @ G.T) + (V @ M @ V.T)

        #Linear measurement
        H = env.H(mu_pred, marker_id)
        zhat = env.observe(mu_pred, marker_id)
        z = minimized_angle(z)
       
        #Measurement update steps
        k1 = sigma_pred @ H.T
        k2 = np.linalg.inv(H @ sigma_pred @ H.T + self.beta)
        K = k1*k2
        self.mu = mu_pred + K@ (minimized_angle(z - zhat))
        self.sigma = (np.eye(3) - K@H) @ sigma_pred

        return self.mu, self.sigma

