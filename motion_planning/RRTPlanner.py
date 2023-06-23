from __future__ import absolute_import, print_function
import numpy as np
from RRTTree import RRTTree
import sys
import time

class RRTPlanner(object):

    def __init__(self, planning_env, bias = 0.05, eta = 1.0, max_iter = 10000):
        self.env = planning_env         # Map Environment
        self.tree = RRTTree(self.env)
        self.bias = bias                # Goal Bias
        self.max_iter = max_iter        # Max Iterations
        self.eta = eta                  # Distance to extend

    def Plan(self, start_config, goal_config):
        self.tree.AddVertex(start_config)
        path = []
        for iter in range(self.max_iter):
            x_rand = self.sample(goal_config)
            x_near_id, x_near = self.tree.GetNearestVertex(x_rand)
            t_f, new_x = self.extend(x_near, x_rand)
            if t_f:
                distance = self.env.compute_distance(x_near, new_x)
                new_id = self.tree.AddVertex(new_x, distance)
                self.tree.AddEdge(x_near_id, len(self.tree.vertices) - 1)
                if self.env.goal_criterion(new_x):
                    root_id = self.tree.GetRootID() 
                    current_id = new_id
                    while current_id != root_id:
                        path.append(self.tree.vertices[current_id].flatten())
                        current_id = self.tree.edges[current_id]
                    path.append(self.tree.vertices[root_id].flatten())
                    path.reverse()
                    break
        return np.array(path)        

    def extend(self, x_near, x_rand):
        # TODO: YOUR IMPLEMENTATION HERE
        new_x = (np.array(x_rand - x_near)*self.eta + x_near).astype(int)
        if self.env.edge_validity_checker(x_near, new_x):
            return True, new_x
        else:
            return False, new_x

    def sample(self, goal):
        # Sample random point from map
        if np.random.uniform() < self.bias:
            return goal

        return self.env.sample()