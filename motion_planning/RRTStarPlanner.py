from __future__ import absolute_import, print_function
import numpy as np
from RRTTree import RRTTree
import sys
import time
from tqdm import tqdm

class RRTStarPlanner(object):

    def __init__(self, planning_env, bias = 0.05, eta = 1.0, max_iter = 10000):
        self.env = planning_env         # Map Environment
        self.tree = RRTTree(self.env)
        self.bias = bias                # Goal Bias
        self.max_iter = max_iter        # Max Iterations
        self.eta = eta                  # Distance to extend

    def compute_cost(self, node_id):
        root_id = self.tree.GetRootID()

        cost = 0
        node = self.tree.vertices[node_id]
        while node_id != root_id:
            parent_id = self.tree.edges[node_id]
            parent = self.tree.vertices[parent_id]
            cost += self.env.compute_distance(node, parent)

            node_id = parent_id
            node = parent

        return cost

    def Plan(self, start_config, goal_config, rad=10):
        # TODO: YOUR IMPLEMENTATION HERE
        plan = []
        # x_new_id = self.tree.GetRootID()
        self.tree.AddVertex(start_config)

        for _ in range(self.max_iter):
            x_rand = self.sample(goal_config)
            x_near_id, x_near = self.tree.GetNearestVertex(x_rand)
            x_new = self.extend(x_near, x_rand)
            if self.env.edge_validity_checker(x_near, x_new):
                cost = self.env.compute_distance(x_new, x_near)  
                near_ids, near_vertices = self.tree.GetNNInRad(x_new, rad)
                x_new_id = self.tree.AddVertex(x_new, cost)
                # self.tree.AddEdge(x_near_id, x_new_id)
                c_min = self.compute_cost(x_near_id)  + cost
                x_min_id = x_near_id
                for i, near_id in enumerate(near_ids):
                    near_vertex = near_vertices[i]
                    if (self.compute_cost(near_id) + self.env.compute_distance(x_new, near_vertex) < c_min):
                        if self.env.edge_validity_checker(x_new, near_vertex):
                            x_min_id = near_id
                            c_min = self.compute_cost(near_id) + self.env.compute_distance(x_new, near_vertex)
                self.tree.AddEdge(x_min_id, x_new_id)

                for i, near_id in enumerate(near_ids):
                    near_vertex = near_vertices[i]
                    if (self.compute_cost(x_new_id) + self.env.compute_distance(x_new, near_vertex) < self.compute_cost(near_id)):
                        if self.env.edge_validity_checker(x_new, near_vertex):
                            # parent_id = self.tree.edges[near_id]
                            self.tree.AddEdge(x_new_id, near_id)
                
                if self.env.goal_criterion(x_new):
                    root_id = self.tree.GetRootID() 
                    current_id = x_new_id
                    while current_id != root_id:
                        plan.append(self.tree.vertices[current_id].flatten())
                        current_id = self.tree.edges[current_id]
                    plan.append(self.tree.vertices[root_id].flatten())
                    plan.reverse()
                    break
        return np.array(plan)

    def extend(self, x_near, x_rand):
        # TODO: YOUR IMPLEMENTATION HERE
        direction = np.array(x_rand - x_near)
        # distance = np.linalg.norm(direction)
        # direction = direction / distance
        x_new = (x_near + self.eta * direction).astype(int) 
        return x_new

    def sample(self, goal):
        # Sample random point from map
        if np.random.uniform() < self.bias:
            return goal

        return self.env.sample()