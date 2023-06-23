import numpy as np
from RRTTree import RRTTree
import time

class RRTPlannerNonholonomic(object):
    def __init__(self, planning_env, bias=0.05, max_iter=10000, num_control_samples=25):
        self.env = planning_env                 # Car Environment
        self.tree = RRTTree(self.env)
        self.bias = bias                        # Goal Bias
        self.max_iter = max_iter                # Max Iterations
        self.num_control_samples = 25           # Number of controls to sample

    def Plan(self, start_config, goal_config):
        plan = []
        plan_time = time.time()
        x_new_id = self.tree.GetRootID()
        self.tree.AddVertex(start_config)
        # TODO: YOUR IMPLEMENTATION HERE
        for iter in range(self.max_iter):
            x_rand = self.sample(goal_config)
            x_near_id, x_near = self.tree.GetNearestVertex(x_rand)
            new_x, x_delta_t = self.extend(x_near, x_rand)
            if self.env.edge_validity_checker(new_x, x_near):
                # distance = self.env.compute_distance(x_near, new_x)
                x_new_id = self.tree.AddVertex(new_x, x_delta_t)
                self.tree.AddEdge(x_near_id, x_new_id)
                if self.env.goal_criterion(new_x, goal_config):
                    break

        # YOUR IMPLEMENTATION END HERE
        cost = 0
        while x_new_id != self.tree.GetRootID():
            cost+=self.tree.costs[x_new_id]
            plan.insert(1, self.tree.vertices[x_new_id])
            x_new_id = self.tree.edges[x_new_id]
        plan_time = time.time() - plan_time
        print("Cost: %f" % cost)
        if len(plan)>0:
            return np.concatenate(plan, axis=1)

    def extend(self, x_near, x_rand):
        """ Extend method for non-holonomic RRT

            Generate n control samples, with n = self.num_control_samples
            Simulate trajectories with these control samples
            Compute the closest closest trajectory and return the resulting state (and cost)
        """
        # Compute the distance between x_new and x_rand. If this distance is less than best_dist, update best_dist, best_x_new, and best_delta_t.
        # If best_dist is less than infinity, return best_x_new and best_delta_t. Otherwise, return None and None.

        # TODO: YOUR IMPLEMENTATION HERE
        best_state = None
        best_cost = float('inf')
        best_delta_t = None

        for _ in range(self.num_control_samples):
            linear_vel, steer_angle = self.env.sample_action()
            x_new, delta_t = self.env.simulate_car(x_near, x_rand, linear_vel, steer_angle)
            if x_new is None:
                continue
            # distance = self.env.compute_distance(x_new, x_rand)
            distance = self.distance_new(x_new, x_rand)
            if distance < best_cost:
                best_cost = distance
                best_state = x_new
                best_delta_t = delta_t

        return best_state, best_delta_t
    
                
    def sample(self, goal):
        # Sample random point from map
        if np.random.uniform() < self.bias:
            return goal

        return self.env.sample()
    
    def distance_new(self, state, x_rand):
        d = np.sqrt((state[0] - x_rand[0])**2 + (state[1] - x_rand[1])**2 + abs(state[2] - x_rand[2])*(180/np.pi))
        return d

    