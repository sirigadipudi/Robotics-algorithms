# import re
# import subprocess
# import numpy as np
# import matplotlib.pyplot as plt


# def to_num(num_str):
#     extracted = num_str.rsplit('\\', 1)[0]
#     num = float(extracted)
#     return num


# seeds = [int(x) for x in np.rint(np.random.rand(5) * 100)]
# for bias in [.2, .05]:
#     costs = []

#     for seed in seeds:
#         # '--data-factor', f'{r}',
#         result = subprocess.run(
#             ['python', 'run.py', '-s', '2dof_robot_arm', '-p', 'rrt', '-o', '2', '--seed', f'{seed}', '-b', f'{bias}'],
#             capture_output=True, text=True)

#         result = result.stdout
#         cost_match = re.findall('cost.*', result, re.M)
#         cost = cost_match[1]
#         cost = float(cost.split()[1])
#         costs.append(cost)
#     costs = np.array(costs)
#     mean = np.mean(costs)
#     std = np.std(costs)
#     print(f'for a bias of {bias}, we have mean of {mean} and std of {std}')
#     # plt.plot(r_s, mean_pos_err, '-o', label='average position error')
#     # plt.plot(r_s, mean_anees, '-o', label='average ANEES')
#     # plt.xlabel('r')
#     # plt.ylabel('mean')
#     # plt.xscale('log')
#     # plt.yscale('log')
#     # plt.title(f'PF as in part c except with {i} particles')
#     # plt.legend()
#     # plt.savefig(f"3(d)with{i}particles.png")
#     # plt.cla(); plt.clf()
#     # mean_pos_err.clear()
#     # mean_anees.clear()

import numpy as np

cost_data = [416, 480, 549, 514, 520, 528, 480, 416]
time_data = [10.26293612, 15.02714491, 20.82529616, 7.538421154, 30.45296693, 14.33376598, 14.92605114, 10.73486114]

# Calculate mean
cost_mean = np.mean(cost_data)
time_mean = np.mean(time_data)

# Calculate standard deviation
cost_std = np.std(cost_data)
time_std = np.std(time_data)

# Print the results
print("Cost Mean:", cost_mean)
print("Cost Standard Deviation:", cost_std)
print("Time Mean:", time_mean)
print("Time Standard Deviation:", time_std)
