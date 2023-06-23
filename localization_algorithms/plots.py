# import subprocess


# # define lists of values for the --data-factor and --filter-factor arguments
# data_factors = [0.0156, 0.0625, 0.25, 4, 16, 64]
# #filter_factors = [0.0156, 0.0625, 0.25, 4, 16, 64]
# #seeds = [1,2,3,4]
# sizes = [20, 50, 500]

# # iterate over the values and call the command for each combination of factors
# # for size in sizes:
# for data_factor in data_factors:
#     for seed in range(4):
#         subprocess.run(['python', 'localization.py', 'pf', '--num-particles', str(50), '--seed', str(seed), '--filter-factor', str(data_factor)])

# # import matplotlib.pyplot as plt

# # Data
# r = [1/64, 1/16, 1/4, 4, 16, 64]
# mean_errors = [0.82, 1.546, 2.99, 11.73, 30.73, 123.2]
# anees_errors = [0.817, 0.7987, 0.81, 0.84, 2.215, 1.391]

# # Plot mean errors
# plt.plot(r, mean_errors)
# plt.xscale("log")
# plt.xlabel("r")
# plt.ylabel("Mean errors")
# plt.show()

# # Plot ANEES errors
# plt.plot(r, anees_errors)
# plt.xscale("log")
# plt.xlabel("r")
# plt.ylabel("ANEES errors")
# plt.show()

import matplotlib.pyplot as plt

# Data
r = [1/64, 1/16, 1/4, 4, 16, 64]
mean_errors = [74, 43.4, 24.04, 12.61, 12.84, 27.86]
anees_errors = [500, 400, 300, 0.4, 0.6, 0.69]

# Plot data
plt.plot(r, mean_errors, label="Mean errors (50 particles)")
plt.plot(r, anees_errors, label="ANEES errors (50 particles)")
plt.xscale("log")
plt.xlabel("r")
plt.ylabel("Error")
plt.ylim(0, 100)
plt.legend()
plt.show()
