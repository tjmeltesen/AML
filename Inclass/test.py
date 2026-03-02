# Plot Miss Rate for Each Workload
import matplotlib.pyplot as plt

# Data
workloads = ["mat_mul", "daxpy", "queens"]
miss_rates = [2.8748, 4.5121, 2.1589]

# Create single plot (no subplots, no specific colors)
plt.figure()
plt.bar(workloads, miss_rates)

plt.xlabel("Workload")
plt.ylabel("Miss Rate (%)")
plt.title("L1D Cache Miss Rate by Workload (8KB, 8-way)")
plt.savefig("miss_rate_plot.png")  # Save the plot as an image file
plt.show()