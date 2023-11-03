import timeit
import matplotlib.pyplot as plt
import MPN

dataset = [[2.7810836,2.550537003,0],
    [1.465489372,2.362125076,0],
    [3.396561688,4.400293529,0],
    [1.38807019,1.850220317,0],
    [3.06407232,3.005305973,0],
    [7.627531214,2.759262235,1],
    [5.332441248,2.088626775,1],
    [6.922596716,1.77106367,1],
    [8.675418651,-0.242068655,1],
    [7.673756466,3.508563011,1]]
l = 500
n_epoch = 40

time_mpn = timeit.timeit("MPN.test_mpn(dataset, l, n_epoch)", globals=locals(), number=1000)
parameters_mpn = len(MPN.weights_mpn)
error_x = [row[0] for row in MPN.error_data]
error_y = [row[1] for row in MPN.error_data]
dataset_x = [dataset.index(row) for row in dataset]
dataset_y = [row[-1] for row in dataset]
training_y = [row[-1] for row in MPN.training_data[-1]]

plt.style.use("seaborn-v0_8-bright")
fig, (ax1, ax2) = plt.subplots(nrows=2)
plt.tight_layout()

ax1.plot(error_x, error_y, marker=".")
ax1.set_yscale("log")
ax1.set_title("Error Rate over Epochs")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Error Rate")

ax2.plot(dataset_x, dataset_y, label="Dataset", color="#FF0000", marker="s")
ax2.plot(dataset_x, training_y, label="Training Data", color="#00FF00", marker="o")
ax2.set_title("Dataset vs Training Data")

print("MPN: parameters: %d, time: %f ms, ppms: %f" % (parameters_mpn, time_mpn, parameters_mpn/time_mpn))
plt.show()