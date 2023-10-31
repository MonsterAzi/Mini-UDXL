import timeit
import matplotlib.pyplot as plt

# Predict with weights for MPN
def predict_mpn(row, weights_mpn):
    activation = weights_mpn[0]
    for i in range(len(row)-1):
        activation += weights_mpn[i + 1] * row[i]
    return activation

# Estimate MPN weights using stochastic gradient descent
def train_weights_mpn(train, l_rate, n_epoch):
    global weights_mpn
    global training_data
    global error_data
    weights_mpn = [0.0 for i in range(len(train[0]))]
    error_data = []
    training_data = []
    for epoch in range(n_epoch):
        sum_error = 0.0
        training_data_subset = []
        for row in train:
            prediction = predict_mpn(row, weights_mpn)
            training_data_subset.append([row[0],row[1],prediction])
            error = row[-1] - prediction
            sum_error += error**2
            weights_mpn[0] = weights_mpn[0] + l_rate * error
            for i in range(len(row)-1):
                weights_mpn[i + 1] = weights_mpn[i + 1] + l_rate * error * row[i]
        training_data.append(training_data_subset)
        error_data.append([epoch, sum_error])
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

# test MPN
def test_mpn():
    global dataset
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
    l_rate = 1 / l
    n_epoch = 40
    weights_mpn = train_weights_mpn(dataset, l_rate, n_epoch)
    return weights_mpn

time = timeit.timeit("test_mpn()", globals=locals(), number=1)*1000
parameters = len(weights_mpn)
error_x = [row[0] for row in error_data]
error_y = [row[1] for row in error_data]
dataset_x = [dataset.index(row) for row in dataset]
dataset_y = [row[-1] for row in dataset]
training_y = [row[-1] for row in training_data[-1]]

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

print("Parameters: %d, Time: %f ms, Ppms: %f" % (parameters, time, parameters/time))
plt.show()