# Predict with weights for MPN (McCulloch-Pitts Neuron)
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
def test_mpn(dataset, l, n_epoch):
    l_rate = 1 / l
    weights_mpn = train_weights_mpn(dataset, l_rate, n_epoch)
    return weights_mpn