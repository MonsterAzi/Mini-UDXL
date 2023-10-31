def predict(row, weights):
    activation = weights[0]
    for i in range(len(row)-1)
        activation += weights[i + 1] * row[i]
    return activation