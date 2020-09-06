import csv

def compute_error(x, y, weight, bias):
    data_len = len(y)
    error = 0
    for i in range(0, data_len):
        error += (y[i] - (weight* x[i] + bias)) ** 2
    return error/data_len

# updates weight and bias
def updateWB(x, y, weight, bias, learning_rate):
    data_len = len(y)
    weight_deriv = 0;
    bias_deriv = 0;

    for i in range(data_len):
        # partial derivative of the cost function with respect to weight
        weight_deriv += -2*x[i] * (y[i] - (weight*x[i] + bias))
        # partial derivative of the cost function with respect to bias
        bias_deriv += -2*(y[i] - (weight*x[i] + bias))

    # update the weight and the bias
    weight -= (weight_deriv/data_len)*learning_rate
    bias -= (bias_deriv/data_len)*learning_rate

    return weight, bias


def linear_regression(x, y, weight, bias, learning_rate, phases):
    phase = 0
    for i in range(phases):
        # update the weight and bias so that the cost function value decrease
        weight, bias = updateWB(x, y, weight, bias, learning_rate)
        cost = compute_error(x, y, weight, bias)

        # print after each phase
        if i % 100000 == 0:
            print("phase= {:d}, weight= {:.2f}, bias= {:.4f}, cost= {:.2}\n\n".format(phase, weight, bias, cost))
            phase += 1

    return weight, bias

# return a prediction based upon the current weight and the bias
def prediction(x, weight, bias):
    return x*weight + bias


def run():
    x = []
    y = []

    # initial values
    weight = 1
    bias = 0

    # read the csv file
    with open("simple_dataset.csv") as f:
        csv_reader = csv.reader(f, delimiter=',')
        first_line = True
        for row in csv_reader:
            if first_line:
                first_line = False
            else:
                y.append(float(row[1]))
                x.append(float(row[0]))

    weight, bias = linear_regression(x, y, weight, bias, 0.00000001, 1000000)
    print(prediction(14.031, weight, bias))
    weight, bias = linear_regression(x, y, weight, bias, 0.00000001, 1000000)
    print(prediction(14.031, weight, bias))
    

if __name__ == '__main__':
    run()