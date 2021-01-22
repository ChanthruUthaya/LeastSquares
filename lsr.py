from __future__ import print_function

import utilities as ut
import numpy as np
import sys
import matplotlib.pyplot as plt


def get_parameters_linear(Xs,Ys):
    X = np.column_stack((np.ones(np.size(Xs), dtype=int),Xs))
    X = np.matrix(X)
    Y = np.reshape(np.matrix(Ys), (len(Ys),1))
    return np.linalg.inv(X.transpose().dot(X)).dot(X.transpose().dot(Y))

def get_parameters_poly(Xs, Ys):
    X = np.column_stack((np.ones(np.size(Xs), dtype=int), Xs, np.square(Xs), np.power(Xs,3)))
    X = np.matrix(X)
    Y = np.reshape(np.matrix(Ys), (len(Ys),1))
    return np.linalg.inv(X.transpose().dot(X)).dot(X.transpose().dot(Y))

def get_params_sinusoidal(Xs, Ys):
     X = np.column_stack((np.ones(np.size(Xs), dtype=int), np.sin(Xs)))
     X = np.matrix(X)
     Y = np.reshape(np.matrix(Ys), (len(Ys),1))
     return np.linalg.inv(X.transpose().dot(X)).dot(X.transpose().dot(Y))

def linear(params, x):
    return params.item(1)*x + params.item(0)

def polynomial(params, x):
    return params.item(3)*(np.power(x,3)) + params.item(2)*(np.square(x)) + params.item(1)*x + params.item(0)

def sinfunc(params, x):
    return params.item(0) + params.item(1)*np.sin(x)

def myfuctions(params, x, index):
    if(index == 0):
        return linear(params, x)
    elif(index == 1):
        return polynomial(params, x)
    else:
        return sinfunc(params, x)

def parameters(Xs, Ys, choice):
    if(choice == 0):
        return get_parameters_linear(Xs, Ys)
    elif(choice == 1):
        return get_parameters_poly(Xs, Ys)
    else:
        return get_params_sinusoidal(Xs, Ys)

def summed_squared_error(Xs, Ys, params, func_call):
    if(func_call == 0):
        Y_vals_line = linear(params, Xs)
    elif(func_call == 1):
        Y_vals_line = polynomial(params, Xs)
    else:
        Y_vals_line = sinfunc(params, Xs)
    error_temp = np.square(Y_vals_line - Ys)
    return error_temp.sum()

def create_lines(Xs, Ys):
    y_vals = np.split(Ys,len(Ys)//20)
    x_vals = np.split(Xs,len(Xs)//20)
    segment_params = []
    for i in range(len(Xs)//20):
        errors = []
        for a in range(3):
            params = parameters(x_vals[i], y_vals[i], a)
            errors.append(summed_squared_error(x_vals[i], y_vals[i], params, a))
        index = errors.index(min(errors))
        if(index == 1 and (errors[1]/errors[0] > 0.775)):
            index = 0
        params = parameters(x_vals[i], y_vals[i], index)
        segment_params.append([params, index])
    return segment_params

def error_calc(Xs, Ys, params):
    y_vals = np.split(Ys,len(Ys)//20)
    x_vals = np.split(Xs,len(Xs)//20)
    error = 0
    for i in range(len(Xs)//20):
        error = error + summed_squared_error(x_vals[i], y_vals[i],params[i][0],params[i][1])
    return error

def plot_graph(Xs, Ys, params):
    y_vals = np.split(Ys,len(Ys)//20)
    x_vals = np.split(Xs,len(Xs)//20)
    len_data = len(Xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    plt.scatter(Xs, Ys, c=colour)
    error = 0
    for i in range(num_segments):
        plt.plot(x_vals[i], myfuctions(params[i][0], x_vals[i], params[i][1]), 'r')
        error = error + summed_squared_error(x_vals[i], y_vals[i],params[i][0],params[i][1])
    print(error)
    plt.show()

def main(args):
    x_values, y_values = ut.load_points_from_file(args[1])
    if(len(args) == 3 and args[2] == "--plot"):
        plot_graph(x_values, y_values, create_lines(x_values, y_values))

    else:
        print(error_calc(x_values, y_values,create_lines(x_values, y_values)))

if __name__ == "__main__":
    main(sys.argv)
