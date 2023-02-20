import numpy as np


def loss_energy_mae(y_pred, y_true, batch_size=32):
    n = len(y_pred)
    loss = 0.0

    for i in range(n):
        loss += (abs(y_true[i, 0] - y_pred[i, 0]) + abs(y_true[i, 1] - y_pred[i, 1])) / 2
    return loss / n


def loss_energy_mse_relative(y_pred, y_true, batch_size=32):
    n = len(y_pred)
    loss = 0.0

    for i in range(n):
        loss += ((y_true[i, 0] - y_pred[i, 0]) ** 2 / y_true[i, 0] +
                 (y_true[i, 1] - y_pred[i, 1]) ** 2 / y_true[i, 1]) / 2
    return loss / n


def loss_energy_mae_asym(y_pred, y_true, batch_size=32):
    n = len(y_pred)
    a = 10
    loss = 0.0

    for i in range(n):
        loss1 = (y_true[i, 0] - y_pred[i, 0]) / y_true[i, 0]
        loss2 = (y_true[i, 1] - y_pred[i, 1]) / y_true[i, 1]

        if loss1 < 0.0:
            loss1 *= a
        if loss2 < 0.0:
            loss2 *= a

        loss += (abs(loss1) + abs(loss2)) / 2

    return loss / n
