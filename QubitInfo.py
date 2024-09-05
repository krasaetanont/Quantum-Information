# Copyright 2024 Wachirawit Krasaetanont, Muhammad Nafiz Nazhan
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import math as mt
import matplotlib.pyplot as plt
import random as rd
import multiprocessing

rho0 = np.array([0, 0])
rhoBase = np.array([1, 0])  # Base Rho


def mutual_info(P0, P1, rho0, rho1, ang):  # Function to find mutual information I
    x = np.array([])
    y = np.array([])
    z = np.array([0])
    maxima = 0  # Maxima of I
    derMaxima = 0  # Maxima of derivation of I
    epsilon = 1e-10  # Small value to avoid log of zero

    for i in range(101):
        a = i * mt.pi / 100 + ang

        # Born Rule P{y|x}
        p00 = (1 + np.dot(rho0, (mt.cos(a), mt.sin(a)))) / 2
        p01 = (1 + np.dot(rho0, (mt.cos(a + mt.pi), mt.sin(a + mt.pi)))) / 2
        p10 = (1 + np.dot(rho1, (mt.cos(a), mt.sin(a)))) / 2
        p11 = (1 + np.dot(rho1, (mt.cos(a + mt.pi), mt.sin(a + mt.pi)))) / 2

        # Ensure no zero or negative probabilities for logarithm
        p00 = max(p00, epsilon)
        p01 = max(p01, epsilon)
        p10 = max(p10, epsilon)
        p11 = max(p11, epsilon)

        try:
            I = (P0 * p00 * mt.log2(p00 / (P0 * p00 + P1 * p10)) +
                 P0 * p01 * mt.log2(p01 / (P0 * p01 + P1 * p11)) +
                 P1 * p10 * mt.log2(p10 / (P0 * p00 + P1 * p10)) +
                 P1 * p11 * mt.log2(p11 / (P0 * p01 + P1 * p11)))
        except Exception as e:
            print(f"Error at angle {a}: {e}")
            continue  # Skip appending to x and y if there is an error

        x = np.append(x, a / mt.pi)
        y = np.append(y, I)

    # Maxima detection logic with adjusted range check
    for j in range(1, len(y) - 1):
        z = np.append(z, (y[j] - y[j - 1]) / (x[j] - x[j - 1]))  # Deriving the mutual information function
        if j > 4:
            if z[j - 2] < z[j - 1] and z[j - 1] > z[j]:
                derMaxima += 1
        if y[j - 1] < y[j] and y[j] > y[j + 1]:
            maxima += 1
            peak = round(x[j], 3)
    z = np.append(z, [0])

    print(f"Maxima(s) of I: {maxima}, Maxima(s) of derivative of I: {derMaxima} | rho0 = {rho0}, rho1 = {rho1}")
    # plot(rho0, rho1, ang, P0, [maxima, x, y, derMaxima, z, peak])
    return [maxima, x, y, derMaxima, z, peak]


def findini(a, b):  # Function to find initial angle
    epsilon = 0.001
    min_diff = float('inf')
    best_ang = 0

    # Finding the best initial alignment
    for o in range(1001):
        ang = o * mt.pi / 1000
        dot_product_diff = abs(np.dot(np.array([mt.cos(ang), mt.sin(ang)]), (a - b)))

        # Check if the dot product difference is closest to zero
        if dot_product_diff < min_diff:
            min_diff = dot_product_diff
            best_ang = ang

        # Break early if best angle is found
        if dot_product_diff <= epsilon:
            break

    return best_ang

# plot graph
def plot(rho0, rho1, ang, P0, k):
    # Rounding the variables to 3 decimal places
    rho0 = np.array([round(rho0[0], 3), round(rho0[1], 3)])
    rho1 = np.array([round(rho1[0], 3), round(rho1[1], 3)])
    ang = round(ang, 3)
    P0 = round(P0, 3)

    # # Plotting the graphs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # # Graph 1: I and the derivative
    ax1.plot(k[1], k[2], label='Mutual Information I')
    ax1.plot(k[1], k[4], color="r", label='Derivative of I')
    ax1.set_title(f"P0 = {round(P0, 3)}, argmax = {k[5]}")
    # ax1.legend(loc='upper right')

    # # Graph 2: Sampling point of rho and initial angle
    circle = plt.Circle((0, 0), 1, color="r", fill=False, linestyle="--")
    ax2.add_patch(circle)
    ax2.plot(0, 0)
    ax2.plot([rho0[0], rho1[0]], [rho0[1], rho1[1]], color="blue")
    ax2.plot([0, mt.cos(ang)], [0, mt.sin(ang)])
    ax2.set_title(f"ρ0 = {rho0}, ρ1 = {rho1}")
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.show()


# Set a random two point
def random_point():
    count = 0
    while (count < 100000):
        count+= 1

        length0 = np.sqrt(np.random.uniform(0, 1))
        angle0 = np.pi * np.random.uniform(0, 2)
        length1 = np.sqrt(np.random.uniform(0, 1))
        angle1 = np.pi * np.random.uniform(0, 2)

        x0 = length0 * np.cos(angle0)
        y0 = length0 * np.sin(angle0)
        x1 = length1 * np.cos(angle1)
        y1 = length1 * np.sin(angle1)
        rho0 = np.array([x0, y0])
        rho1 = np.array([x1, y1])
        ang = findini(rho0, rho1)
        P0 = rd.random()
        P1 = 1 - P0

        k = mutual_info(P0, P1, rho0, rho1, ang)
        if k[0] != 1:
            print(k[0], "rho0 = ", rho0, "rho1 = ", rho1)
            plot(rho0, rho1, ang, P0, k)
            break
        # for finding multi maxima in the derivative curve     
        # if k[3] != 1:
        #     print("Maxima(s) of I: ", k[0], "rho0 = ", rho0, "rho1 = ", rho1, "derMaxima = ", k[3], P0)
        #     plot(rho0, rho1, ang, P0, k)
        #     break

random_point()
