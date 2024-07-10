import numpy as np
import math
import random
import matplotlib.pyplot as plt
import time


J = 1
kb = 1
L = 100
N = L**2


def init_state(initial_state):
    lattice = []
    if initial_state == 1 or initial_state == -1:
        for i in range(L):
            line = []
            for j in range(L):
                line.append(initial_state)
            lattice.append(line)
        return lattice
    if initial_state == 0:
        for i in range(L):
            line = []
            for j in range(L):
                if random.random() <= 0.5:
                    line.append(1)
                else:
                    line.append(-1)
            lattice.append(line)
        return lattice
    else:
        raise ValueError("Invalid initial_state")


def hamiltonian(lattice):  # count hamiltonian of state A
    sum = 0
    for i in range(L):
        for j in range(L):
            sum += lattice[i][j] * (lattice[i - 1][j] + lattice[i][j - 1])
    return -1 * J * sum

def magnetization(lattice):
    return float(np.sum(lattice)) / N  # return magnetization


def energy(lattice):
    return hamiltonian(lattice) / N  # return energy
    

def simulate_states(init_st, t, a, b, step):
    states_logs = []
    A = init_state(init_st)
    for i in range(b * step + 1):
        if i % step == 0 and i >= a * step and i <= b * step:
            state = []
            for i in range(L):
                line = []
                for j in range(L):
                    line.append(A[i][j])
                state.append(line)
            states_logs.append(state)
        i = random.randint(0, L - 1)
        j = random.randint(0, L - 1)
        neighbors = A[(i+1)%L][j] + A[i][(j+1)%L] + A[(i-1)%L][j] + A[i][(j-1)%L]
        delta_E = 2 * J * A[i][j] * neighbors
        if delta_E < 0 or random.random() < math.exp(-delta_E / (kB * t)):
            A[i][j] *= -1
            
    return states_logs


def simulation(init_st, t):  # return magnetization and energy of every N-th step
    magnetization_logs = []
    energy_logs = []
    A = init_state(init_st)
    
    for k in range(60000000):
        i = random.randint(0, L - 1)
        j = random.randint(0, L - 1)
        neighbors = A[(i+1)%L][j] + A[i][(j+1)%L] + A[(i-1)%L][j] + A[i][(j-1)%L]
        delta_E = 2 * J * A[i][j] * neighbors
        if delta_E < 0 or random.random() < math.exp(-delta_E / (kB * t)):
            A[i][j] *= -1
        if k % N == 0:
            magnetization_logs.append(magnetization(A))
            energy_logs.append(energy(A))
            
    return magnetization_logs, energy_logs


def note():  # draws graphics of magnetization and energy
    m1_1, e1_1 = simulation(1, 2)
    m1_2, e1_2 = simulation(0, 2)
    m1_2_3, e1_2_3 = simulation(0, 2)
    m1_3, e1_3 = simulation(-1, 2)
    m2_1, e2_1 = simulation(1, 2.5)
    m2_2, e2_2 = simulation(0, 2.5)
    m2_3, e2_3 = simulation(-1, 2.5)
    
    figure, axis = plt.subplots(2, 2, figsize = (15, 15))
    axis[0, 0].plot(m1_1)
    axis[0, 0].plot(m1_2)
    axis[0, 0].plot(m1_2_3)
    axis[0, 0].plot(m1_3)
    axis[0, 0].legend(
        [
            "initial state = 1",
            "initial state = 1 with probability 1/2",
            "initial state = 1 with probability 1/2",
            "initial state = -1",
        ]
    )
    axis[0, 0].set_title("magnetization, T = 2")
    axis[0, 0].set_xlabel('Time')
    axis[0, 0].set_ylabel('Magnetization')
    
    axis[0, 1].plot(m2_1)
    axis[0, 1].plot(m2_2)
    axis[0, 1].plot(m2_3)
    axis[0, 1].legend(
        [
            "initial state = 1",
            "initial state = 1 with probability 1/2",
            "initial state = -1",
        ]
    )
    axis[0, 1].set_title("magnetization, T = 2.5")
    axis[0, 1].set_xlabel('Time')
    axis[0, 1].set_ylabel('Magnetization')

    axis[1, 0].plot(e1_1)
    axis[1, 0].plot(e1_2)
    axis[1, 0].plot(e1_3)
    axis[1, 0].legend(
        [
            "initial state = 1",
            "initial state = 1 with probability 1/2",
            "initial state = -1",
        ]
    )
    axis[1, 0].set_title("energy, T = 2")
    axis[1, 0].set_xlabel('Time')
    axis[1, 0].set_ylabel('Energy')

    axis[1, 1].plot(e2_1)
    axis[1, 1].plot(e2_2)
    axis[1, 1].plot(e2_3)
    axis[1, 1].legend(
        [
            "initial state = 1",
            "initial state = 1 with probability 1/2",
            "initial state = -1",
        ]
    )
    axis[1, 1].set_title("energy, T = 2.5")
    axis[1, 1].set_xlabel('Time')
    axis[1, 1].set_ylabel('Energy')
    plt.show()


def note1():  # draws mean magnetization, mean energy, magnetic susceptibility and specific heat as functions of time
    t = []
    for i in range(1, 41, 2):
        t.append(i * 0.1)
    mean_magnetization = []
    mean_energy = []
    magnetic_susceptibility = []
    specific_heat = []
    for tt in t:
        m, e = simulation(0, tt)

        avg_m = np.mean(np.abs(m))
        avg_e = np.mean(e)
        cv = (np.var(e) * (L * L)) / (kB * tt * tt)
        chi = (np.var(m) * (L * L)) / (kB * tt)
        
        mean_magnetization.append(avg_m)
        mean_energy.append(avg_e)
        magnetic_susceptibility.append(chi)
        specific_heat.append(cv)
    figure, axis = plt.subplots(2, 2, figsize = (15, 15))
    axis[0, 0].plot(t, mean_magnetization, "o")
    axis[0, 0].set_title("mean magnetization, T [0, 4]")
    axis[0, 0].set_xlabel('Temperature')
    axis[0, 0].set_ylabel('Magnetization')
    
    axis[0, 1].plot(t, mean_energy, "o")
    axis[0, 1].set_title("mean energy, T [0, 4]")
    axis[0, 1].set_xlabel('Temperature')
    axis[0, 1].set_ylabel('Energy')
    
    axis[1, 0].plot(t, magnetic_susceptibility, "o")
    axis[1, 0].set_title("magnetic susceptibility, T [0, 4]")
    axis[1, 0].set_xlabel('Temperature')
    axis[1, 0].set_ylabel('X')
    
    axis[1, 1].plot(t, specific_heat, "o")
    axis[1, 1].set_title("specific heat, T [0, 4]")
    axis[1, 1].set_xlabel('Temperature')
    axis[1, 1].set_ylabel('C')
    
    plt.show()


def note2(t):  # draws microscopic configurations
    s1 = simulate_states(0, t, 1, 9, 20000)
    figure, axis = plt.subplots(3, 3, figsize = (10, 10))
    for i in range(3):
        for j in range(3):
            axis[i, j].pcolor(s1[3 * i + j])
    plt.show()
