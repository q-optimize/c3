"""This file represents a library of common hamiltonians"""


# function is not working will be removed most likely
# building a lambda function off other functions during execution possible???
def expand_hamiltonian(hlist, args):
    for i in range(len(hlist)):
        if i == 0:
            H = lambda t: hlist[i](t)
        else:
            H += hlist[i][1](t, args) * hlist[i][0](t)
    return H


def resonator(a, omega):
    return omega * a.dag() * a


def duffing(a, omega, delta):
    return omega * a.dag() * a + delta/2 * (a.dag() * a - 1) * a.dag() * a


def int_XX(a, b, g):
    return g * (a.dag() + a) * (b.dag() + b)


def int_jaynes_cummings(a, b, g):
    return g * (a.dag() * b + a * b.dag())

