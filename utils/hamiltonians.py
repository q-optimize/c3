"""This file represents a library of common hamiltonians"""

def resonator(a, omega):
    return omega * a.dag() * a


def duffing(a, omega, delta):
    return omega * a.dag() * a + delta/2 * (a.dag() * a - 1) * a.dag() * a


def int_XX(a, b, g):
    return g * (a.dag() + a) * (b.dag() + b)


def int_jaynes_cummings(a, b, g):
    return g * (a.dag() * b + a * b.dag())

