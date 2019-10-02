#!/usr/bin/python

import pickle
import sys
import os
import numpy as np
import matplotlib.pyplot as plt


"""
Log format is always:
    'open_loop', 'closed_loop', 'learn_model'
Each of them have a list with elements:
    [parameter_vector, goal_value]
"""

lims = [1e-14, 2]

save_figs = False
filename = sys.argv[1]
if len(sys.argv) > 2:
    save_figs = sys.argv[2] == '-save'

datafile = open(filename, 'rb')
logs = pickle.load(datafile)
datafile.close()

if save_figs:

    foldername = filename.replace('.pickle', '')
    if not os.path.exists(foldername):
        os.mkdir(foldername)
    os.chdir(foldername)

fid = []
for m in logs['open_loop']:
    fid.append(m[1])

plt.figure()
plt.semilogy(fid)
plt.title('C1: Open loop simulation (wrong model)')
plt.xlabel('Evaluations')
plt.ylabel('1-Fidelity')
plt.ylim(lims)
plt.grid()
plt.show(block=False)
if save_figs:
    plt.savefig(filename+'_openloop.png')

fid = []
for m in logs['closed_loop']:
    fid.append(m[1])

plt.figure()
plt.semilogy(fid)
plt.title('C2: Closed loop simulation (real model)')
plt.xlabel('Evaluations')
plt.ylabel('1-Fidelity')
plt.ylim(lims)
plt.grid()
plt.show(block=True)
if save_figs:
    plt.savefig(filename+'_closedloop.png')

freq = []
alpha = []
real_freq = 5.05e9*2*np.pi
real_alpha = 0.72*2e9*np.pi
pp_error = []

# for m in logs['learn_model']:
#     freq.append(np.abs(m[0][0]-real_freq)/real_freq)
#     alpha.append(np.abs(m[0][1]-real_alpha)/real_alpha)
#     pp_error.append(m[1])
#
# fig=plt.figure()
# ax=fig.add_subplot(111)
# ax.semilogy(freq)
# ax.semilogy(alpha)
# plt.ylim(lims)
# plt.title('C3: Characterization (fix wrong model)')
# plt.xlabel('Evaluations')
# plt.ylabel('Relative parameter error')
# plt.legend(['qubit frequency', 'response'])
# plt.grid()
# plt.show(block=False)
# if save_figs:
#     plt.savefig(filename+'_learning.png')
#
# fig=plt.figure()
# ax=fig.add_subplot(111)
# ax.semilogy(pp_error)
# plt.ylim(lims)
# plt.title('C3: Characterization (fix wrong model)')
# plt.xlabel('Evaluations')
# plt.ylabel('per sample error')
# plt.grid()
# plt.show(block=False)
# if save_figs:
#     plt.savefig(filename+'_learning_convergence.png')
