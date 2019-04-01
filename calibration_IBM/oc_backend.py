import numpy as np
from qutip import *
import numpy.random as rand
import json
import copy
import matplotlib.pyplot as plt
import cma.evolution_strategy as cmaes
import os
from IPython import display

Gates = ['Y90p','X90p']

#import yaml




#import warnings
#warnings.filterwarnings('ignore')

# Define plot characteristics
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 36
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.linewidth'] = 2.0

#BASE_DIR = '/home/measure/measurements/2019-01-11-XLD1-F-chip8/'
#SQORE_DIR = '/home/measure/qc_code/ibmqc_package/share/sqore/optimal_control/'
#RUN_SQORE = 'run_sqore_precalh'

def plot_scatter_data(x, y, ax, fig, color='blue'):
    """
    Live update of a scatter plot during optimization.
    """
    if ax.lines:
        ax.lines[0].set_xdata(x)
        ax.lines[0].set_ydata(y)
    else:
        ax.scatter(x, y, color=color)
    fig.canvas.draw()


def prepare_plot():
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('1-F')
    ax.tick_params(direction='in', length=6, width=2)

    return ax, fig


def update_plot(ebr, results, ax, fig):
    plot_scatter_data([ebr.iter_num] * len(results), results, ax, fig)
    plot_scatter_data([ebr.iter_num], np.average(results), ax, fig, color='red')
    plt.tight_layout()
    display.clear_output(wait=True)
    display.display(plt.gcf())
    # fig.savefig(ebr.base_dir + ebr.dir_name + '/population.png', dpi=300, bbox_inches='tight')
    # fig.savefig(ebr.base_dir + ebr.dir_name + '/population.pdf', bbox_inches='tight')



def cmaes_par_search(initial_values, error_bars, br, pop_size=30, rb_len=10,
                     rep=10, change_RB_seq=False, qubits=1):
    """
    Function to run a cma evolution strategy optimization
    This function calls evaluate_sequences() with a set of solutions in the parameter space

    :param initial_values: dictionary with pulse parameters and initial values
    :param error_bars: dictionairy with pulse parameters and values for errors
    :param pop_size: population size of the CMA-ES optimizer
    :param rb_len: length of the RB sequences
    :param rep: repetition of RB with different sequences
    """

    # transform the given parameters with error bars to normalize
    par_values_norm = normalize(initial_values, initial_values)
    error_bars_norm = normalize(error_bars, initial_values)

    # This is our unwrapping function
    par_vec = br.unwrap_parameters(par_values_norm)
    sigma0s = br.unwrap_parameters(error_bars_norm)

    # we can only pass one sigma and then multipliers for other parameters
    CMA_stds = [x / sigma0s[0] for x in sigma0s]
    es = cmaes.CMAEvolutionStrategy(par_vec,  # initial values
                                    sigma0s[0],  # initial std
                                    {'CMA_stds': CMA_stds,  # multipliers of sigma0
                                     'popsize': pop_size,
                                     'tolfun': 1e-8}
                                    )

    # Setup plot for live-plotting.
    ax, fig = prepare_plot()

    # Main part of algorithm, like doing fminsearch.
    while not es.stop():
        solutions = es.ask()  # list of new solutions
        # TODO batch is the whole population.
        value_batch = []
        # transform solutions back to real values dict
        par_values = []
        for i in range(len(solutions)):
            par_values = inverse_norm(br.wrap_parameters(solutions[i]), initial_values)
            value_batch.append(par_values)

        # generate appropriate set of sequences
        # TODO make sure this function is the same for exp and sim
        sequences = br.single_length_RB(rep, rb_len)
        # Query backend for new infidelities
        results = br.evaluate_sequences(sequences, value_batch, rounds=300)
        print(results)
        update_plot(br, results, ax, fig)

        # TODO adapt RB sequence length to gain in sensitivity.
        # rb_len = br. ....

        # tell the cmaes object the performance of each solution and update
        #print(results)
        es.tell(solutions, results)
        # show current evaluation status
        #print(br.best_ever)

        es.result_pretty()

    # show final results in a nice format
    res = es.result
    # result contains
    # 0 xbest best solution evaluated
    # 1 fbest objective function value of best solution
    # 2 evals_best evaluation count when xbest was evaluated
    # 3 evaluations evaluations overall done
    # 4 iterations
    # 5 xfavorite distribution mean in "phenotype" space, to be considered as current best estimate of the optimum
    # 6 stds effective standard deviations, can be used to compute a lower bound on the expected coordinate-wise distance to the true optimum

    # print again
    print('best fidelity', res[1])

    # return the values of the parameters

    # cal_dir = get_cal_dir_name(BASE_DIR)

### Normalization function and inverse
def normalize(par_values, initial_values):
    """normalize values for optimization
    param: par_values: dictionairy with parameters and corresponding values
    param:initial_values: dictionairy of initial pulse parameters for normalization
    :returns: normalized dictionairy of pulse parameters"""
    norm_par_values = copy.deepcopy(initial_values)

    for ch in par_values:
        for pulse in par_values[ch]:
            for param in par_values[ch][pulse]:
                if param is not 'mapto':
                    norm_par_values[ch][pulse][param] = par_values[ch][pulse][param] / initial_values[ch][pulse][param]
    return norm_par_values


def inverse_norm(norm_par_values, initial_values):
    """revert normalization for use in experiment
    :param par_values_norm: normalized values (for optimizer)
    :param initial_values: initial values for rescaling
    return: rescaled values
    """
    par_values = copy.deepcopy(initial_values)

    for ch in norm_par_values:
        for pulse in norm_par_values[ch]:
            for param in norm_par_values[ch][pulse]:
                if param is not 'mapto':
                    par_values[ch][pulse][param] = norm_par_values[ch][pulse][param] * initial_values[ch][pulse][param]
    return par_values

def rect_space(Dims,dims):
    I=np.eye(np.prod(Dims))
    rect=[]
    for state in state_number_enumerate(dims):
        i=state_number_index(Dims, state)
        rect.append(I[:,i])
    rect=np.array(rect).T
    return Qobj(rect,dims=[Dims,dims])

def evolution(H,U0,ts,history=False):
    dt = ts[1]
    if history:
        U = [U0]
        for t in ts[1::]:
            du = (-1j * dt * H(t+dt/2)).expm()
            U.append(du * U[-1])
    else:
        U = [U0]
        for t in ts[1::]:
            du = (-1j * dt * H(t+dt/2)).expm()
            U[0]= du * U[0]
    return U0.trans()*U  # if the unitary to be propagated is truncated to be rectangular(i.e. some colum vectors that are interesting)

# Duffing oscillator is the unharmonic oscillator
def duffing(w, d, a):
    return w * a.dag() * a + d/2*(a.dag() * a - 1)*a.dag() * a

# Hamiltonian including resonator
def jc_hamiltonian(omega_r, a, omega_q, delta, b, g):
    return omega_r * a.dag() * a + duffing(omega_q, delta, b) \
        + g * (a.dag()+a) * (b.dag() + b)

# produces a given state....
def state(q, N_q, r, N_r):
    qudit_states = {'g': 0,
                    'e': 1,
                    'f': 2,
                    'h': 3,
                    'k': 4,
                    'l': 5
                   }
    return tensor(projection(N_r, r, r), projection(N_q, qudit_states[q], qudit_states[q]))

# 2 qubit hamiltonian
def jc_hamiltonian_2(omega_r1, a1, omega_q1, delta1, b1, g1,\
                     omega_r2, a2, omega_q2, delta2, b2, g2):
    return omega_r1 * a1.dag() * a1 + duffing(omega_q1, delta1, b1) + g1 * (a1.dag()+a1) * (b1.dag() + b1) \
         + omega_r2 * a2.dag() * a2 + duffing(omega_q2, delta2, b2) + g2 * (a2.dag()+a2) * (b2.dag() + b2)
        

def is_identity(sequence):
    operation = QId
    for gate_str in sequence:
        gate = eval(gate_str)
        operation = operation @ gate
    is_id = abs( 2 - abs(np.trace(operation))) < 0.0001
    return is_id

############## GATES ##############

QId = np.array([ [1,0] ,
                 [0,1] ], 
                 dtype = np.complex128)

Xp = np.array([ [0,1] ,
                [1,0] ], 
                dtype = np.complex128)

X90p = np.sqrt(2)/2 * np.array([ [1,-1j] ,
                                 [-1j,1] ], 
                                 dtype = np.complex128)

X90m = np.sqrt(2)/2 * np.array([ [1,1j] ,
                                 [1j,1] ],
                                 dtype = np.complex128)

Y90p = np.sqrt(2)/2 * np.array([ [1,-1] ,
                                 [1,1]  ], 
                                 dtype = np.complex128)

Y90m = np.sqrt(2)/2 * np.array([ [1,1] ,
                                 [-1,1]  ],
                                 dtype = np.complex128)

def gaussian(t,T,amp):
    sigma = T / 6
    return amp * (np.exp(-(t - T / 2) ** 2 / (2 * sigma ** 2)) - np.exp(-T ** 2 / (8 * sigma ** 2))) / \
    (np.sqrt(2 * np.pi * sigma ** 2) - T * np.exp(-T ** 2 / (8 * sigma ** 2)))

def gaussian_der(t,T,amp):
    sigma = T / 6
    return amp * (np.exp(-(t - T / 2) ** 2 / (2 * sigma ** 2))) * (t - T / 2) / sigma ** 2 / \
    (np.sqrt(2 * np.pi * sigma ** 2) - T * np.exp(-T ** 2 / (8 * sigma ** 2)))


class CMARunner:
    def __init__(self):
        self.iter_num = 0
        self.fidelity_history = []
        self.parameter_history = []
        self.best_ever = {'result': np.inf, 'parameters': []}
        self.parameter_map = None
        self.n_parameters = 0

    def update_best_ever(self, results, solution_parameters):
        """
        Replace the best solution if the proposed solutions have a smaller
        function value.
        :param results: new function values
        :param solution_parameters: parameters that produced results.
        """
        for i in range(len(results)):
            if results[i] < self.best_ever['result']:
                self.best_ever['result'] = results[i]
                self.best_ever['parameters'] = solution_parameters[i]

    def unwrap_parameters(self, parameters):
        """
        convert a dictionary of parameters into a one dimensional parameter list readable by an optimizer
        from the format of the dictionary we also create a parameter map to rewrap such a list back to the dict
        :param parameters: dict object sorted by logical channel, qubits and pulses
        :return: one dimensional list of parameters
        """

        # Build a nested dictionary to list mapping if the
        # dictionary of parameters has never been encountered.
        if not self.parameter_map:
            idx = 0
            self.parameter_map = copy.deepcopy(parameters)
            for ch in parameters:
                self.parameter_map[ch] = {}
                for pulse in parameters[ch]:
                    self.parameter_map[ch][pulse] = {}
                    for param in parameters[ch][pulse]:
                        if param is not 'mapto':
                            self.parameter_map[ch][pulse][param] = idx
                            idx += 1

            self.n_parameters = idx

        # Convert the dictionary of parameters to a list.
        param_list = [None]*self.n_parameters
        for ch in self.parameter_map:
            for pulse in self.parameter_map[ch]:
                for param in self.parameter_map[ch][pulse]:
                    if param is not 'mapto':
                        idx = self.parameter_map[ch][pulse][param]
                        param_list[idx] = parameters[ch][pulse][param]

        return param_list

    def wrap_parameters(self, parameter_list):
        """
        This function converts a one dimensional list of parameters into a parameter dictionary readable by the sqore converter
        :param parameters: two dimensional list of parameter values indexed in the first dimension by solution index and in the 2nd according to the parameter map
        :return: dict of parameters labeled with logical channels, qubits and pulses
        """
        parameter_dict = copy.deepcopy(self.parameter_map)
        for ch in self.parameter_map:
            parameter_dict[ch] = {}

            for pulse in self.parameter_map[ch]:
                parameter_dict[ch][pulse] = {}

                for param in self.parameter_map[ch][pulse]:

                    if param is not 'mapto':
                        idx = self.parameter_map[ch][pulse][param]
                        parameter_dict[ch][pulse][param] = parameter_list[idx]
        return parameter_dict

    def save_history(self, rb_avg, solutions):
        """
        generate a history of rb fidelity and solutions
        :param rb_avg: measured RB fidelity
        :param solutions: corresponding solutions in parameter space
        """
        self.fidelity_history.append(rb_avg)
        self.parameter_history.append(solutions)

        # This is done at every iteration as we still do not cleanly kill the optimizer.
        with open('rb_point_fidelities.json', 'w') as fout:
            json.dump(self.fidelity_history, fout)

        with open('parameter_history.json', 'w') as fout:
            json.dump(self.parameter_history, fout)

class SimCMARunner(CMARunner):
    """
    Class to simulate Rb sequences based on parameter batches from CMA optimizer
    """

    def __init__(self, H0, psi0, omega_d,pulse_shape='drag',T=50e-9,dt=1e-11):
        CMARunner.__init__(self)
        self.H0 = H0
        self.psi0 = psi0
        self.T =T
        self.pulse_shape = pulse_shape
        self.ts = np.linspace(0, T, int(T/dt))
        self.omega_d = omega_d
        self.angles = {'X90p': 0, 'Y90p': 90, 'X90m': 180, 'Y90m': 270}

    def evaluate_sequences(self, sequences, solutions, qubits=1, rounds=300):
        U = {}
        N_r = self.H0.dims[0][0]
        N_q = self.H0.dims[0][1]
        n_r = self.psi0.dims[0][0]
        n_q = self.psi0.dims[0][1]
        b = tensor(qeye(N_r),destroy(N_q))
        U0 = rect_space([N_r,N_q],[n_r,n_q])
        F = []
        for i in range(len(solutions)):
            print(i)
            value = solutions[i]
            for gate in value['Q1'].keys():
                print(gate)
                new_params = value['Q1'][gate]
                new_params['angle'] = self.angles[gate]
                if 'mapto' in new_params.keys():
                    mapto = new_params['mapto']
                    del new_params['mapto']
                    map_params = value['Q1'][mapto]
                    for key in map_params.keys():
                        if key not in new_params.keys():
                            new_params[key] = map_params[key]
                drive = self.get_drive(value, b, t)
                H = lambda t: self.H0 + drive(t)
                U[gate] = evolution(H,U0,self.ts)[-1]
            F.append([])
            for seq in sequences:
                psif = self.psi0
                for clifford in seq:
                    psif = U[clifford]*psif
                F[i].append(1-tracedist(psif.ptrace(1),self.psi0.ptrace(1)))
        return np.average(F,1)

    def get_drive(self,value,a,t):
        ## based on params build a pulse specified in the BatchRunner
        T = self.ts[-1]
        drive = gaussian(t, T, value['amp'])*(a.dag() + a) - 1.0j * gaussian_der(t, T, value['amp']) * (a.dag() - a) / value['delta']
        return 0.5*np.exp(1.0j*value['phase'])*drive

    # The function generating the single length RB sequences
    def single_length_RB(self,RB_number, RB_length):
        S = []
        for seq_idx in range(RB_number):
            seq = rand.choice(Gates, size=RB_length)
            while not is_identity(seq):
                seq = rand.choice(Gates, size=RB_length)
            S.append(seq)
        return S



class TestCMARunner(CMARunner):
    """
    Test batch runner that does not connect to the experiment but returns
    pseudo-random data that can be used for testing.
    """
    def __init__(self, dir_name, base_dir):
        CMARunner.__init__(self)
        self.dir_name = dir_name  # Name of the directory where the data is saved.
        self.base_dir = base_dir  # Base directory. Contains directory dir_name.

        dir_path = self.base_dir + self.dir_name
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def evaluate_sequences(self, rep, rb_len, solutions, rate=0.1, qubits=None, rounds=None):
        """
        Returns some noisy random numbers that immitate data.
        :param rep:
        :param rb_len:
        :param solutions:
        :param rate:
        :return:
        """
        start = 5
        r = 0.1**(rate*(self.iter_num+start))
        p = 1-2*r

        f = 0.5 - 0.5 * p**rb_len

        noise = np.cos(np.random.normal(0, r/np.sqrt(rep), len(solutions)))

        results = [f * n for n in noise]

        self.iter_num += 1
        self.update_best_ever(results, solutions)

        return results



#Milestone 1
# TODO: later, Floquet solver to find good first guess.
# TODO: debug everything
# TODO: change to be rectangular evolution
# TODO: do not propagate gates in initial_parameters but propagate all RB gates and just vary the ones with presence in initial parameters
# TODO: return gate fidelity instead of sequence fidelity by normalizing with the analytic formular of ORBIT -
# TODO: produce RB curve of well calibrated gate and see how it looks like
# TODO: debug everything
#Mileston 2
# more complicated model
# 


