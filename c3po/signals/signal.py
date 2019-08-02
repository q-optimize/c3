import numpy as np
import matplotlib.pyplot as plt


class Signal:
    # define an internal id for the created instance of the pulse object
    # as private attributes are not per se available in python use 'hack'
    # see: https://stackoverflow.com/questions/1641219/does-python-have-private-variables-in-classes
    __id = 0

    def __init__(
            self,
            t_start = None,
            t_end = None,
            res = 1e9,
            comps = []
            ):

        self.__id = Signal.__id + 1
        Signal.__id = self.__id

        self.t_start = t_start
        self.t_end = t_end
        self.res = res

        self.slice_num = 0
        self.calc_slice_num()

        self.ts = None
        self.create_ts()


        self.comps = comps

        self.history = []



    def calc_slice_num(self):
        if self.t_start != None and self.t_end != None and self.res != None:
            self.slice_num = int(np.abs(self.t_start - self.t_end) * self.res)


    def create_ts(self):
        if self.t_start != None and self.t_end != None and self.slice_num != None:
            self.ts = np.linspace(self.t_start, self.t_end, self.slice_num)


    def get_parameters(self):
        params = {}

        for comp in self.comps:
            for key in comp.params.keys():
                if key not in params:
                    params[key] = {}

                comp_id = comp.get_id()
                params[key][comp_id] = {}

                params[key][comp_id]['value'] = comp.params[key]
                if key in comp.bounds:
                    params[key][comp_id]['bounds'] = comp.bounds[key]

        return params


    def save_params_to_history(self, name):
        self.history.append((name, self.get_parameters()))


    def get_history(self):
        return self.history


    def get_id(self):
        return self.__id



    ####
    #
    # HAS TO BE CODED BY THE USER. BUT HAS TO RETURN np.array OF SIGNAL
    #
    ####
    def generate_signal(self):
        raise NotImplementedError()


    def plot_signal(self):
        """ Plotting control functions """
        plt.rcParams['figure.dpi'] = 100
        signal = self.generate_signal()

        plt.plot(self.ts * self.res, signal)

        plt.show(block=False)
        plt.show()


    def plot_fft_signal(self):

        print("WARNING: still have to adjust the x-axis")

        """ Plotting control functions """
        plt.rcParams['figure.dpi'] = 100
        signal = self.generate_signal()

        fft_signal = np.fft.fft(signal)
        fft_signal = np.fft.fftshift(fft_signal.real / max(fft_signal.real))

        plt.plot(self.ts * self.res, fft_signal)

        plt.show(block=False)
        plt.show()




