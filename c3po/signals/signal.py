import numpy as np
import matplotlib.pyplot as plt


class Signal:
    # define an internal id for the created instance of the signal object
    # as private attributes are not a thing in python use this 'hack' by 
    # naming the variable with two underscores. this will prompt the compiler
    # to internally rename the variable (so it's not accessible anymore under
    # the original name). this is the most dumb thing I have ever seen but 
    # apparently it's the 'pythonian way'. #internalscreaming
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


    def get_parameter_value(self, key, comp_id):
        for comp in self.comps:
            if comp_id == comp.get_id():
                return comp.params[key]


    def set_parameter_value(self, key, comp_id, val):
        for comp in self.comps:
            if comp_id == comp.get_id():
                comp.params[key] = val


    def get_parameter_bounds(self, key, comp_id):
        for comp in self.comps:
            if comp_id == comp.get_id():
                return comp.bounds[key]


    def set_parameter_bounds(self, key, comp_id, bounds):
        for comp in self.comps:
            if comp_id == comp.get_id():
                comp.bounds[key] = bounds


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


    def generate_signal(self):
        ####
        #
        # PLACEHOLDER
        # HAS TO BE CODED BY THE USER. BUT HAS TO RETURN np.array OF SIGNAL
        #
        ####
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



####
#
# CHILD CLASS OF class Signal
#
####
class IQ(Signal):
    def get_IQ(self):
        """
        Construct the in-phase (I) and quadrature (Q) components of the

        control signals. These are universal to either experiment or
        simulation. In the experiment these will be routed to AWG and mixer
        electronics, while in the simulation they provide the shapes of the
        controlfields to be added to the Hamiltonian.

        """
        signal = {}

        Inphase = []
        Quadrature = []

        for comp in self.comps:
            # Identification via desc
            if comp.desc == "carrier":
                carrier = comp

            # Identification via id
            if comp.get_id() == 3:
                carrier = comp


        omega_d = carrier.params['freq']

        amp_tot_sq = 0
        components = []
        for comp in self.comps:
            if "pulse" not in comp.desc:
                continue

            amp = comp.params['amp']

            amp_tot_sq += amp**2

            xy_angle = comp.params['xy_angle']
            freq_offset = comp.params['freq_offset']
            components.append(
                amp * comp.get_shape_values(self.ts) *
                np.exp(1j * (xy_angle + freq_offset * self.ts))
                )

        norm = np.sqrt(amp_tot_sq)
        Inphase = np.real(np.sum(components, axis = 0)) / norm
        Quadrature = np.imag(np.sum(components, axis = 0)) / norm

        signal['omega'] = omega_d
        signal['amp'] = amp_tot_sq
        signal['I'] = Inphase
        signal['Q'] = Quadrature

        return signal



    def generate_signal(self):
        IQ = self.get_IQ()
        """
        NICO: Federico raised the question if the xy_angle should be added
        here. After some research, this should be the correct way. The
        signal is E = I cos() + Q sin(), such that E^2 = I^2+Q^2.
        """
        sig = np.zeros_like(self.ts)

        AWG_I = IQ['I']
        AWG_Q = IQ['Q']
        amp = IQ['amp']
        omega_d = IQ['omega']

        sig += amp * (AWG_I * np.cos(omega_d * self.ts) +
                      AWG_Q * np.sin(omega_d * self.ts))

        return sig


    def plot_IQ_components(self, axs=None):
        """ Plotting control functions """
        plt.rcParams['figure.dpi'] = 100
        IQ = self.get_IQ()
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(self.ts * self.res, IQ['I'])
        axs[1].plot(self.ts * self.res, IQ['Q'])
        # I (Kevin) don't really understand the behaviour of plt.show()
        # here. If I only put plt.show(block=False), I get error messages 
        # on my system at home. Adding a second plt.show() resolves that 
        # issue???
        # look at:  https://github.com/matplotlib/matplotlib/issues/12692/
        plt.show(block=False)
        plt.show()


    def plot_fft_IQ_components(self, axs=None):


        print("WARNING: still have to adjust the x-axis")

        """ Plotting control functions """
        plt.rcParams['figure.dpi'] = 100
        IQ = self.get_IQ()
        fft_IQ = {}
        fft_I = np.fft.fft(IQ['I'])
        fft_Q = np.fft.fft(IQ['Q'])
        fft_IQ['I'] = np.fft.fftshift(fft_I.real / max(fft_I.real))
        fft_IQ['Q'] = np.fft.fftshift(fft_Q.real / max(fft_Q.real))

        fig, axs = plt.subplots(2, 1)
        axs[0].plot(self.ts * self.res, fft_IQ['I'])
        axs[1].plot(self.ts * self.res, fft_IQ['Q'])
        # I (Kevin) don't really understand the behaviour of plt.show()
        # here. If I only put plt.show(block=False), I get error messages 
        # on my system at home. Adding a second plt.show() resolves that 
        # issue???
        # look at:  https://github.com/matplotlib/matplotlib/issues/12692/
        plt.show(block=False)
        plt.show()







