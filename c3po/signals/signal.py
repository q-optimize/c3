import uuid
import numpy as np
import matplotlib.pyplot as plt


class Signal:
    """

    """
    def __init__(
            self,
            t_start = None,
            t_end = None,
            res = [],
            comps = []
            ):


        # make a random UUID which uniquely identifies/represents the component
        # https://docs.python.org/2/library/uuid.html#uuid.uuid4
        self.__uuid = uuid.uuid4()

        self.t_start = t_start
        self.t_end = t_end
        self.res = res

        self.slice_num = []
        self.calc_slice_num()

        self.ts = []
        self.create_ts()

        self.comps = comps

        self.history = []



    def get_uuid(self):
        return self.__uuid


    def set_uuid(self, uuid):
        self.__uuid = uuid


    def calc_slice_num(self):
        if self.t_start != None and self.t_end != None and self.res != []:
            for r in self.res:
                self.slice_num.append(int(
                    np.abs(self.t_start - self.t_end) * r) + 1)


    def create_ts(self):
        if self.t_start != None and self.t_end != None and self.slice_num != []:
            for num in self.slice_num:
                self.ts.append(np.linspace(self.t_start, self.t_end, num))


    def get_parameter_value(self, key, uuid):
        for comp in self.comps:
            if uuid == comp.get_uuid():
                return comp.params[key]


    def set_parameter_value(self, key, uuid, val):
        for comp in self.comps:
            if uuid == comp.get_uuid():
                comp.params[key] = val


    def get_parameter_bounds(self, key, uuid):
        for comp in self.comps:
            if uuid == comp.get_uuid():
                return comp.bounds[key]


    def set_parameter_bounds(self, key, uuid, bounds):
        for comp in self.comps:
            if uuid == comp.get_uuid():
                comp.bounds[key] = bounds


    def get_parameters(self):
        params = {}

        for comp in self.comps:
            for key in comp.params.keys():
                if key not in params:
                    params[key] = {}

                uuid = comp.get_uuid()
                params[key][uuid] = {}

                params[key][uuid]['value'] = comp.params[key]
                if key in comp.bounds:
                    params[key][uuid]['bounds'] = comp.bounds[key]

        return params


    def save_params_to_history(self, name):
        self.history.append((name, self.get_parameters()))


    def get_history(self):
        return self.history


    def generate_opt_map(self, opt_map={}):
        sig_id = self.get_uuid()
        for cmp in self.comps:
            for key in cmp.params.keys():
                entry = (cmp.desc, sig_id, cmp.get_uuid())
                if key in opt_map.keys():
                    opt_map[key].append(entry)
                else:
                    opt_map[key] = [(entry)]

        return opt_map

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
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.ts[1] * self.res[0], signal)
        ax.set_xlabel('Time [ns]')
    #    plt.show(block=False)


    def plot_fft_signal(self):

        print("WARNING: still have to adjust the x-axis")

        """ Plotting control functions """
        plt.rcParams['figure.dpi'] = 100
        signal = self.generate_signal()

        fft_signal = np.fft.fft(signal)
        fft_signal = np.fft.fftshift(fft_signal.real / max(fft_signal.real))

        plt.plot(self.ts[0] * self.res[0], fft_signal)

        plt.show(block=False)
        plt.show()



####
#
# CHILD CLASS OF class Signal
#
####
class IQ(Signal):
    def get_IQ(self, carrier_uuid = None):
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
            if carrier_uuid is not None and comp.get_uuid() == carrier_uuid:
                carrier = comp


        omega_d = carrier.params['freq']

        amp_tot_sq = 0
        components = []
        for comp in self.comps:
            if "carrier" in comp.desc:
                continue

            amp = comp.params['amp']

            amp_tot_sq += amp**2

            xy_angle = comp.params['xy_angle']
            freq_offset = comp.params['freq_offset']
            components.append(
                amp * comp.get_shape_values(self.ts[0]) *
                np.exp(1j * (xy_angle + freq_offset * self.ts[0]))
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
        """

        AWG_I = np.interp(self.ts[1], self.ts[0], IQ['I'])
        AWG_Q = np.interp(self.ts[1], self.ts[0], IQ['Q'])
        amp = IQ['amp']
        omega_d = IQ['omega']

        sig = np.zeros_like(self.ts[1])

        sig += amp * (AWG_I * np.cos(omega_d * self.ts[1]) +
                      AWG_Q * np.sin(omega_d * self.ts[1]))

        return sig


    def plot_IQ_components(self):
        """ Plotting control functions """

        IQ = self.get_IQ()
        plt.rcParams['figure.dpi'] = 100
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(self.ts[0] * self.res[0], IQ['I'])
        axs[1].plot(self.ts[0] * self.res[0], IQ['Q'])
        # I (Kevin) don't really understand the behaviour of plt.show()
        # here. If I only put plt.show(block=False), I get error messages
        # on my system at home. Adding a second plt.show() resolves that
        # issue???
        # look at:  https://github.com/matplotlib/matplotlib/issues/12692/
        plt.show(block=False)
        plt.show()

    def plot_IQ(self, ts, Is, Qs):
        """
        Plot (hopefully) into an existing figure.
        """
        plt.cla()
        plt.plot(ts / 1e-9, Is)
        plt.plot(ts / 1e-9, Qs)
        plt.legend(("I", "Q"))
        plt.ylabel('I/Q')
        plt.xlabel('Time[ns]')
        plt.tick_params('both',direction='in')
        plt.tick_params('both', direction='in')


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
        axs[0].plot(self.ts[1] * self.res[0], fft_IQ['I'])
        axs[1].plot(self.ts[1] * self.res[0], fft_IQ['Q'])
        # I (Kevin) don't really understand the behaviour of plt.show()
        # here. If I only put plt.show(block=False), I get error messages
        # on my system at home. Adding a second plt.show() resolves that
        # issue???
        # look at:  https://github.com/matplotlib/matplotlib/issues/12692/
        plt.show(block=False)
        plt.show()
