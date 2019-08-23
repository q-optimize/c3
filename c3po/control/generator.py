import uuid
import numpy as np
import matplotlib.pyplot as plt





class Device:
    """
    """
    def __init__(
            self,
            name = " ",
            desc = " ",
            comment = " ",
            resolutions = {},
            ressources = [],
            ressource_groups = {}
            ):

        self.name = name
        self.desc = desc
        self.comment = comment
        self.resolutions = resolutions
        self.ressources = ressources
        self.ressource_groups = ressource_groups


    def calc_slice_num(self, res_key):
        res = self.resolutions[res_key]

        if self.t_start != None and self.t_end != None and res != None:
            self.slice_num = (int(np.abs(self.t_start - self.t_end) * res) + 1)
        else:
            self.slice_num = None


    def create_ts(self, res_key):
        if self.t_start != None and self.t_end != None and self.slice_num != None:
            self.ts = np.linspace(self.t_start, self.t_end, self.slice_num)
        else:
            self.ts = None





class Mixer(Device):
    """
    """
    def __init__(
            self,
            name = " ",
            desc = " ",
            comment = " ",
            resolutions = {},
            ressources = [],
            ressource_groups = {},
            t_start = None,
            t_end = None,
            Inphase = [],
            Quadrature = []
            ):

        super().__init__(name, desc, comment, resolutions, ressources, ressource_groups)

        self.t_start = t_start
        self.t_end = t_end

        self.slice_num = None
        self.ts = []

        self.Inphase = Inphase
        self.Quadrature = Quadrature

        self.output = []


    def combine(self, res_key):

        self.calc_slice_num(res_key)
        self.create_ts(res_key)


        ts = self.ts

        carrier_group = self.ressource_groups["carrier"]


        control = self.ressources[0]
        for comp in control.comps:
            if carrier_group in comp.groups:
                omega_lo = comp.params["freq"]


        self.output = np.zeros_like(ts)

        self.output += (self.Inphase * np.cos(omega_lo * ts) +
                        self.Quadrature * np.sin(omega_lo * ts))


class AWG(Device):
    """
    """
    def __init__(
            self,
            name = " ",
            desc = " ",
            comment = " ",
            resolutions = {},
            ressources = [],
            ressource_groups = {},
            t_start = None,
            t_end = None
            ):

        super().__init__(name, desc, comment, resolutions, ressources, ressource_groups)

        self.t_start = t_start
        self.t_end = t_end

        self.slice_num = None
        self.ts = []

        self.Inphase = []
        self.Quadrature = []
        self.amp_tot_sq = None


    def create_IQ(self, res_key):
        """
        Construct the in-phase (I) and quadrature (Q) components of the

        control signals. These are universal to either experiment or
        simulation. In the experiment these will be routed to AWG and mixer
        electronics, while in the simulation they provide the shapes of the
        controlfields to be added to the Hamiltonian.

        """

        self.calc_slice_num(res_key)
        self.create_ts(res_key)

        ts = self.ts


        Inphase = []
        Quadrature = []

        comp_group = self.ressource_groups["comp"]


        amp_tot_sq = 0
        components = []

        control = self.ressources[0]
        for comp in control.comps:
            if comp_group in comp.groups:

                amp = comp.params['amp']

                amp_tot_sq += amp**2

                xy_angle = comp.params['xy_angle']
                freq_offset = comp.params['freq_offset']
                components.append(
                    amp * comp.get_shape_values(ts) *
                    np.exp(1j * (xy_angle + freq_offset * ts))
                    )

        norm = np.sqrt(amp_tot_sq)
        Inphase = np.real(np.sum(components, axis = 0)) / norm
        Quadrature = np.imag(np.sum(components, axis = 0)) / norm

        self.amp_tot_sq = amp_tot_sq
        self.Inphase = Inphase
        self.Quadrature = Quadrature


    def get_I(self):
        return self.amp_tot_sq * self.Inphase


    def get_Q(self):
        return self.amp_tot_sq * self.Quadrature



    def plot_IQ_components(self, res_key):
        """ Plotting control functions """

        ts = self.ts
        res = self.resolutions[res_key]
        I = self.Inphase
        Q = self.Quadrature

        plt.rcParams['figure.dpi'] = 100
        fig, axs = plt.subplots(2, 1)

        axs[0].plot(ts * res, I)
        axs[1].plot(ts * res, Q)
        # I (Kevin) don't really understand the behaviour of plt.show()
        # here. If I only put plt.show(block=False), I get error messages
        # on my system at home. Adding a second plt.show() resolves that
        # issue???
        # look at:  https://github.com/matplotlib/matplotlib/issues/12692/
        plt.show(block=False)
        plt.show()



    # def plot_IQ(self, ts, Is, Qs):
        # """
        # Plot (hopefully) into an existing figure.
        # """
        # plt.cla()
        # plt.plot(ts / 1e-9, Is)
        # plt.plot(ts / 1e-9, Qs)
        # plt.legend(("I", "Q"))
        # plt.ylabel('I/Q')
        # plt.xlabel('Time[ns]')
        # plt.tick_params('both',direction='in')
        # plt.tick_params('both', direction='in')




    def plot_fft_IQ_components(self, res_key, axs=None):


        print("WARNING: still have to adjust the x-axis")


        ts = self.ts
        res = self.resolutions[res_key]
        I = self.Inphase
        Q = self.Quadrature


        """ Plotting control functions """
        plt.rcParams['figure.dpi'] = 100

        fft_I = np.fft.fft(I)
        fft_Q = np.fft.fft(Q)
        fft_I = np.fft.fftshift(fft_I.real / max(fft_I.real))
        fft_Q = np.fft.fftshift(fft_Q.real / max(fft_Q.real))

        fig, axs = plt.subplots(2, 1)
        axs[0].plot(ts * res, fft_I)
        axs[1].plot(ts * res, fft_Q)
        # I (Kevin) don't really understand the behaviour of plt.show()
        # here. If I only put plt.show(block=False), I get error messages
        # on my system at home. Adding a second plt.show() resolves that
        # issue???
        # look at:  https://github.com/matplotlib/matplotlib/issues/12692/
        plt.show(block=False)
        plt.show()



class Generator:
    """

    """
    def __init__(
            self,
            devices = {},
            resolutions = {},
            ressources = [],
            ressource_groups = {}
            ):

        self.devices = devices
        self.resolutions = resolutions
        self.ressources = ressources
        self.ressource_groups = ressource_groups

        self.output = None


    def generate_signals(self, ressources = []):
        ####
        #
        # PLACEHOLDER
        #
        ####
        raise NotImplementedError()



    def plot_signals(self, ressources = []):

        if ressources != []:
            self.generate_signals(ressources)

        for entry in self.output:
            ctrl_name = entry[0]
            control = self.output[entry]

            """ Plotting control functions """
            plt.rcParams['figure.dpi'] = 100


            ts = control["ts"]
            signal = control["signal"]

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(ts, signal)
            ax.set_xlabel('Time [ns]')
            plt.title(ctrl_name)

            plt.show(block=True)


    def plot_fft_signals(self, ressources = []):

        if ressources != []:
            self.generate_signals(ressources)

        print("WARNING: still have to adjust the x-axis")

        for entry in self.output:
            ctrl_name = entry[0]
            control = self.output[entry]


            """ Plotting control functions """
            plt.rcParams['figure.dpi'] = 100

            ts = control["ts"]
            signal = control["signal"]


            fft_signal = np.fft.fft(signal)
            fft_signal = np.fft.fftshift(fft_signal.real / max(fft_signal.real))

            plt.plot(ts, fft_signal)
            plt.title(ctrl_name + " (fft)")

            plt.show(block=True)


