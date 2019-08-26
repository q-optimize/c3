import uuid
import numpy as np
import tensorflow as tf
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
            self.ts = tf.linspace(self.t_start, self.t_end, self.slice_num)
        else:
            self.ts = None





class Mixer(Device):
    """
    """
    def __init__(
            self,
            t_start = None,
            t_end = None
            ):

        self.t_start = t_start
        self.t_end = t_end

        self.slice_num = None
        self.ts = []


        self.Inphase = []
        self.Quadrature = []


        self.output = []


    def combine(self, res_key):

        self.calc_slice_num(res_key)
        self.create_ts(res_key)


        ts = self.ts

        carrier_group = self.ressource_groups["carrier"]


        signal = self.ressources[0]
        for comp in signal.comps:
            if carrier_group in comp.groups:
                omega_lo = comp.params["freq"]


        self.output = tf.zeros_like(ts)

        self.output += (self.Inphase * tf.cos(omega_lo * ts) +
                        self.Quadrature * tf.sin(omega_lo * ts))


class AWG(Device):
    """
    """
    def __init__(
            self,
            t_start = None,
            t_end = None,
            ):

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
        I_components = []
        Q_components = []

        signal = self.ressources[0]
        for comp in signal.comps:
            if comp_group in comp.groups:

                amp = comp.params['amp']

                amp_tot_sq += amp**2

                xy_angle = comp.params['xy_angle']
                freq_offset = comp.params['freq_offset']
                I_components.append(
                    amp * comp.get_shape_values(ts) *
                    tf.cos(xy_angle + freq_offset * ts)
                    )
                Q_components.append(
                    amp * comp.get_shape_values(ts) *
                    tf.sin(xy_angle + freq_offset * ts)
                    )

        norm = tf.sqrt(amp_tot_sq)
        Inphase = tf.add_n(I_components)/norm
        Quadrature = tf.add_n(Q_components)/norm

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




#     def plot_fft_IQ_components(self, axs=None):


        # print("WARNING: still have to adjust the x-axis")

        # """ Plotting control functions """
        # plt.rcParams['figure.dpi'] = 100
        # IQ = self.get_IQ()
        # fft_IQ = {}
        # fft_I = np.fft.fft(IQ['I'])
        # fft_Q = np.fft.fft(IQ['Q'])
        # fft_IQ['I'] = np.fft.fftshift(fft_I.real / max(fft_I.real))
        # fft_IQ['Q'] = np.fft.fftshift(fft_Q.real / max(fft_Q.real))

        # fig, axs = plt.subplots(2, 1)
        # axs[0].plot(self.ts[0] * self.res[0], fft_IQ['I'])
        # axs[1].plot(self.ts[0] * self.res[0], fft_IQ['Q'])
        # # I (Kevin) don't really understand the behaviour of plt.show()
        # # here. If I only put plt.show(block=False), I get error messages
        # # on my system at home. Adding a second plt.show() resolves that
        # # issue???
        # # look at:  https://github.com/matplotlib/matplotlib/issues/12692/
        # plt.show(block=False)
        # plt.show()



class Generator:
    """
    """
    def __init__(
            self,
            devices = {}
            ):

        self.devices = devices


    def generate_signal(self):
        ####
        #
        # PLACEHOLDER
        # HAS TO BE CODED BY THE USER. BUT HAS TO RETURN np.array OF SIGNAL
        #
        ####
        raise NotImplementedError()
