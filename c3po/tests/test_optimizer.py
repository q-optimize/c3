from c3po.signals.envelopes import *
from c3po.signals.component import Component as Comp
from c3po.signals.signal import Signal as Signal

from c3po.optimizer.optimizer import Optimizer as Optimizer

import matplotlib.pyplot as plt

flattop_params1 = {
    'amp' : 15e6 * 2 * np.pi,
    'T_up' : 5e-9,
    'T_down' : 45e-9,
    'xy_angle' : 0,
    'freq_offset' : 0e6 * 2 * np.pi
}

flattop_params2 = {
    'amp' : 3e6 * 2 * np.pi,
    'T_up' : 25e-9,
    'T_down' : 30e-9,
    'xy_angle' : np.pi / 2.0,
    'freq_offset' : 0e6 * 2 * np.pi
}

params_bounds = {
    'T_up' : [2e-9, 98e-9],
    'T_down' : [2e-9, 98e-9],
    'freq_offset' : [-1e9 * 2 * np.pi, 1e9 * 2 * np.pi]
}

p1 = Comp(desc = "pulse1", shape = flattop, params = flattop_params1, bounds = params_bounds)
print("p1 id: " + str(p1.get_id()))

p2 = Comp(desc = "pulse2", shape = flattop, params = flattop_params2, bounds = params_bounds)
print("p2 id: " + str(p2.get_id()))

####
# Below code: For checking the single signal components 
####

# t = np.linspace(0, 150e-9, int(150e-9*1e9))
# plt.plot(t, p1.get_shape_values(t))
# plt.plot(t, p2.get_shape_values(t))
# plt.show()


carrier_parameters = {
    'freq' : 6e9 * 2 * np.pi
}

carrier_bounds = {
    'freq' : [2e9 * 2 * np.pi, 10e9 * 2 * np.pi]
}

carr = Comp(desc = "carrier", params = carrier_parameters, bounds = carrier_bounds)
print("carr id: " + str(carr.get_id()))


comps = []
comps.append(carr)
comps.append(p1)
comps.append(p2)




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


        carrier_params = comp.params
        omega_d = carrier_parameters['freq']

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



sig = IQ()
sig.t_start = 0
sig.t_end = 150e-9
sig.res = 1e9

sig.calc_slice_num()
sig.create_ts()

sig.comps = comps


opt_params = {
    'T_up' : [1,2],
    'T_down' : [1,2],
    'freq' : [3]
}

####
#
# Maybe rather simplify the structure of opt_params to the below version
#
####

# opt_params = {
    # 'T_up' : {
        # 1 : None,
        # 2 : None
        # },
    # 'T_down' : {
        # 1 : None,
        # 2 : None
        # },
    # 'freq' : {
        # 3 : None
    # }
# }



optim = Optimizer()


# print("Signal Parameter Values")
# print(sig.get_parameters())
# print(" ")
# print(" ")
# print(" ")


# optim.get_corresponding_signal_parameters(sig, opt_params)

# print(opt_params)
# print(" ")
# print(" ")
# print(" ")

# opt_params['values'] = [0, 0, 0, 0, 0]
# opt_params['bounds'] = [[0,0], [0,0], [0,0], [0,0], [0.0]]
# print(opt_params)
# print(" ")
# print(" ")
# print(" ")


# optim.set_corresponding_signal_parameters(sig, opt_params)

# optim.get_corresponding_signal_parameters(sig, opt_params)

# print(opt_params)
# print(" ")
# print(" ")
# print(" ")



# print("Signal Parameter Values")
# print(sig.get_parameters())


opt_settings = {

}

optim.optimize_signal(
    signal= sig,
    opt_params = opt_params,
    opt = 'cmaes',
    settings = opt_settings)








