from c3po.signals.envelopes import *
from c3po.signals.component import Component as Comp
from c3po.signals.signal import Signal as Signal

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


def my_flattop(t, params):
    t_up = params['T_up']
    t_down = params['T_down']
    return flattop(t, t_up, t_down)


p1 = Comp(
    desc = "pulse1",
    shape = my_flattop,
    params = flattop_params1,
    bounds = params_bounds
)
print("p1 uuid: " + str(p1.get_uuid()))

p2 = Comp(
    desc = "pulse2",
    shape = my_flattop,
    params = flattop_params2,
    bounds = params_bounds
)
print("p2 uuid: " + str(p2.get_uuid()))

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

carr = Comp(
    desc = "carrier",
    params = carrier_parameters
)
print("carr uuid: " + str(carr.get_uuid()))


comps = []
comps.append(carr)
comps.append(p1)
comps.append(p2)




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


sig.plot_IQ_components()

sig.plot_fft_IQ_components()

sig.plot_signal()

sig.plot_fft_signal()


print(sig.get_parameters())
print(" ")
print(" ")
print(" ")

print(sig.get_history())
print(" ")
print(" ")
print(" ")


sig.save_params_to_history("initial")

print(sig.get_history())
print(" ")
print(" ")
print(" ")


sig.save_params_to_history("test2")

print(sig.get_history())


