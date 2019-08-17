
from c3po.cobj.cfunc import CFunc as CFunc


import numpy as np



class Device:
    """

    """
    def __init__(
            self,
            string = None,
            comment = None,
            res = None,
            ressources = [],
            cfuncs = []
            ):

        self.string = string
        self.comment = comment
        self.res = res
        self.cfuncs = cfuncs



def get_IQ(t, attr):
    """
    Construct the in-phase (I) and quadrature (Q) components of the

    control signals. These are universal to either experiment or
    simulation. In the experiment these will be routed to AWG and mixer
    electronics, while in the simulation they provide the shapes of the
    controlfields to be added to the Hamiltonian.

    """

    print(attr)


    Inphase = []
    Quadrature = []

    amp_tot_sq = 0
    components = []
    for comp in attr.values():
#        amp = comp["amp"]
        amp = 1
        amp_tot_sq += amp**2

        components.append(comp)

    norm = np.sqrt(amp_tot_sq)
    Inphase = np.real(np.sum(components, axis = 0)) / norm
    Quadrature = np.imag(np.sum(components, axis = 0)) / norm


    signal = {}
    signal['amp'] = amp_tot_sq
    signal['I'] = Inphase
    signal['Q'] = Quadrature

    return signal


IQ = CFunc(
    string = "IQ",
    comment = "IQ Comp of Signal",
    latex = "IQ(t)",
    params = [],
    insts = [],
    deps = [],
    body = get_IQ,
    body_latex = " ... "
)


class AWG(Device):
    """

    """
    def __init__(
            self,
            t_start = None,
            t_end = None,
            cfunc_IQ = IQ
            ):

        self.t_start = t_start
        self.t_end = t_end
        self.cfunc_IQ = cfunc_IQ
        self.slice_num = self.calc_slice_num()
        self.ts = self.create_ts()


    def calc_slice_num(self):
        if self.t_start != None and self.t_end != None and self.res != None:
            self.slice_num = (int(np.abs(self.t_start - self.t_end) * self.res) + 1)
        else:
            self.slice_num = None


    def create_ts(self):
        if self.t_start != None and self.t_end != None and self.slice_num != None:
            self.ts = np.linspace(self.t_start, self.t_end, self.slice_num)
        else:
            self.ts = None


    def feed_ressources(self, ressources):
        self.cfunc_IQ.deps = ressources


    def get_I(self):
        if self.slice_num == None:
            self.calc_slice_num()

        if self.ts == []:
            self.create_ts()


        print("self.slice_num: " + str(self.slice_num))
        print("self.ts : " + str(self.ts))

        out = self.cfunc_IQ.evaluate(self.ts)
        return out["amp"] * out["I"]


    def get_Q(self):
        if self.slice_num == None:
            self.calc_slice_num()

        if self.ts == []:
            self.create_ts()


        out = self.cfunc_IQ.evaluate(self.ts)
        return out["amp"] * out["Q"]


class Generator:
    """

    """
    def __init__(
            self,
            devs = []
            ):

        self.devs = devs

