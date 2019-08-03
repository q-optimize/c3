from c3po.signals.envelopes import *

from c3po.optimizer.optimizer import Optimizer as Optimizer



optim = Optimizer()



opt_settings = {

}

def evaluate_signal(samples_rescaled, signal = None):
    print(" ")



values = [5e-09, 2.5e-08, 4.5e-08, 3e-08, 37699111843.077515]
bounds = [[2e-09, 9.8e-08], [2e-09, 9.8e-08], [2e-09, 9.8e-08], [2e-09, 9.8e-08], [12566370614.359173, 62831853071.79586]]


optim.cmaes(values, bounds, opt_settings, evaluate_signal, signal = None)



