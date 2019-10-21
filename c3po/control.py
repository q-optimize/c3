import numpy as np
from c3po.component import C3obj


class GateSet:
    """Contains all operation that can be performed by the experiment."""

    def __init__(self, controlset_list):
        self.controlsets = controlset_list


class ControlSet:
    """Contains all control drives (i.e. to all lines) for an operation."""

    def __init__(self, control_list):
        self.controls = control_list

    def get_corresponding_control_parameters(self, opt_map):
        """
        Return list of values and bounds of parameters in opt_map.

        Takes a list of paramaters that are supposed to be optimized
        and returns the corresponding values and bounds.

        Parameters
        -------
        opt_map : list
            List of parameters that are supposed to be optimized, specified
            with a tupple of control name, component name and parameter.

            Example:
            opt_map = [
                ('line1','gauss1','sigma')
                ('line1','gauss2','amp')
                ('line1','gauss2','freq')
                ('line2','DC','amp')
                ]

        Returns
        -------
        opt_params : dict
            Dictionary with values, bounds lists.

            Example:
            opt_params = {
                'values': [0,           0,           0,             0],
                'bounds': [[0, 0],      [0, 0],      [0, 0],        [0.0]]
                }

        """
        opt_params = {}
        opt_params['values'] = []
        opt_params['bounds'] = []

        for id in opt_map:
            ctrl_name = id[0]
            comp_name = id[1]
            param = id[2]
            for control in self.controls:
                if ctrl_name == control.name:
                    value = control.get_parameter_value(param, comp_name)
                    bounds = control.get_parameter_bounds(param, comp_name)
                    opt_params['values'].append(value)
                    opt_params['bounds'].append(bounds)
        return opt_params

    def set_corresponding_control_parameters(self, opt_params, opt_map):
        """Set the values in opt_params in the original control class."""
        # TODO make this more efficient: check index of control name beforehand
        set_bounds = ('bounds' in opt_params)
        for indx in range(len(opt_map)):
            id = opt_map[indx]
            ctrl_name = id[0]
            comp_name = id[1]
            param = id[2]
            for control in self.controls:
                if ctrl_name == control.name:
                    value = opt_params['values'][indx]
                    control.set_parameter_value(param, comp_name, value)
                    if set_bounds:
                        bounds = opt_params['bounds'][indx]
                        control.set_parameter_bounds(param, comp_name, bounds)

    def get_values_bounds(self, opt_params):
        values = opt_params['values']
        bounds = opt_params['bounds']
        return values, bounds

    def update_controls(self, values, opt_map):
        opt_params = {}
        opt_params['values'] = values
        self.set_corresponding_control_parameters(opt_params, opt_map)

    def save_params_to_history(self, name):
        # TODO save history in ControlSet too?
        for control in self.controls:
            control.save_params_to_history(name)

    def get_history(self, opt_map=None):
        # TODO make function return only history of given params
        for control in self.controls:
            control.get_history()
        return None


class Control(C3obj):
    """Collection of components making up the control signal for a line."""

    def __init__(
            self,
            name: str = " ",
            desc: str = " ",
            comment: str = " ",
            t_start: np.float64 = 0.0,
            t_end: np.float64 = 0.0,
            comps: list = []
            ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment
            )
        self.t_start = t_start
        self.t_end = t_end
        self.history = {}
        self.comps = comps

    def get_parameter_value(self, param, comp_name):
        for comp in self.comps:
            if comp_name == comp.name:
                return comp.params[param]

    def set_parameter_value(self, param, comp_name, value):
        for comp in self.comps:
            if comp_name == comp.name:
                comp.params[param] = value

    def get_parameter_bounds(self, param, comp_name):
        for comp in self.comps:
            if comp_name == comp.name:
                return comp.bounds[param]

    def set_parameter_bounds(self, param, comp_name, value):
        for comp in self.comps:
            if comp_name == comp.name:
                comp.bounds[param] = value

    def get_parameters(self):
        params = {}
        for comp in self.comps:
            for key in comp.params.keys():
                if key not in params:
                    params[key] = {}
                comp_name = comp.name
                params[key][comp_name] = {}
                params[key][comp_name]['value'] = comp.params[key]
                # TODO discuss: shouldn't all params have bounds? why if?
                if key in comp.bounds:
                    params[key][comp_name]['bounds'] = comp.bounds[key]
        return params

    def save_params_to_history(self, name):
        self.history['name'] = self.get_parameters()

    def get_history(self):
        return self.history

    def generate_opt_map(self):
        opt_map = []
        ctrl_id = self.name
        for comp in self.comps:
            for key in comp.params.keys():
                entry = (ctrl_id, comp.name, key)
                opt_map.append(entry)
        return opt_map
