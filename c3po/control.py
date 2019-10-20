import uuid
import numpy as np

class GateSet:

    def __init__(self, controlset_list):
        self.controlsets = controlset_list

class ControlSet:

    def __init__(self, control_list):
        self.controls = control_list

    def get_corresponding_control_parameters(self, opt_map):
        """
        Takes a dictionary of paramaters that are supposed to be optimized
        and returns the corresponding values and bounds. Writes them together
        with a tuple (key, id) to a dict and returns them

        Parameters
        ----------

        opt_map : dict
            Dictionary of parameters that are supposed to be optimized, and
            corresponding component identifier used to map
            opt_params = {
                    'T_up' : [1,2],
                    'T_down' : [1,2],
                    'freq' : [3]
                }


        Returns
        -------

        opt_params : dict
            Dictionary with values, bounds lists, plus a list with pairs of
            shape (key, id) at the corresponding pos in the list as the
            values and bounds to be able to identify the origin of the values,
            bounds in the other lists.

            Example:

            opt_params = {
                'values': [0,           0,           0,             0,             0],
                'bounds': [[0, 0],      [0, 0],      [0, 0],        [0, 0],        [0.0]],
                'origin': [('T_up', 1), ('T_up', 2), ('T_down', 1), ('T_down', 2), ('freq', 3)]
                }

        """

        opt_params = {}
        opt_params['values'] = []
        opt_params['bounds'] = []
        opt_params['origin'] = [] # array that holds tuple of (key, id) to be
                                  # identify each entry in the above lists
                                  # with it's corresponding entry

        for key in opt_map:
            for id_pair in opt_map[key]:
                control_uuid = id_pair[0]
                for control in self.controls:
                    if control_uuid == control.get_uuid():
                        comp_uuid = id_pair[1]
                        val = control.get_parameter_value(key, comp_uuid)
                        bounds = control.get_parameter_bounds(key, comp_uuid)

                        opt_params['values'].append(val)
                        opt_params['bounds'].append(bounds)
                        opt_params['origin'].append((key, id_pair))
        return opt_params


    def set_corresponding_control_parameters(self, opt_params):
        """
            sets the values in opt_params in the original control class
        """
        for i in range(len(opt_params['origin'])):
            key = opt_params['origin'][i][0]
            id_pair = opt_params['origin'][i][1]

            control_uuid = id_pair[0]
            comp_uuid = id_pair[1]

            for control in self.controls:
                val = opt_params['values'][i]
                bounds = opt_params['bounds'][i]

                control.set_parameter_value(key, comp_uuid, val)
                control.set_parameter_bounds(key, comp_uuid, bounds)


    def get_values_bounds(self, opt_params):
        values = opt_params['values']
        bounds = opt_params['bounds']
        return values, bounds


    def update_controls(self, values, opt_params):
        opt_params['values'] = values
        self.set_corresponding_control_parameters(opt_params)

    def save_params_to_history(self, name):
        for control in self.controls:
            control.save_params_to_history(name)

    def get_history(self, opt_params):
        #TODO
        return None

class Control(C3obj):
    """

    """
    def __init__(
            self,
            name = " ",
            desc = " ",
            comment = " ",
            t_start = None,
            t_end = None,
            comps = []
            ):
        super().__init__(
            name=name,
            desc=desc,
            comment=comment
            )

        # make a random UUID which uniquely identifies/represents the component
        # https://docs.python.org/2/library/uuid.html#uuid.uuid4
        self.__uuid = uuid.uuid4()


        self.name = name
        self.desc = desc
        self.comment = comment

        self.t_start = t_start
        self.t_end = t_end


        self.comps = comps

        self.history = []



    def get_uuid(self):
        return self.__uuid


    def set_uuid(self, uuid):
        self.__uuid = uuid


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
        sig_id = self.name
        for cmp in self.comps:
            for key in cmp.params.keys():
                entry = (sig_id, cmp.name)
                if key in opt_map.keys():
                    opt_map[key].append(entry)
                else:
                    opt_map[key] = [(entry)]
        return opt_map
