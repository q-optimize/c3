import uuid
import numpy as np
import matplotlib.pyplot as plt


class Signal:
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
        sig_id = self.get_uuid()
        for cmp in self.comps:
            for key in cmp.params.keys():
                entry = (cmp.desc, sig_id, cmp.get_uuid())
                if key in opt_map.keys():
                    opt_map[key].append(entry)
                else:
                    opt_map[key] = [(entry)]

        return opt_map

