


    def serialize_bounds(self, bounds_in):
        """Read in the bounds from a dictionary and store for rescaling.

        Parameters
        ----------
        bounds_in : dict
            A dictionary with the same structure as the pulse
            parametrization. Every dimension specified in the bounds will be
            optimized. Parameters present in the initial guess but not in the
            bounds are considered to be frozen.

        Returns
        -------
        list, list
            Linearized representation of the bounds and Indices in
            the linearized parameters that will be optimized.

        """
        opt_idxes = []
        bounds = []
        if self.env_shape == 'flat':
            for k in bounds_in:
                bounds.append(bounds_in[k])
                opt_idxes.append(self.idxes[k])
            bounds = np.array(bounds)
        else:
            for ctrl in sorted(bounds_in.keys()):
                for carr in sorted(bounds_in[ctrl].keys()):
                    for puls in sorted(
                        bounds_in[ctrl][carr]['pulses'].keys()
                        ):
                        params = (
                            bounds_in[ctrl][carr]['pulses'][puls]['params']
                        )
                        p_idx = (
                            self.idxes[ctrl][carr]['pulses'][puls]['params']
                            )
                        for prop in sorted(params.keys()):
                            opt_idxes.append(p_idx[prop])
                            bounds.append(params[prop])
        return bounds, opt_idxes



    def serialize_parameters(self, p, redefine=False):
        """
        Takes a nested dictionary of pulse parameters and returns a linear
        list, compatible with the parametrization of this gate. Input can
        also be the name of a stored pulse.

        Parameters
        ----------
        p : dict
            Parameters in (nested) dictionary format

        Returns
        -------
        numpy array
            Linearized parameters
        """
        q = []
        idx = 0
        idxes = {}
        for ctrl in sorted(p.keys()):
            idxes[ctrl] = {}
            for carr in sorted(p[ctrl].keys()):
                idxes[ctrl][carr] = {}
                idxes[ctrl][carr]['freq'] = p[ctrl][carr]['freq']
                idxes[ctrl][carr]['pulses'] = {}
                # TODO discuss adding target
                for puls in sorted(p[ctrl][carr]['pulses'].keys()):
                    idxes[ctrl][carr]['pulses'][puls] = {}
                    idxes[ctrl][carr]['pulses'][puls]['func']\
                        = p[ctrl][carr]['pulses'][puls]['func']
                    idxes[ctrl][carr]['pulses'][puls]['params'] = {}
                    for prop in sorted(
                            p[ctrl][carr]['pulses'][puls]['params'].keys()
                            ):
                        idxes[ctrl][carr]['pulses'][puls]['params'][prop] = idx
                        q.append(p[ctrl][carr]['pulses'][puls]['params'][prop])
                        idx += 1
        if redefine:
            self.idxes = idxes
        return np.array(q)


    def deserialize_parameters(self, q_in):
        """ Give a vector of parameters that conform to the parametrization for
        this gate and get the structured version back. Input can also be the
        name of a stored pulse.

        Parameters
        ----------
        q : array
            Numpy array containing the serialized parameters

        Returns
        -------
        type
            Description of returned object.
        """
        p = {}
        if isinstance(q_in, str):
            q_in = self.parameters[q_in]
        idxes = self.idxes
        if self.env_shape == 'flat':
            for key in idxes:
                p[key] = q_in[idxes[key]]
        else:
            for ctrl in sorted(idxes):
                p[ctrl] = {}
                for carr in sorted(idxes[ctrl]):
                    p[ctrl][carr] = {}
                    p[ctrl][carr]['pulses'] = {}
                    p[ctrl][carr]['freq'] = idxes[ctrl][carr]['freq']
                    for puls in sorted(idxes[ctrl][carr]['pulses']):
                        p[ctrl][carr]['pulses'][puls] = {
                                'params': {}
                                }
                        params = idxes[ctrl][carr]['pulses'][puls]['params']
                        for prop in sorted(params):
                            idx = params[prop]
                            p[ctrl][carr]['pulses'][puls]['params'][prop] = q[idx]
        return p


#   JSON stuff should maybe go to 'utils'
    def print_pulse(self, p):
        """Print out the pulse parameters in JSON format.

        Parameters
        ----------
        p : array
            Array of parameters in physical units.

        """
        print(
                json.dumps(
                    self.deserialize_parameters(p),
                    indent=4,
                    sort_keys=True
                    )
            )

