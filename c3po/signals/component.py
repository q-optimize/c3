import uuid


# this is originally the class "Pulse"
class Component:
    """Represents a pulse.

    Parameters
    ----------
    shape : func
        Function handle to function specifying the exact shape of the pulse
    parameters : dict
        dictionary of the parameters needed for the shape-function to
        create the desired pulse
    bounds : dict
        boundaries of the parameters, i.e. technical limits of experimental
        setup or physical boundaries. needed for optimizer

    """
    def __init__(
            self,
            desc = None,
            shape = None,
            params = {},
            bounds = {}
            ):

        # make a random UUID which uniquely identifies/represents the component
        # https://docs.python.org/2/library/uuid.html#uuid.uuid4
        self.__uuid = uuid.uuid4()

        self.desc = desc
        self.shape = shape
        self.params = params
        self.bounds = bounds
#        self.SIunit = 


    def get_uuid(self):
        return self.__uuid


    def set_uuid(self, uuid):
        self.__uuid = uuid


    def get_shape_values(self, ts):
        return self.shape(ts, self.params)
