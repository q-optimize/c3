import matplotlib.pyplot as plt



class Pulse:
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


    # define an internal id for the created instance of the pulse object
    # as private attributes are not per se available in python use 'hack'
    # see: https://stackoverflow.com/questions/1641219/does-python-have-private-variables-in-classes
    __id = 0

    def __init__(
            self,
            shape = None,
            parameters = {},
            bounds = {}
            ):

        self.__id = Pulse.__id + 1
        Pulse.__id = self.__id
        self.shape = shape
        self.parameters = parameters
        self.bounds = bounds

    def get_id(self):
        return self.__id


    def get_shape_values(self, tlist):
        return shape(tlist, self.parameters)



