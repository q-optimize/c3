import uuid



class Instance:
    def __init__(
            self,
            value = None,
            unit = None,
            bounds = [],
            param_uuid = None
            ):


        self.value = value
        self.unit = unit
        self.bounds = bounds
        self.param_uuid = param_uuid



class Parameter:
    """

    """
    def __init__(
            self,
            string = " ", # == key in dict
            comment = None,
            latex = " "
            ):

        # make a random UUID which uniquely identifies/represents the component
        # https://docs.python.org/2/library/uuid.html#uuid.uuid4
        self.__uuid = uuid.uuid4()

        self.string = string # == key in dict
        self.comment = comment
        self.latex = latex


    def get_uuid(self):
        return self.__uuid


    def set_uuid(self, uuid):
        self.__uuid = uuid


    def get_value(self, t, depends = None):
        if depends == None:
            depends = self.depends
        return self.shape(t, depends)




