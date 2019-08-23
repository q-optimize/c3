import uuid


# this is originally the class "Pulse"
class Component:
    """

    """
    def __init__(
            self,
            name = " ",
            desc = " ",
            comment = " ",
            params = {},
            bounds = {},
            groups = []
            ):

        # make a random UUID which uniquely identifies/represents the component
        # https://docs.python.org/2/library/uuid.html#uuid.uuid4
        self.__uuid = uuid.uuid4()

        self.name = name
        self.desc = desc
        self.params = params
        self.bounds = bounds
        self.groups = groups


    def get_uuid(self):
        return self.__uuid


    def set_uuid(self, uuid):
        self.__uuid = uuid





class ControlComponent(Component):
    """

    """
    def __init__(
            self,
            name = " ",
            desc = " ",
            comment = " ",
            params = {},
            bounds = {},
            groups = [],
            shape = None
            ):

        super().__init__(name, desc, comment, params, bounds, groups)
        self.shape = shape


    def get_shape_values(self, ts):
        return self.shape(ts, self.params)
