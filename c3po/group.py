import uuid




class Group:
    """

    """
    def __init__(
            self,
            name = " ",
            desc = " ",
            comment = " ",
            elements = []
            ):



        # make a random UUID which uniquely identifies/represents the component
        # https://docs.python.org/2/library/uuid.html#uuid.uuid4
        self.__uuid = uuid.uuid4()


        self.name = name
        self.desc = desc
        self.comment = comment
        self.elements = elements



    def get_uuid(self):
        return self.__uuid


    def set_uuid(self, uuid):
        self.__uuid = uuid


    def add_element(self, ele):
        self.elements.append(ele)


    def remove_element(self, ele):
        self.elements.remove(ele)


    def get_element(self, uuid):
        for ele in self.elements:
            if uuid == ele.get_uuid():
                return ele



class ComponentGroup(Group):
    def __init__(
            self,
            name = " ",
            desc = " ",
            comment = " ",
            elements = []
            ):

        super().__init__(name, desc, comment, elements)




    def get_parameter_value(self, key, uuid):
        for ele in self.elements:
            if uuid == ele.get_uuid():
                return ele.params[key]


    def set_parameter_value(self, key, uuid, val):
        for ele in self.elements:
            if uuid == ele.get_uuid():
                ele.params[key] = val



    def get_parameter_bounds(self, key, uuid):
        for ele in self.elements:
            if uuid == ele.get_uuid():
                return ele.bounds[key]


    def set_parameter_bounds(self, key, uuid, bounds):
        for ele in self.elements:
            if uuid == ele.get_uuid():
                ele.bounds[key] = bounds


    def get_parameters(self):
        params = {}

        for ele in self.elements:
            for key in ele.params.keys():
                if key not in params:
                    params[key] = {}

                uuid = ele.get_uuid()
                params[key][uuid] = {}

                params[key][uuid]['value'] = ele.params[key]
                if key in ele.bounds:
                    params[key][uuid]['bounds'] = ele.bounds[key]

        return params


    def set_parameter_values_in_elements(self, key, val, uuids = []):
        for ele in self.elements:
            if uuids != []:
                for uuid in uuids:
                    if uuid == ele.get_uuid():
                        ele.params[key] = val
            else:
                ele.params[key] = val

