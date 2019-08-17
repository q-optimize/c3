import uuid


class CFunc:
    """

    """
    def __init__(
            self,
            string = " ", # == key in dict
            comment = None,
            latex = " ",
            params = [],
            insts = [],
            deps = [], #other CFuncs needed for the body evaluation
            body = None,
            body_latex = " ",
            ):

        # make a random UUID which uniquely identifies/represents the component
        # https://docs.python.org/2/library/uuid.html#uuid.uuid4
        self.__uuid = uuid.uuid4()

        self.string = string # == key in dict
        self.comment = comment
        self.latex = latex
        self.params = params
        self.insts = insts
        self.deps = deps
        self.body = body
        self.body_latex = body_latex

    def get_uuid(self):
        return self.__uuid


    def set_uuid(self, uuid):
        self.__uuid = uuid


    def evaluate_full(self, t, insts = None):
        if insts == None:
            insts = self.insts

        attr = {}
        for param in self.params:
            for inst in insts:
                if param.get_uuid() == inst.param_uuid:
                    attr[param.string] = inst.value

        for dep in self.deps:
            attr.update(dep.evaluate_full(t))


        output = {}
        output[(self.string, self.get_uuid())] = {"attr" : attr}
        output[(self.string, self.get_uuid())].update(
            {"result" : self.body(t, attr)}
        )

        return output



    def evaluate(self, t, insts = None):
        return self.evaluate_full(t, insts)[(self.string, self.get_uuid)]["result"]


