from types import MethodType

import openmdao.api as om

# The PythonCall/JuliaCall docs say I should create a new module to avoid polluting Main, but then the seval command below doesn't work.
# import juliacall; jl = juliacall.newmodule("OMJL")
import juliacall; jl = juliacall.Main

# This imports the Julia package OpenMDAOCore:
jl.seval("using OpenMDAOCore: OpenMDAOCore")


def _initialize_common(self):
    self.options.declare('jlcomp')


def _setup_common(self):
    self._jlcomp = self.options['jlcomp']
    input_data, output_data, partials_data = jl.OpenMDAOCore.setup(self._jlcomp)

    for var in input_data:
        self.add_input(var.name, shape=var.shape, val=var.val,
                       units=var.units)

    for var in output_data:
        self.add_output(var.name, shape=var.shape, val=var.val,
                        units=var.units)

    for data in partials_data:
        self.declare_partials(data.of, data.wrt,
                              rows=data.rows, cols=data.cols,
                              val=data.val)


class JuliaExplicitComp(om.ExplicitComponent):

    def initialize(self):
        _initialize_common(self)

    def setup(self):
        _setup_common(self)

    def compute(self, inputs, outputs):
        inputs_dict = juliacall.convert(jl.Dict, dict(inputs))
        outputs_dict = juliacall.convert(jl.Dict, dict(outputs))

        jl.OpenMDAOCore.compute_b(self._jlcomp, inputs_dict, outputs_dict)

    def compute_partials(self, inputs, partials):
        inputs_dict = juliacall.convert(jl.Dict, dict(inputs))

        partials_dict = {}
        for of_wrt in self._declared_partials:
            partials_dict[of_wrt] = partials[of_wrt]
        partials_dict = juliacall.convert(jl.Dict, partials_dict)

        jl.OpenMDAOCore.compute_partials_b(self._jlcomp, inputs_dict, partials_dict)


class JuliaImplicitComp(om.ImplicitComponent):

    def initialize(self):
        _initialize_common(self)

    def setup(self):
        _setup_common(self)
        #self._julia_apply_nonlinear = get_py2jl_apply_nonlinear(self._jlcomp)
        #self._julia_linearize = get_py2jl_linearize(self._jlcomp)
        #self._julia_guess_nonlinear = get_py2jl_guess_nonlinear(self._jlcomp)
        #self._julia_solve_nonlinear = get_py2jl_solve_nonlinear(self._jlcomp)
        #self._julia_apply_linear = get_py2jl_apply_linear(self._jlcomp)
        ## Trying to avoid the exception
        ##
        ##     RuntimeError: AssembledJacobian not supported for matrix-free subcomponent.
        ##
        ## when the Julia component doesn't implement apply_linear!.
        #if self._julia_apply_linear:
        #    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        #        inputs_dict = dict(inputs)
        #        outputs_dict = dict(outputs)
        #        d_inputs_dict = dict(d_inputs)
        #        d_outputs_dict = dict(d_outputs)
        #        d_residuals_dict = dict(d_residuals)

        #        self._julia_apply_linear(self._jlcomp,
        #                                 inputs_dict, outputs_dict,
        #                                 d_inputs_dict, d_outputs_dict,
        #                                 d_residuals_dict, mode)
        #    # https://www.ianlewis.org/en/dynamically-adding-method-classes-or-class-instanc
        #    self.apply_linear = MethodType(apply_linear, self)

    def apply_nonlinear(self, inputs, outputs, residuals):
        inputs_dict = juliacall.convert(jl.Dict, dict(inputs))
        outputs_dict = juliacall.convert(jl.Dict, dict(outputs))
        residuals_dict = juliacall.convert(jl.Dict, dict(residuals))

        jl.OpenMDAOCore.apply_nonlinear_b(self._jlcomp, inputs_dict, outputs_dict, residuals_dict)

    def linearize(self, inputs, outputs, partials):
        inputs_dict = juliacall.convert(jl.Dict, dict(inputs))
        outputs_dict = juliacall.convert(jl.Dict, dict(outputs))

        partials_dict = {}
        for of_wrt in self._declared_partials:
            partials_dict[of_wrt] = partials[of_wrt]
        partials_dict = juliacall.convert(jl.Dict, partials_dict)

        jl.OpenMDAOCore.linearize_b(self._jlcomp, inputs_dict, outputs_dict, partials_dict)

    def guess_nonlinear(self, inputs, outputs, residuals):
        pass
        # if self._julia_guess_nonlinear:
        #     inputs_dict = dict(inputs)
        #     outputs_dict = dict(outputs)
        #     residuals_dict = dict(residuals)

        #     self._julia_guess_nonlinear(self._jlcomp, inputs_dict, outputs_dict,
        #                                 residuals_dict)

    def solve_nonlinear(self, inputs, outputs):
        pass
        # if self._julia_solve_nonlinear:
        #     inputs_dict = dict(inputs)
        #     outputs_dict = dict(outputs)

        #     self._julia_solve_nonlinear(self._jlcomp, inputs_dict, outputs_dict)

