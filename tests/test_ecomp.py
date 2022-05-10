from pytest import approx

import openmdao.api as om
from omjl import julia_comps

from juliacall import Main as jl

jl.seval(
"""
module EComp1Test

using OpenMDAOCore

struct EComp1 <: OpenMDAOCore.AbstractExplicitComp end

function OpenMDAOCore.setup(self::EComp1)
    input_data = [VarData("x")]
    output_data = [VarData("y")]
    partials_data = [PartialsData("y", "x")]

    return input_data, output_data, partials_data
end

function OpenMDAOCore.compute!(self::EComp1, inputs, outputs)
    outputs["y"][1] = 2*inputs["x"][1]^2 + 1
    return nothing
end

function OpenMDAOCore.compute_partials!(self::EComp1, inputs, partials)
    partials["y", "x"][1] = 4*inputs["x"][1]
    return nothing
end

struct EComp2 <: OpenMDAOCore.AbstractExplicitComp
  a::Float64
end

function OpenMDAOCore.setup(self::EComp2)
    input_data = [VarData("x")]
    output_data = [VarData("y")]
    partials_data = [PartialsData("y", "x")]

    return input_data, output_data, partials_data
end

function OpenMDAOCore.compute!(self::EComp2, inputs, outputs)
    outputs["y"][1] = 2*self.a*inputs["x"][1]^2 + 1
    return nothing
end

function OpenMDAOCore.compute_partials!(self::EComp2, inputs, partials)
    partials["y", "x"][1] = 4*self.a*inputs["x"][1]
    return nothing
end

end # module
""")


def test1():
    p = om.Problem()
    ecomp = jl.EComp1Test.EComp1()
    comp = julia_comps.JuliaExplicitComp(jlcomp=ecomp)
    p.model.add_subsystem("ecomp", comp, promotes_inputs=["x"], promotes_outputs=["y"])
    p.setup()
    p.set_val("x", 3.0)
    p.run_model()
    print(p.compute_totals(of="y", wrt="x"))
    assert p.get_val("y")[0] == approx(2*p.get_val("x")[0]**2 + 1)
    assert p.compute_totals(of="y", wrt="x")["y", "x"][0,0] == approx(4*p.get_val("x")[0])


def test2():
    p = om.Problem()
    a = 3.0
    ecomp = jl.EComp1Test.EComp2(a)
    comp = julia_comps.JuliaExplicitComp(jlcomp=ecomp)
    p.model.add_subsystem("ecomp", comp, promotes_inputs=["x"], promotes_outputs=["y"])
    p.setup()
    p.set_val("x", 3.0)
    p.run_model()
    assert p.get_val("y")[0] == approx(2*a*p.get_val("x")[0]**2 + 1)
    assert p.compute_totals(of="y", wrt="x")["y", "x"][0,0] == approx(4*a*p.get_val("x")[0])
