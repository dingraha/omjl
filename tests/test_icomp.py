from pytest import approx

import openmdao.api as om
from omjl import julia_comps

from juliacall import Main as jl


jl.seval(
"""
module ICompTest

using OpenMDAOCore

struct SimpleImplicit{TF} <: OpenMDAOCore.AbstractImplicitComp
    a::TF  # these would be like "options" in openmdao
end

function OpenMDAOCore.setup(::SimpleImplicit)
 
    inputs = VarData[
        VarData("x", shape=(1,), val=[2.0]),
        VarData("y", shape=(1,), val=3.0)
    ]

    outputs = VarData[
        VarData("z1", shape=1, val=[2.0]),
        VarData("z2", shape=1, val=3.0)
    ]

    partials = PartialsData[
        PartialsData("z1", "x"),            
        PartialsData("z1", "y"),            
        PartialsData("z1", "z1"),           
        PartialsData("z2", "x"),            
        PartialsData("z2", "y"),            
        PartialsData("z2", "z2")
    ]

    return inputs, outputs, partials
end

function OpenMDAOCore.apply_nonlinear!(square::SimpleImplicit, inputs, outputs, residuals)
    a = square.a
    x = inputs["x"]
    y = inputs["y"]

    @. residuals["z1"] = (a*x*x + y*y) - outputs["z1"]
    @. residuals["z2"] = (a*x + y) - outputs["z2"]
end

function OpenMDAOCore.linearize!(square::SimpleImplicit, inputs, outputs, partials)
    a = square.a
    x = inputs["x"]
    y = inputs["y"]

    @. partials["z1", "z1"] = -1.0
    @. partials["z1", "x"] = 2*a*x
    @. partials["z1", "y"] = 2*y

    @. partials["z2", "z2"] = -1.0
    @. partials["z2", "x"] = a
    @. partials["z2", "y"] = 1.0
end

end # module
""")


def test1():
    p = om.Problem()
    a = 3.0
    icomp = jl.ICompTest.SimpleImplicit(a)
    comp = julia_comps.JuliaImplicitComp(jlcomp=icomp)
    comp.linear_solver = om.DirectSolver(assemble_jac=True)
    comp.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, iprint=2, err_on_non_converge=True)
    p.model.add_subsystem("icomp", comp, promotes_inputs=["x", "y"], promotes_outputs=["z1", "z2"])
    p.setup()
    p.set_val("x", 3.0)
    p.set_val("y", 4.0)
    p.run_model()
    assert p.get_val("z1")[0] == approx(a*p.get_val("x")[0]**2 + p.get_val("y")[0]**2)
    assert p.get_val("z2")[0] == approx(a*p.get_val("x")[0] + p.get_val("y")[0])
    derivs = p.compute_totals(of=["z1", "z2"], wrt=["x", "y"])
    assert derivs["z1", "x"][0, 0] == approx(2*a*p.get_val("x")[0])
    assert derivs["z1", "y"][0, 0] == approx(2*p.get_val("y")[0])
    assert derivs["z2", "x"][0, 0] == approx(a)
    assert derivs["z2", "y"][0, 0] == approx(1)
