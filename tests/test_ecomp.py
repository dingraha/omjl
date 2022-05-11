import time

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
    @show 4*inputs["x"]
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

struct EComp3 <: OpenMDAOCore.AbstractExplicitComp
    a::Vector{Float64}
end

EComp3(n::Integer) = EComp3(range(1.0, n; length=n) |> collect)

function OpenMDAOCore.setup(self::EComp3)
    input_data = [VarData("x")]
    output_data = [VarData("y")]
    partials_data = [PartialsData("y", "x")]

    return input_data, output_data, partials_data
end

function OpenMDAOCore.compute!(self::EComp3, inputs, outputs)
    outputs["y"][1] = 2*self.a[1]*inputs["x"][1]^2 + 1
    return nothing
end

function OpenMDAOCore.compute_partials!(self::EComp3, inputs, partials)
    partials["y", "x"][1] = 4*self.a[1]*inputs["x"][1]
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
    p.final_setup()
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

def test3():
    times_d = {}
    for n in [10, 10000000]:
        # Run once to get rid of the Julia JIT time.
        p = om.Problem()
        ecomp = jl.EComp1Test.EComp3(n)
        comp = julia_comps.JuliaExplicitComp(jlcomp=ecomp)
        p.model.add_subsystem("ecomp", comp, promotes_inputs=["x"], promotes_outputs=["y"])
        p.setup()
        p.set_val("x", 3.0)
        p.run_model()
        assert p.get_val("y")[0] == approx(2*p.get_val("x")[0]**2 + 1)
        assert p.compute_totals(of="y", wrt="x")["y", "x"][0,0] == approx(4*p.get_val("x")[0])

        print(f"n = {n}")
        times = []
        samples = 50
        for _ in range(samples):
            p = om.Problem()
            ecomp = jl.EComp1Test.EComp3(n)
            comp = julia_comps.JuliaExplicitComp(jlcomp=ecomp)
            p.model.add_subsystem("ecomp", comp, promotes_inputs=["x"], promotes_outputs=["y"])
            p.setup()
            p.set_val("x", 3.0)
            t = time.time()
            p.run_model()
            assert p.get_val("y")[0] == approx(2*p.get_val("x")[0]**2 + 1)
            assert p.compute_totals(of="y", wrt="x")["y", "x"][0,0] == approx(4*p.get_val("x")[0])
            times.append(time.time() - t)
            print(f"time = {time.time() - t}")

        times_d[n] = sum(times)/samples

    assert times_d[10] == approx(times_d[10000000], rel=1e-1)
