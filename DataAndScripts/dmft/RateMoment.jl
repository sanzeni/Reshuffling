abstract type RateParams end

struct RicciardiParams <: RateParams
    θ::AbstractFloat
    Vr::AbstractFloat
    σn::AbstractFloat
    τs::AbstractFloat
    τr::AbstractFloat
end

function μtox(μ::AbstractFloat)::AbstractFloat
    sign(μ/100-0.2)*abs(μ/100-0.2)^0.5
end

function xtoμ(x::AbstractFloat)::AbstractFloat
    100(sign(x)*abs(x)^2.0+0.2)
end

function Φint(rp::RicciardiParams,μ::AbstractFloat)
    umax = (rp.θ-μ)/rp.σn
    umin = (rp.Vr-μ)/rp.σn
    if umin > 10
        return umax*exp(-umax^2)/rp.τs
    elseif umin > -4
        return 1/(rp.τr+rp.τs*(0.5*π*(erfi(umax)-erfi(umin)) +
                umax^2*pFq([1.0,1.0],[1.5,2.0],umax^2) -
                umin^2*pFq([1.0,1.0],[1.5,2.0],umin^2)))
    else
        return 1/(rp.τr+rp.τs*(log(abs(umin))-log(abs(umax)) +
                (0.25umin^-2-0.1875umin^-4+0.3125umin^-6-
                    0.8203125umin^-8+2.953125umin^-10) -
                (0.25umax^-2-0.1875umax^-4+0.3125umax^-6-
                    0.8203125umax^-8+2.953125umax^-10)))
    end
end

function Φitp(rps::Vector{T}) where {T<:RateParams}
    xs = range(μtox(-1E3), μtox(5E5), length=2*10^5+1)
    Φint(rps[1],-100.0)
    Φint(rps[1],0.0)
    Φint(rps[1],100.0)
    global Φitps = Dict()
    for rp in rps
        Φs = [Φint(rp,xtoμ(x)) for x in xs]
        Φitps[hash(rp)] = CubicSplineInterpolation(xs, Φs, extrapolation_bc = Line())
    end
end

function Φ(rp::RateParams,μ::AbstractFloat)
    return Φitps[hash(rp)](μtox(μ))
end

function dΦ(rp::RateParams,μ::AbstractFloat)
    dμ = 0.01
    return (Φitps[hash(rp)](μtox(μ+dμ))-Φitps[hash(rp)](μtox(μ-dμ)))/(2dμ)
end

function Mint(rp::RateParams,μ::AbstractFloat,σ::AbstractFloat)
    if σ ≈ 0.0
        return Φ(rp,μ)
    end
    quadgk(x->exp(-0.5x^2)/sr2π*Φ(rp,μ+σ*x),-8,8,rtol=1e-8)[1]
end

function Mitp(rps::Vector{T}) where {T<:RateParams}
    xs = range(μtox(-3000.0), μtox(3000.0), length=3000+1)
    σs = range(0.0, 600.0, length=1200+1)
    Mint(rps[1],0.0,1.0)
    global Mitps = Dict()
    for rp in rps
        Ms = [Mint(rp,xtoμ(x),σ) for x in xs, σ in σs]
        Mitps[hash(rp)] = CubicSplineInterpolation((xs,σs), Ms, extrapolation_bc = Line())
    end
end

function M(rp::RateParams,μ::AbstractFloat,σ::AbstractFloat)
    return Mitps[hash(rp)](μtox(μ),σ)
end

function Cint(rp::RateParams,μ1::AbstractFloat,μ2::AbstractFloat,
            σ1::AbstractFloat,σ2::AbstractFloat,ρ::AbstractFloat)
    if ρ ≈ 0.0
        return M(rp,μ1,σ1)*M(rp,μ2,σ2)
    end
    if σ1 ≈ 0.0 || σ2 ≈ 0.0
        c = 0.0
    else
        c = sign(ρ)*min(abs(ρ)/(σ1*σ2),1)
    end
    quadgk(x->exp(-0.5x^2)/sr2π*
        M(rp,μ1+sign(c)*σ1*√(abs(c))*x,σ1*√(1-abs(c)))*
        M(rp,μ2+σ2*√(abs(c))*x,σ2*√(1-abs(c))),-8,8,rtol=1e-8)[1]
end

function Cint(rp::RateParams,μ::AbstractFloat,σ::AbstractFloat,ρ::AbstractFloat)
    if ρ ≈ 0.0
        return M(rp,μ,σ)^2
    end
    if σ ≈ 0.0
        c = 0.0
    else
        c = sign(ρ)*min(abs(ρ)/σ^2,1)
    end
    quadgk(x->exp(-0.5x^2)/sr2π*
        M(rp,μ+sign(c)*σ*√(abs(c))*x,σ*√(1-abs(c)))*
        M(rp,μ+σ*√(abs(c))*x,σ*√(1-abs(c))),-8,8,rtol=1e-8)[1]
end

function Citp(rps::Vector{T}) where {T<:RateParams}
    xs = range(μtox(-1000.0), μtox(1000.0), length=400+1)
    σs = range(0.0, 600.0, length=900+1)
    cs = range(0.2, 1.0, length=48+1)
    Cint(rps[1],0.0,0.0,1.0,1.0,0.0)
    global Citps = Dict()
    for rp in rps
        Cs = [Cint(rp,xtoμ(x),σ,σ^2*c) for x in xs, σ in σs, c in cs]
        Citps[hash(rp)] = CubicSplineInterpolation((xs,σs,cs), Cs,
            extrapolation_bc = Line())
    end
end

function C(rp::RateParams,μ::AbstractFloat,σ::AbstractFloat,ρ::AbstractFloat)
    if σ ≈ 0.0
        c = 0.0
    else
        c = sign(ρ)*min(abs(ρ)/σ^2,1)
    end
    return Citps[hash(rp)](μtox(μ),σ,c)
end

abstract type OptoParams end

struct TruncNormOptoParams <: OptoParams
    f::AbstractFloat                # fraction potentially expressing opsin
    f̄::AbstractFloat                # true fraction unaffected by opsin
    s::AbstractFloat                # normalized std of opsin expression
    λL::AbstractFloat               # mean optogenetic stimulus voltage
end

function TruncNormOptoParams(f::AbstractFloat,s::AbstractFloat,λL::AbstractFloat)
    f̄ = 1-f*0.5*(1+erf(1/(s*sr2)))
    TruncNormOptoParams(f,f̄,s,λL)
end

struct LogNormOptoParams <: OptoParams
    s::AbstractFloat                # normalized std of opsin expression
    λL::AbstractFloat               # mean optogenetic stimulus voltage
    μ::AbstractFloat                # mean of log opsin expression
    σ::AbstractFloat                # std of log opsin expression
end

function LogNormOptoParams(s::AbstractFloat,λL::AbstractFloat)
    σ2 = log(1+s^2)
    σ = √(σ2)
    μ = log(λL)-0.5σ2
    LogNormOptoParams(s,λL,μ,σ)
end

function ΦLint(op::TruncNormOptoParams,rp::RateParams,μ::AbstractFloat)
    int = quadgk(x->exp(-0.5((x-1)/op.s)^2)/(sr2π*op.s)*op.f*Φ(rp,μ+x*op.λL),
                0,10op.s,rtol=1e-8)[1]
    int += op.f̄*Φ(rp,μ)
    return int
end

function ΦLint(op::LogNormOptoParams,rp::RateParams,μ::AbstractFloat)
    quadgk(x->exp(-0.5((log(x)-op.μ)/op.σ)^2)/(sr2π*op.σ*x)*Φ(rp,μ+x),
        0,150op.λL,rtol=1e-8)[1]
end

function ΦLitp(op::OptoParams,rps::Vector{T}) where {T <: RateParams}
    xs = range(μtox(-1E3), μtox(1E5), length=10^5+1)
    ΦLint(op,rps[1],0.0)
    global ΦLitps = Dict()
    for rp in rps
        ΦLs = [ΦLint(op,rp,xtoμ(x)) for x in xs]
        ΦLitps[hash(rp)] = CubicSplineInterpolation(xs, ΦLs, extrapolation_bc = Line())
    end
end

function ΦL(op::OptoParams,rp::RateParams,μ::AbstractFloat)
    return ΦLitps[hash(rp)](μtox(μ))
end

function MLint(op::TruncNormOptoParams,rp::RateParams,μ::AbstractFloat,σ::AbstractFloat)
    if σ ≈ 0.0
        return ΦL(op,rp,μ)
    end
    int = quadgk(x->exp(-0.5((x-1)/op.s)^2)/(sr2π*op.s)*op.f*M(rp,μ+x*op.λL,σ),
                0,10op.s,rtol=1e-8)[1]
    int += op.f̄*M(rp,μ,σ)
    return int
end

function MLint(op::LogNormOptoParams,rp::RateParams,μ::AbstractFloat,σ::AbstractFloat)
    if σ ≈ 0.0
        return ΦL(op,rp,μ)
    end
    quadgk(x->exp(-0.5((log(x)-op.μ)/op.σ)^2)/(sr2π*op.σ*x)*M(rp,μ+x,σ),
        0,150op.λL,rtol=1e-8)[1]
end

function MLitp(op::OptoParams,rps::Vector{T}) where {T <: RateParams}
    xs = range(μtox(-3000.0), μtox(1000.0), length=2000+1)
    σs = range(0.0, 600.0, length=1200+1)
    MLint(op,rps[1],0.0,1.0)
    global MLitps = Dict()
    for rp in rps
        MLs = [MLint(op,rp,xtoμ(x),σ) for x in xs, σ in σs]
        MLitps[hash(rp)] = CubicSplineInterpolation((xs,σs), MLs,
            extrapolation_bc = Line())
    end
end

function ML(op::OptoParams,rp::RateParams,μ::AbstractFloat,σ::AbstractFloat)
    return MLitps[hash(rp)](μtox(μ),σ)
end

function CLint(op::TruncNormOptoParams,rp::RateParams,μ1::AbstractFloat,μ2::AbstractFloat,
        σ1::AbstractFloat,σ2::AbstractFloat,ρ::AbstractFloat)
    int = quadgk(x->exp(-0.5((x-1)/op.s)^2)/(sr2π*op.s)*op.f*
                Cint(rp,μ1+x*op.λL,μ2+x*op.λL,σ1,σ2,ρ),
                0,10op.s,rtol=1e-8)[1]
    int += op.f̄*Cint(rp,μ1,μ2,σ1,σ2,ρ)
    return int
end

function CLint(op::LogNormOptoParams,rp::RateParams,μ1::AbstractFloat,μ2::AbstractFloat,
        σ1::AbstractFloat,σ2::AbstractFloat,ρ::AbstractFloat)
    if ρ ≈ 0.0
        return quadgk(x->exp(-0.5((log(x)-op.μ)/op.σ)^2)/(sr2π*op.σ*x)*
            M(rp,μ1+x,σ1)*M(rp,μ2+x,σ2),0,150op.λL,rtol=1e-8)[1]
    end
    quadgk(x->exp(-0.5((log(x)-op.μ)/op.σ)^2)/(sr2π*op.σ*x)*
                Cint(rp,μ1+x,μ2+x,σ1,σ2,ρ),0,150op.λL,rtol=1e-8)[1]
end

function CLint(op::TruncNormOptoParams,rp::RateParams,μ::AbstractFloat,σ::AbstractFloat,
        ρ::AbstractFloat)
    if ρ ≈ 0.0
        int = quadgk(x->exp(-0.5((x-1)/op.s)^2)/(sr2π*op.s)*op.f*M(rp,μ+x*op.λL,σ)^2,
                    0,10op.s,rtol=1e-8)[1]
    else
        int = quadgk(x->exp(-0.5((x-1)/op.s)^2)/(sr2π*op.s)*op.f*C(rp,μ+x*op.λL,σ,ρ),
                    0,10op.s,rtol=1e-8)[1]
    end
    int += op.f̄*C(rp,μ,σ,ρ)
    return int
end

function CLint(op::LogNormOptoParams,rp::RateParams,μ::AbstractFloat,σ::AbstractFloat,
        ρ::AbstractFloat)
    if ρ ≈ 0.0
        return quadgk(x->exp(-0.5((log(x)-op.μ)/op.σ)^2)/(sr2π*op.σ*x)*M(rp,μ+x,σ)^2,
                    0,150op.λL,rtol=1e-8)[1]
    end
    quadgk(x->exp(-0.5((log(x)-op.μ)/op.σ)^2)/(sr2π*op.σ*x)*
                C(rp,μ+x,σ,ρ),0,50op.λL,rtol=1e-8)[1]
end

function CLitp(op::OptoParams,rps::Vector{T}) where {T <: RateParams}
    xs = range(μtox(-1000.0), μtox(800.0), length=300+1)
    σs = range(0.0, 600.0, length=900+1)
    cs = range(0.2, 1.0, length=48+1)
    CLint(op,rps[1],0.0,1.0,0.0)
    global CLitps = Dict()
    for rp in rps
        CLs = [CLint(op,rp,xtoμ(x),σ,σ^2*c) for x in xs, σ in σs, c in cs]
        CLitps[hash(rp)] = CubicSplineInterpolation((xs,σs,cs), CLs,
            extrapolation_bc = Line())
    end
end

function CL(op::OptoParams,rp::RateParams,μ::AbstractFloat,σ::AbstractFloat,
        ρ::AbstractFloat)
    if σ ≈ 0.0
        c = 0.0
    else
        c = sign(ρ)*min(abs(ρ)/σ^2,1)
    end
    return CLitps[hash(rp)](μtox(μ),σ,c)
end

function RLint(op::TruncNormOptoParams,rp::RateParams,μ1::AbstractFloat,μ2::AbstractFloat,
        σ1::AbstractFloat,σ2::AbstractFloat,ρ::AbstractFloat)
    if σ1 ≈ 0.0 || σ2 ≈ 0.0
        c = 0.0
    else
        c = sign(ρ)*min(abs(ρ)/(σ1*σ2),1)
    end
    int = quadgk(x->exp(-0.5x^2)/sr2π*
        M(rp,μ1+sign(c)*σ1*√(abs(c))*x,σ1*√(1-abs(c)))*
        ML(op,rp,μ2+σ2*√(abs(c))*x,σ2*√(1-abs(c))),-8,8,rtol=1e-8)[1]
    return int
end

function RLint(op::LogNormOptoParams,rp::RateParams,μ1::AbstractFloat,μ2::AbstractFloat,
        σ1::AbstractFloat,σ2::AbstractFloat,ρ::AbstractFloat)
    if σ1 ≈ 0.0 || σ2 ≈ 0.0
        c = 0.0
    else
        c = sign(ρ)*min(abs(ρ)/(σ1*σ2),1)
    end
    int = quadgk(x->exp(-0.5x^2)/sr2π*
        M(rp,μ1+sign(c)*σ1*√(abs(c))*x,σ1*√(1-abs(c)))*
        ML(op,rp,μ2+σ2*√(abs(c))*x,σ2*√(1-abs(c))),-8,8,rtol=1e-8)[1]
    return int
end

function RL(op::OptoParams,rp::RateParams,μ1::AbstractFloat,μ2::AbstractFloat,
        σ1::AbstractFloat,σ2::AbstractFloat,ρ::AbstractFloat)
    return RLint(op,rp,μ1,μ2,σ1,σ2,ρ)
end