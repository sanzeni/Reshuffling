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

function Mint(rp::SSNParams,μ::AbstractFloat,σ::AbstractFloat)
    if σ ≈ 0.0
        return Φ(rp,μ)
    end
    x = μ/σ
    φ = exp(-0.5x^2)/sr2π
    ψ = 0.5*(1+erf(x/sr2))
    if n ≈ -4
        Mx = -x*(x^2-3)*φ
    elseif n ≈ -3
        Mx = (x^2-1)*φ
    elseif n ≈ -2
        Mx = -x*φ
    elseif n ≈ -1
        Mx = φ
    elseif n ≈ 0
        Mx = ψ
    elseif n ≈ 1
        Mx = φ+x*ψ
    elseif n ≈ 2
        Mx = x*φ+(1+x^2)*ψ
    elseif n ≈ 3
        Mx = (2+x^2)*φ+x*(3+x^2)*ψ
    elseif n ≈ 4
        Mx = x*(5+x^2)*φ+(3+6x^2+x^4)*ψ
    elseif n ≈ 5
        Mx = (8+9x^2+x^4)*φ+x*(15+10x^2+x^4)*ψ
    elseif n ≈ 6
        Mx = x*(3+x^2)*(11+x^2)*φ+(15+45x^2+15x^4+x^6)*ψ
    elseif n ≈ 7
        Mx = (48+87x^2+20x^4+x^6)*φ+x*(105+105x^2+21x^4+x^6)*ψ
    elseif n ≈ 8
        Mx = x*(279+185x^2+27x^4+x^6)*φ+(105+420x^2+210x^4+28x^6+x^8)*ψ
    else
        return quadgk(x->exp(-0.5x^2)/sr2π*Φ(rp,μ+σ*x),-8,8,rtol=1e-8)[1]
    end
    rp.k*σ^n*Mx
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