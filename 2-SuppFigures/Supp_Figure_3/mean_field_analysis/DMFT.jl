struct NetworkParams
    K::Int                          # mean number of E->E connections
    p::AbstractFloat                # connection probability
    γ::AbstractFloat                # ratio of I vs E cells
    sX::AbstractFloat               # std/mean of external firing rates
    rpE::RateParams                 # rate params for excitatory cells
    rpI::RateParams                 # rate params for inhibitory cells
    W::Matrix{AbstractFloat}        # O(1) recurrent coupling matrix (EIX -> EI)
end

function NetworkParams(K::Int,p::AbstractFloat,γ::AbstractFloat,gE::AbstractFloat,
        gI::AbstractFloat,β::AbstractFloat,ΓE::AbstractFloat,ΓI::AbstractFloat,
        sX::AbstractFloat,rpE::RateParams,rpI::RateParams)
    W = [1.0 -gE (ΓI*γ*gE-ΓE);
        1/β -gI/β (ΓI*γ*gI-ΓE)/β]
    NetworkParams(K,p,γ,sX,rpE,rpI,W)
end

function dmft(np::NetworkParams,J::AbstractFloat,rX::AbstractFloat,Twrm::AbstractFloat,
        Tsave::AbstractFloat,dt::AbstractFloat;DE::Function=t->0.0,DI::Function=t->0.0,
        rEinit::AbstractFloat=1E-8,rIinit::AbstractFloat=1E-8,
        CrEinit::Vector{Float64}=[1E2],CrIinit::Vector{Float64}=[1E2])
    Nint = round(Int,(Twrm+Tsave)/dt)+1
    Nclc = round(Int,1.5Tsave/dt)+1
    
    rE = Array{Float64}(undef,Nint)
    rI = Array{Float64}(undef,Nint)
    CrE = Array{Float64}(undef,(Nint,Nint))
    CrI = Array{Float64}(undef,(Nint,Nint))

    rE[1] = rEinit
    rI[1] = rIinit
    Nσ2init = length(CrEinit)
    if Nclc >= Nσ2init
        CrE[1,1:Nσ2init] .= CrEinit
        CrE[1,Nσ2init+1:Nclc] .= CrEinit[end]
        CrE[1:Nσ2init,1] .= CrEinit
        CrE[Nσ2init+1:Nclc,1] .= CrEinit[end]
    else
        CrE[1,1:Nclc] .= CrEinit[1:Nclc]
        CrE[1:Nclc,1] .= CrEinit[1:Nclc]
    end
    Nσ2init = length(CrIinit)
    if Nclc >= Nσ2init
        CrI[1,1:Nσ2init] .= CrIinit
        CrI[1,Nσ2init+1:Nclc] .= CrIinit[end]
        CrI[1:Nσ2init,1] .= CrIinit
        CrI[Nσ2init+1:Nclc,1] .= CrIinit[end]
    else
        CrI[1,1:Nclc] .= CrIinit[1:Nclc]
        CrI[1:Nclc,1] .= CrIinit[1:Nclc]
    end
    
    τE = np.rpE.τs
    τI = np.rpI.τs
    τE2 = τE^2
    τI2 = τI^2
    τEinv = 1/τE
    τIinv = 1/τI
    τEinv2 = τEinv^2
    τIinv2 = τIinv^2
    dtτEinv = dt*τEinv
    dtτIinv = dt*τIinv
    dtτEinv2 = dtτEinv^2
    dtτIinv2 = dtτIinv^2
    dt2 = dt^2
    
    q = 1-np.p
    
    JK = J*np.K
    τEJK = τE*JK
    τIJK = τI*JK
    J2K = J^2*np.K
    τE2J2K = τE2*J2K
    τI2J2K = τI2*J2K
    W2 = np.W.^2
    rX2 = rX^2
    σrX2 = (np.sX*rX)^2

    for i in 1:Nint-1
        rE2 = rE[i]^2
        rI2 = rI[i]^2
        σrE20 = CrE[i,i]-rE2
        σrI20 = CrI[i,i]-rI2
        μE = τEJK*(np.W[1,3]*rX+np.W[1,1]*rE[i]+np.γ*np.W[1,2]*rI[i])
        μI = τIJK*(np.W[2,3]*rX+np.W[2,1]*rE[i]+np.γ*np.W[2,2]*rI[i])
        σμE20 = τE2J2K*(q*(W2[1,3]*rX2+W2[1,1]*rE2+np.γ*W2[1,2]*rI2) +
                        W2[1,3]*σrX2+W2[1,1]*σrE20+np.γ*W2[1,2]*σrI20)
        σμI20 = τI2J2K*(q*(W2[2,3]*rX2+W2[2,1]*rE2+np.γ*W2[2,2]*rI2) +
                        W2[2,3]*σrX2+W2[2,1]*σrE20+np.γ*W2[2,2]*σrI20)
        kE1 = -τEinv*rE[i]+τEinv*M(rpE,μE,√(σμE20))
        kI1 = -τIinv*rI[i]+τIinv*M(rpI,μI,√(σμI20))
        σrE20k = max(CrE[i,i]-(rE[i]+0.5dt*kE1)^2,0)
        σrI20k = max(CrI[i,i]-(rI[i]+0.5dt*kI1)^2,0)
        μEk = τEJK*(np.W[1,3]*rX+np.W[1,1]*(rE[i]+0.5dt*kE1)+
                np.γ*np.W[1,2]*(rI[i]+0.5dt*kI1))
        μIk = τIJK*(np.W[2,3]*rX+np.W[2,1]*(rE[i]+0.5dt*kE1)+
                np.γ*np.W[2,2]*(rI[i]+0.5dt*kI1))
        σμE20k = τE2J2K*(q*(W2[1,3]*rX2+W2[1,1]*(rE[i]+0.5dt*kE1)^2+
                            np.γ*W2[1,2]*(rI[i]+0.5dt*kI1)^2) +
                        W2[1,3]*σrX2+W2[1,1]*σrE20k+np.γ*W2[1,2]*σrI20k)
        σμI20k = τI2J2K*(q*(W2[2,3]*rX2+W2[2,1]*(rE[i]+0.5dt*kE1)^2+
                            np.γ*W2[2,2]*(rI[i]+0.5dt*kI1)^2) +
                        W2[2,3]*σrX2+W2[2,1]*σrE20k+np.γ*W2[2,2]*σrI20k)
        kE2 = -τEinv*(rE[i]+0.5dt*kE1)+τEinv*M(rpE,μEk,√(σμE20k))
        kI2 = -τIinv*(rI[i]+0.5dt*kI1)+τIinv*M(rpI,μIk,√(σμI20k))
        σrE20k = max(CrE[i,i]-(rE[i]+0.5dt*kE2)^2,0)
        σrI20k = max(CrI[i,i]-(rI[i]+0.5dt*kI2)^2,0)
        μEk = τEJK*(np.W[1,3]*rX+np.W[1,1]*(rE[i]+0.5dt*kE2)+
                np.γ*np.W[1,2]*(rI[i]+0.5dt*kI2))
        μIk = τIJK*(np.W[2,3]*rX+np.W[2,1]*(rE[i]+0.5dt*kE2)+
                np.γ*np.W[2,2]*(rI[i]+0.5dt*kI2))
        σμE20k = τE2J2K*(q*(W2[1,3]*rX2+W2[1,1]*(rE[i]+0.5dt*kE2)^2+
                            np.γ*W2[1,2]*(rI[i]+0.5dt*kI2)^2) +
                        W2[1,3]*σrX2+W2[1,1]*σrE20k+np.γ*W2[1,2]*σrI20k)
        σμI20k = τI2J2K*(q*(W2[2,3]*rX2+W2[2,1]*(rE[i]+0.5dt*kE2)^2+
                            np.γ*W2[2,2]*(rI[i]+0.5dt*kI2)^2) +
                        W2[2,3]*σrX2+W2[2,1]*σrE20k+np.γ*W2[2,2]*σrI20k)
        kE3 = -τEinv*(rE[i]+0.5dt*kE2)+τEinv*M(rpE,μEk,√(σμE20k))
        kI3 = -τIinv*(rI[i]+0.5dt*kI2)+τIinv*M(rpI,μIk,√(σμI20k))
        σrE20k = max(CrE[i,i]-(rE[i]+dt*kE3)^2,0)
        σrI20k = max(CrI[i,i]-(rI[i]+dt*kI3)^2,0)
        μEk = τEJK*(np.W[1,3]*rX+np.W[1,1]*(rE[i]+dt*kE3)+
                np.γ*np.W[1,2]*(rI[i]+dt*kI3))
        μIk = τIJK*(np.W[2,3]*rX+np.W[2,1]*(rE[i]+dt*kE3)+
                np.γ*np.W[2,2]*(rI[i]+dt*kI3))
        σμE20k = τE2J2K*(q*(W2[1,3]*rX2+W2[1,1]*(rE[i]+dt*kE3)^2+
                            np.γ*W2[1,2]*(rI[i]+dt*kI3)^2) +
                        W2[1,3]*σrX2+W2[1,1]*σrE20k+np.γ*W2[1,2]*σrI20k)
        σμI20k = τI2J2K*(q*(W2[2,3]*rX2+W2[2,1]*(rE[i]+dt*kE3)^2+
                            np.γ*W2[2,2]*(rI[i]+dt*kI3)^2) +
                        W2[2,3]*σrX2+W2[2,1]*σrE20k+np.γ*W2[2,2]*σrI20k)
        kE4 = -τEinv*(rE[i]+dt*kE3)+τEinv*M(rpE,μEk,√(σμE20k))
        kI4 = -τIinv*(rI[i]+dt*kI3)+τIinv*M(rpI,μIk,√(σμI20k))
        rE[i+1] = rE[i] + dt/6*(kE1+2kE2+2kE3+kE4)
        rI[i+1] = rI[i] + dt/6*(kI1+2kI2+2kI3+kI4)
        if abs(rE[i+1]) > 1E10 || isnan(rE[i+1])
            @printf "CrE[%d,%d] = %f\n" i i CrE[i,i]
            @printf "CrI[%d,%d] = %f\n" i i CrI[i,i]
            @printf "rE[%d+1] = %f\n" i rE[i+1]
            @printf "rI[%d+1] = %f\n" i rI[i+1]
            @printf "μE = %f\n" μE
            @printf "μI = %f\n" μI
            @printf "σμE20 = %f\n" σμE20
            @printf "σμI20 = %f\n" σμI20
            println("system diverged")
            return (rE,rI,CrE,CrI)
        end
        if i > Nclc
            CrE[i+1,i-Nclc] = CrE[i,i-Nclc]
            CrI[i+1,i-Nclc] = CrI[i,i-Nclc]
        end
        for j in max(1,i-Nclc):i
            σrE2ij = CrE[i,j]-rE2
            σrI2ij = CrI[i,j]-rI2
            σμE2ij = τE2J2K*(q*(W2[1,3]*rX2+W2[1,1]*rE2+np.γ*W2[1,2]*rI2) +
                            W2[1,3]*σrX2+W2[1,1]*σrE2ij+np.γ*W2[1,2]*σrI2ij)
            σμI2ij = τI2J2K*(q*(W2[2,3]*rX2+W2[2,1]*rE2+np.γ*W2[2,2]*rI2) +
                            W2[2,3]*σrX2+W2[2,1]*σrE2ij+np.γ*W2[2,2]*σrI2ij)
            CrE[i+1,j+1] = CrE[i,j+1]+CrE[i+1,j]-CrE[i,j] -
                            dtτEinv*(CrE[i+1,j]+CrE[i,j+1]-2CrE[i,j]) - dtτEinv2*CrE[i,j] +
                            dtτEinv2*C(rpE,μE,√(σμE20),σμE2ij) + dtτEinv2*DE((j-i)*dt)
            CrI[i+1,j+1] = CrI[i,j+1]+CrI[i+1,j]-CrI[i,j] -
                            dtτIinv*(CrI[i+1,j]+CrI[i,j+1]-2CrI[i,j]) - dtτIinv2*CrI[i,j] +
                            dtτIinv2*C(rpI,μI,√(σμI20),σμI2ij) + dtτIinv2*DI((j-i)*dt)
            if CrE[i+1,j+1] > 1E10 || isnan(CrE[i+1,j+1])
                @printf "rE[%d] = %f\n" i rE[i]
                @printf "rI[%d] = %f\n" i rI[i]
                @printf "CrE[%d,%d] = %f\n" i j CrE[i,j]
                @printf "CrI[%d,%d] = %f\n" i j CrI[i,j]
                @printf "CrE[%d+1,%d] = %f\n" i j CrE[i+1,j]
                @printf "CrI[%d+1,%d] = %f\n" i j CrI[i+1,j]
                @printf "CrE[%d,%d+1] = %f\n" i j CrE[i,j+1]
                @printf "CrI[%d,%d+1] = %f\n" i j CrI[i,j+1]
                @printf "μE = %f\n" μE
                @printf "μI = %f\n" μI
                @printf "σμE20 = %f\n" σμE20
                @printf "σμI20 = %f\n" σμI20
                @printf "σμE2ij = %f\n" σμE2ij
                @printf "σμI2ij = %f\n" σμI2ij
                @printf "CrE[%d+1,%d+1] = %f\n" i j CrE[i+1,j+1]
                @printf "CrI[%d+1,%d+1] = %f\n" i j CrI[i+1,j+1]
                println("system diverged")
                return (rE,rI,CrE,CrI)
            end
            CrE[j+1,i+1] = CrE[i+1,j+1]
            CrI[j+1,i+1] = CrI[i+1,j+1]
        end
        ndiv = 5
        if (ndiv*i) % (Nint-1) == 0
#             @printf "%3d%% completed\n" round(Int,100*i/(Nint-1))
        end
    end
    
    Nsave = round(Int,(Tsave)/dt)+1
    return (rE[end-Nsave+1],rI[end-Nsave+1],
        CrE[end-Nsave+1,end-Nsave+1:end],CrI[end-Nsave+1,end-Nsave+1:end],
        (maximum(diag(CrE)[end-Nsave+1:end])-minimum(diag(CrE)[end-Nsave+1:end]))/
        mean(diag(CrE)[end-Nsave+1:end]) < 1E-6)
end

function optodmft(op::OptoParams,np::NetworkParams,J::AbstractFloat,rX::AbstractFloat,
        Twrm::AbstractFloat,Tsave::AbstractFloat,dt::AbstractFloat,rE::AbstractFloat,
        rI::AbstractFloat,CrE::Vector{Float64},CrI::Vector{Float64};
        ΔrEinit::AbstractFloat=rE/2,ΔrIinit::AbstractFloat=rI/2,
        CΔrEinit::Vector{Float64}=CrE,CΔrIinit::Vector{Float64}=CrI,
        RrEΔrEinit::Vector{Float64}=-CrE/2,RrIΔrIinit::Vector{Float64}=-CrI/2)
    Nint = round(Int,(Twrm+Tsave)/dt)+1
    Nclc = round(Int,1.5Tsave/dt)+1
    
    ΔrE = Array{Float64}(undef,Nint)
    ΔrI = Array{Float64}(undef,Nint)
    CΔrE = Array{Float64}(undef,(Nint,Nint))
    CΔrI = Array{Float64}(undef,(Nint,Nint))
    RrEΔrE = Array{Float64}(undef,(Nint,Nint))
    RrIΔrI = Array{Float64}(undef,(Nint,Nint))

    ΔrE[1] = ΔrEinit
    ΔrI[1] = ΔrIinit
    Nσ2init = length(CΔrEinit)
    if Nclc >= Nσ2init
        CΔrE[1,1:Nσ2init] .= CΔrEinit
        CΔrE[1,Nσ2init+1:Nclc] .= CΔrEinit[end]
        CΔrE[1:Nσ2init,1] .= CΔrEinit
        CΔrE[Nσ2init+1:Nclc,1] .= CΔrEinit[end]
    else
        CΔrE[1,1:Nclc] .= CΔrEinit[1:Nclc]
        CΔrE[1:Nclc,1] .= CΔrEinit[1:Nclc]
    end
    Nσ2init = length(CΔrIinit)
    if Nclc >= Nσ2init
        CΔrI[1,1:Nσ2init] .= CΔrIinit
        CΔrI[1,Nσ2init+1:Nclc] .= CΔrIinit[end]
        CΔrI[1:Nσ2init,1] .= CΔrIinit
        CΔrI[Nσ2init+1:Nclc,1] .= CΔrIinit[end]
    else
        CΔrI[1,1:Nclc] .= CΔrIinit[1:Nclc]
        CΔrI[1:Nclc,1] .= CΔrIinit[1:Nclc]
    end
    NRrIΔrInit = length(RrEΔrEinit)
    if Nclc >= NRrIΔrInit
        RrEΔrE[1,1:NRrIΔrInit] .= RrEΔrEinit
        RrEΔrE[1,NRrIΔrInit+1:Nclc] .= RrEΔrEinit[end]
        RrEΔrE[1:NRrIΔrInit,1] .= RrEΔrEinit
        RrEΔrE[NRrIΔrInit+1:Nclc,1] .= RrEΔrEinit[end]
    else
        RrEΔrE[1,1:Nclc] .= RrEΔrEinit[1:Nclc]
        RrEΔrE[1:Nclc,1] .= RrEΔrEinit[1:Nclc]
    end
    NRrIΔrInit = length(RrIΔrIinit)
    if Nclc >= NRrIΔrInit
        RrIΔrI[1,1:NRrIΔrInit] .= RrIΔrIinit
        RrIΔrI[1,NRrIΔrInit+1:Nclc] .= RrIΔrIinit[end]
        RrIΔrI[1:NRrIΔrInit,1] .= RrIΔrIinit
        RrIΔrI[NRrIΔrInit+1:Nclc,1] .= RrIΔrIinit[end]
    else
        RrIΔrI[1,1:Nclc] .= RrIΔrIinit[1:Nclc]
        RrIΔrI[1:Nclc,1] .= RrIΔrIinit[1:Nclc]
    end
    
    τE = np.rpE.τs
    τI = np.rpI.τs
    τE2 = τE^2
    τI2 = τI^2
    τEinv = 1/τE
    τIinv = 1/τI
    τEinv2 = τEinv^2
    τIinv2 = τIinv^2
    dtτEinv = dt*τEinv
    dtτIinv = dt*τIinv
    dtτEinv2 = dtτEinv^2
    dtτIinv2 = dtτIinv^2
    dt2 = dt^2
    
    q = 1-np.p
    
    JK = J*np.K
    τEJK = τE*JK
    τIJK = τI*JK
    J2K = J^2*np.K
    τE2J2K = τE2*J2K
    τI2J2K = τI2*J2K
    W2 = np.W.^2
    rX2 = rX^2
    σrX2 = (np.sX*rX)^2
    
    rE2 = rE^2
    rI2 = rI^2
    σrE2 = CrE.-rE2
    σrI2 = CrI.-rI2
    μE = τEJK*(np.W[1,3]*rX+np.W[1,1]*rE+np.γ*np.W[1,2]*rI)
    μI = τIJK*(np.W[2,3]*rX+np.W[2,1]*rE+np.γ*np.W[2,2]*rI)
    σμE2 = τE2J2K*(q*(W2[1,3]*rX2+W2[1,1]*rE2+np.γ*W2[1,2]*rI2) .+
                    W2[1,3]*σrX2.+W2[1,1]*σrE2.+np.γ*W2[1,2]*σrI2)
    σμI2 = τI2J2K*(q*(W2[2,3]*rX2+W2[2,1]*rE2+np.γ*W2[2,2]*rI2) .+
                    W2[2,3]*σrX2.+W2[2,1]*σrE2.+np.γ*W2[2,2]*σrI2)
    
    Nσ2 = length(σμE2)
    σμE20 = σμE2[1]
    σμI20 = σμI2[1]
    CrE0 = CrE[1]
    CrI0 = CrI[1]

    for i in 1:Nint-1
        ΔrE2 = ΔrE[i]^2
        ΔrI2 = ΔrI[i]^2
        rEΔrE = rE*ΔrE[i]
        rIΔrI = rI*ΔrI[i]
        σΔrE20 = CΔrE[i,i] - ΔrE2
        σΔrI20 = CΔrI[i,i] - ΔrI2
        ρrEΔrE0 = RrEΔrE[i,i] - rEΔrE
        ρrIΔrI0 = RrIΔrI[i,i] - rIΔrI
        ΔμE = τEJK*(np.W[1,1]*ΔrE[i]+np.γ*np.W[1,2]*ΔrI[i])
        ΔμI = τIJK*(np.W[2,1]*ΔrE[i]+np.γ*np.W[2,2]*ΔrI[i])
        σΔμE20 = τE2J2K*(q*(W2[1,1]*ΔrE2+np.γ*W2[1,2]*ΔrI2) +
                        W2[1,1]*σΔrE20+np.γ*W2[1,2]*σΔrI20)
        σΔμI20 = τI2J2K*(q*(W2[2,1]*ΔrE2+np.γ*W2[2,2]*ΔrI2) +
                        W2[2,1]*σΔrE20+np.γ*W2[2,2]*σΔrI20)
        ρμEΔμE0 = τE2J2K*(q*(W2[1,1]*rEΔrE+np.γ*W2[1,2]*rIΔrI) +
                        W2[1,1]*ρrEΔrE0+np.γ*W2[1,2]*ρrIΔrI0)
        ρμIΔμI0 = τI2J2K*(q*(W2[2,1]*rEΔrE+np.γ*W2[2,2]*rIΔrI) +
                        W2[2,1]*ρrEΔrE0+np.γ*W2[2,2]*ρrIΔrI0)
        σμpΔμE20 = max(σμE20+σΔμE20+2ρμEΔμE0,0)
        σμpΔμI20 = max(σμI20+σΔμI20+2ρμIΔμI0,0)
        kE1 = -τEinv*ΔrE[i]+τEinv*(ML(op,np.rpE,μE+ΔμE,√(σμpΔμE20))-rE)
        kI1 = -τIinv*ΔrI[i]+τIinv*(M(np.rpI,μI+ΔμI,√(σμpΔμI20))-rI)
        σΔrE20k = max(CΔrE[i,i] - (ΔrE[i]+0.5dt*kE1)^2,0)
        σΔrI20k = max(CΔrI[i,i] - (ΔrI[i]+0.5dt*kI1)^2,0)
        ρrEΔrE0k = RrEΔrE[i,i] - rE*(ΔrE[i]+0.5dt*kE1)
        ρrIΔrI0k = RrIΔrI[i,i] - rI*(ΔrI[i]+0.5dt*kI1)
        ΔμEk = τEJK*(np.W[1,1]*(ΔrE[i]+0.5dt*kE1)+np.γ*np.W[1,2]*(ΔrI[i]+0.5dt*kI1))
        ΔμIk = τIJK*(np.W[2,1]*(ΔrE[i]+0.5dt*kE1)+np.γ*np.W[2,2]*(ΔrI[i]+0.5dt*kI1))
        σΔμE20k = τE2J2K*(q*(W2[1,1]*(ΔrE[i]+0.5dt*kE1)^2+
                            np.γ*W2[1,2]*(ΔrI[i]+0.5dt*kI1)^2) +
                        W2[1,1]*σΔrE20k+np.γ*W2[1,2]*σΔrI20k)
        σΔμI20k = τI2J2K*(q*(W2[2,1]*(ΔrE[i]+0.5dt*kE1)^2+
                            np.γ*W2[2,2]*(ΔrI[i]+0.5dt*kI1)^2) +
                        W2[2,1]*σΔrE20k+np.γ*W2[2,2]*σΔrI20k)
        ρμEΔμE0k = τE2J2K*(q*(W2[1,1]*rE*(ΔrE[i]+0.5dt*kE1)+
                            np.γ*W2[1,2]*rI*(ΔrI[i]+0.5dt*kI1)) +
                        W2[1,1]*ρrEΔrE0k+np.γ*W2[1,2]*ρrIΔrI0k)
        ρμIΔμI0k = τI2J2K*(q*(W2[2,1]*rE*(ΔrE[i]+0.5dt*kE1)+
                            np.γ*W2[2,2]*rI*(ΔrI[i]+0.5dt*kI1)) +
                        W2[2,1]*ρrEΔrE0k+np.γ*W2[2,2]*ρrIΔrI0k)
        σμpΔμE20k = max(σμE20+σΔμE20k+2ρμEΔμE0k,0)
        σμpΔμI20k = max(σμI20+σΔμI20k+2ρμIΔμI0k,0)
        kE2 = -τEinv*(ΔrE[i]+0.5dt*kE1)+τEinv*(ML(op,np.rpE,μE+ΔμEk,√(σμpΔμE20k))-rE)
        kI2 = -τIinv*(ΔrI[i]+0.5dt*kI1)+τIinv*(M(np.rpI,μI+ΔμIk,√(σμpΔμI20k))-rI)
        σΔrE20k = max(CΔrE[i,i] - (ΔrE[i]+0.5dt*kE2)^2,0)
        σΔrI20k = max(CΔrI[i,i] - (ΔrI[i]+0.5dt*kI2)^2,0)
        ρrEΔrE0k = RrEΔrE[i,i] - rE*(ΔrE[i]+0.5dt*kE2)
        ρrIΔrI0k = RrIΔrI[i,i] - rI*(ΔrI[i]+0.5dt*kI2)
        ΔμEk = τEJK*(np.W[1,1]*(ΔrE[i]+0.5dt*kE2)+np.γ*np.W[1,2]*(ΔrI[i]+0.5dt*kI2))
        ΔμIk = τIJK*(np.W[2,1]*(ΔrE[i]+0.5dt*kE2)+np.γ*np.W[2,2]*(ΔrI[i]+0.5dt*kI2))
        σΔμE20k = τE2J2K*(q*(W2[1,1]*(ΔrE[i]+0.5dt*kE2)^2+
                            np.γ*W2[1,2]*(ΔrI[i]+0.5dt*kI2)^2) +
                        W2[1,1]*σΔrE20k+np.γ*W2[1,2]*σΔrI20k)
        σΔμI20k = τI2J2K*(q*(W2[2,1]*(ΔrE[i]+0.5dt*kE2)^2+
                            np.γ*W2[2,2]*(ΔrI[i]+0.5dt*kI2)^2) +
                        W2[2,1]*σΔrE20k+np.γ*W2[2,2]*σΔrI20k)
        ρμEΔμE0k = τE2J2K*(q*(W2[1,1]*rE*(ΔrE[i]+0.5dt*kE2)+
                            np.γ*W2[1,2]*rI*(ΔrI[i]+0.5dt*kI2)) +
                        W2[1,1]*ρrEΔrE0k+np.γ*W2[1,2]*ρrIΔrI0k)
        ρμIΔμI0k = τI2J2K*(q*(W2[2,1]*rE*(ΔrE[i]+0.5dt*kE2)+
                            np.γ*W2[2,2]*rI*(ΔrI[i]+0.5dt*kI2)) +
                        W2[2,1]*ρrEΔrE0k+np.γ*W2[2,2]*ρrIΔrI0k)
        σμpΔμE20k = max(σμE20+σΔμE20k+2ρμEΔμE0k,0)
        σμpΔμI20k = max(σμI20+σΔμI20k+2ρμIΔμI0k,0)
        kE3 = -τEinv*(ΔrE[i]+0.5dt*kE2)+τEinv*(ML(op,np.rpE,μE+ΔμEk,√(σμpΔμE20k))-rE)
        kI3 = -τIinv*(ΔrI[i]+0.5dt*kI2)+τIinv*(M(np.rpI,μI+ΔμIk,√(σμpΔμI20k))-rI)
        σΔrE20k = max(CΔrE[i,i] - (ΔrE[i]+dt*kE3)^2,0)
        σΔrI20k = max(CΔrI[i,i] - (ΔrI[i]+dt*kI3)^2,0)
        ρrEΔrE0k = RrEΔrE[i,i] - rE*(ΔrE[i]+dt*kE3)
        ρrIΔrI0k = RrIΔrI[i,i] - rI*(ΔrI[i]+dt*kI3)
        ΔμEk = τEJK*(np.W[1,1]*(ΔrE[i]+dt*kE3)+np.γ*np.W[1,2]*(ΔrI[i]+dt*kI3))
        ΔμIk = τIJK*(np.W[2,1]*(ΔrE[i]+dt*kE3)+np.γ*np.W[2,2]*(ΔrI[i]+dt*kI3))
        σΔμE20k = τE2J2K*(q*(W2[1,1]*(ΔrE[i]+dt*kE3)^2+
                            np.γ*W2[1,2]*(ΔrI[i]+dt*kI3)^2) +
                        W2[1,1]*σΔrE20k+np.γ*W2[1,2]*σΔrI20k)
        σΔμI20k = τI2J2K*(q*(W2[2,1]*(ΔrE[i]+dt*kE3)^2+
                            np.γ*W2[2,2]*(ΔrI[i]+dt*kI3)^2) +
                        W2[2,1]*σΔrE20k+np.γ*W2[2,2]*σΔrI20k)
        ρμEΔμE0k = τE2J2K*(q*(W2[1,1]*rE*(ΔrE[i]+dt*kE3)+
                            np.γ*W2[1,2]*rI*(ΔrI[i]+dt*kI3)) +
                        W2[1,1]*ρrEΔrE0k+np.γ*W2[1,2]*ρrIΔrI0k)
        ρμIΔμI0k = τI2J2K*(q*(W2[2,1]*rE*(ΔrE[i]+dt*kE3)+
                            np.γ*W2[2,2]*rI*(ΔrI[i]+dt*kI3)) +
                        W2[2,1]*ρrEΔrE0k+np.γ*W2[2,2]*ρrIΔrI0k)
        σμpΔμE20k = max(σμE20+σΔμE20k+2ρμEΔμE0k,0)
        σμpΔμI20k = max(σμI20+σΔμI20k+2ρμIΔμI0k,0)
        kE4 = -τEinv*(ΔrE[i]+dt*kE3)+τEinv*(ML(op,np.rpE,μE+ΔμEk,√(σμpΔμE20k))-rE)
        kI4 = -τIinv*(ΔrI[i]+dt*kI3)+τIinv*(M(np.rpI,μI+ΔμIk,√(σμpΔμI20k))-rI)
        ΔrE[i+1] = ΔrE[i] + dt/6*(kE1+2kE2+2kE3+kE4)
        ΔrI[i+1] = ΔrI[i] + dt/6*(kI1+2kI2+2kI3+kI4)
        if abs(ΔrE[i+1]) > 1E10 || isnan(ΔrE[i+1])
            @printf "CΔrE[%d,%d] = %f\n" i i CΔrE[i,i]
            @printf "CΔrI[%d,%d] = %f\n" i i CΔrI[i,i]
            @printf "ΔrE[%d+1] = %f\n" i ΔrE[i+1]
            @printf "ΔrI[%d+1] = %f\n" i ΔrI[i+1]
            @printf "ΔμE = %f\n" ΔμE
            @printf "ΔμI = %f\n" ΔμI
            @printf "σμpΔμE20 = %f\n" σμpΔμE20
            @printf "σμpΔμI20 = %f\n" σμpΔμI20
            println("system diverged")
            return (ΔrE,ΔrI,CΔrE,CΔrI)
        end
        if i > Nclc
            CΔrE[i+1,i-Nclc] = CΔrE[i,i-Nclc]
            CΔrI[i+1,i-Nclc] = CΔrI[i,i-Nclc]
            RrEΔrE[i+1,i-Nclc] = RrEΔrE[i,i-Nclc]
            RrIΔrI[i+1,i-Nclc] = RrIΔrI[i,i-Nclc]
        end
        for j in max(1,i-Nclc):i
            CrEij = abs(j-i) < Nσ2 ? CrE[abs(j-i)+1] : CrE[end]
            CrIij = abs(j-i) < Nσ2 ? CrI[abs(j-i)+1] : CrI[end]
            σΔrE2ij = CΔrE[i,j] - ΔrE2
            σΔrI2ij = CΔrI[i,j] - ΔrI2
            ρrEΔrEij = RrEΔrE[i,j] - rEΔrE
            ρrIΔrIij = RrIΔrI[i,j] - rIΔrI
            σμE2ij = abs(j-i) < Nσ2 ? σμE2[abs(j-i)+1] : σμE2[end]
            σμI2ij = abs(j-i) < Nσ2 ? σμI2[abs(j-i)+1] : σμI2[end]
            σΔμE2ij = τE2J2K*(q*(W2[1,1]*ΔrE2+np.γ*W2[1,2]*ΔrI2) +
                            W2[1,1]*σΔrE2ij+np.γ*W2[1,2]*σΔrI2ij)
            σΔμI2ij = τI2J2K*(q*(W2[2,1]*ΔrE2+np.γ*W2[2,2]*ΔrI2) +
                            W2[2,1]*σΔrE2ij+np.γ*W2[2,2]*σΔrI2ij)
            ρμEΔμEij = τE2J2K*(q*(W2[1,1]*rEΔrE+np.γ*W2[1,2]*rIΔrI) +
                            W2[1,1]*ρrEΔrEij+np.γ*W2[1,2]*ρrIΔrIij)
            ρμIΔμIij = τI2J2K*(q*(W2[2,1]*rEΔrE+np.γ*W2[2,2]*rIΔrI) +
                            W2[2,1]*ρrEΔrEij+np.γ*W2[2,2]*ρrIΔrIij)
            σμpΔμE2ij = σμE2ij+σΔμE2ij+2ρμEΔμEij
            σμpΔμI2ij = σμI2ij+σΔμI2ij+2ρμIΔμIij
            RrErEL = RL(op,np.rpE,μE,μE+ΔμE,√(σμE20),√(σμpΔμE20),σμE2ij+ρμEΔμEij)
            RrIrIL = Cint(np.rpI,μI,μI+ΔμI,√(σμI20),√(σμpΔμI20),σμI2ij+ρμIΔμIij)
            CΔrE[i+1,j+1] = CΔrE[i,j+1]+CΔrE[i+1,j]-CΔrE[i,j] -
                            dtτEinv*(CΔrE[i+1,j]+CΔrE[i,j+1]-2CΔrE[i,j]) -
                            dtτEinv2*CΔrE[i,j] +
                            dtτEinv2*(CL(op,np.rpE,μE+ΔμE,√(σμpΔμE20),σμpΔμE2ij)-
                                2RrErEL+CrEij)
            CΔrI[i+1,j+1] = CΔrI[i,j+1]+CΔrI[i+1,j]-CΔrI[i,j] -
                            dtτIinv*(CΔrI[i+1,j]+CΔrI[i,j+1]-2CΔrI[i,j]) -
                            dtτIinv2*CΔrI[i,j] +
                            dtτIinv2*(C(np.rpI,μI+ΔμI,√(σμpΔμI20),σμpΔμI2ij)-
                                2RrIrIL+CrIij)
            if CΔrE[i+1,j+1] > 1E10 || isnan(CΔrE[i+1,j+1])
                @printf "ΔrE[%d] = %f\n" i ΔrE[i]
                @printf "ΔrI[%d] = %f\n" i ΔrI[i]
                @printf "CΔrE[%d,%d] = %f\n" i i CΔrE[i,i]
                @printf "CΔrI[%d,%d] = %f\n" i i CΔrI[i,i]
                @printf "σμE2ij = %f\n" σμE2ij
                @printf "σμI2ij = %f\n" σμI2ij
                @printf "CΔrE[%d+1,%d+1] = %f\n" i j CΔrE[i+1,j+1]
                @printf "CΔrI[%d+1,%d+1] = %f\n" i j CΔrI[i+1,j+1]
                println("system diverged")
                return (ΔrE,ΔrI,CΔrE,CΔrI)
            end
            CΔrE[j+1,i+1] = CΔrE[i+1,j+1]
            CΔrI[j+1,i+1] = CΔrI[i+1,j+1]
            RrEΔrE[i+1,j+1] = RrEΔrE[i,j+1]+RrEΔrE[i+1,j]-RrEΔrE[i,j] -
                            dtτEinv*(RrEΔrE[i+1,j]+RrEΔrE[i,j+1]-2RrEΔrE[i,j]) -
                            dtτEinv2*RrEΔrE[i,j] + dtτEinv2*(RrErEL-CrEij)
            RrIΔrI[i+1,j+1] = RrIΔrI[i,j+1]+RrIΔrI[i+1,j]-RrIΔrI[i,j] -
                            dtτIinv*(RrIΔrI[i+1,j]+RrIΔrI[i,j+1]-2RrIΔrI[i,j]) -
                            dtτIinv2*RrIΔrI[i,j] + dtτIinv2*(RrIrIL-CrIij)
            RrEΔrE[j+1,i+1] = RrEΔrE[i+1,j+1]
            RrIΔrI[j+1,i+1] = RrIΔrI[i+1,j+1]
        end
        ndiv = 5
        if (ndiv*i) % (Nint-1) == 0
#             @printf "%3d%% completed\n" round(Int,100*i/(Nint-1))
        end
    end
    
    Nsave = round(Int,(Tsave)/dt)+1
    return (ΔrE[end-Nsave+1],ΔrI[end-Nsave+1],
        CΔrE[end-Nsave+1,end-Nsave+1:end],CΔrI[end-Nsave+1,end-Nsave+1:end],
        RrEΔrE[end-Nsave+1,end-Nsave+1:end],RrIΔrI[end-Nsave+1,end-Nsave+1:end],
        (maximum(diag(CΔrE)[end-Nsave+1:end])-minimum(diag(CΔrE)[end-Nsave+1:end]))/
        mean(diag(CΔrE)[end-Nsave+1:end]) < 1E-6)
end