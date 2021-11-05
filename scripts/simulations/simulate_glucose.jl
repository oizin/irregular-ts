using Distributions, Random
using DataFrames, CSV
using ArgParse

s = ArgParseSettings()
@add_arg_table s begin
    "N"
        help = "number of examples"
        arg_type = Int
        required = true
    "error"
        help = "measurement error scaling"
        arg_type = Float64
        required = true
    "seed"
        help = "random seed"
        arg_type = Int
        required = true
end

# model
function next_obs(x,m) 
    if m > 0
        base = 3
    else
        base = 5
    end
    if x < 80
        mean = base .* exp.(-((x .- 120)./50) .^2)
    else
        mean = base .* exp.(-((x .- 120)./140) .^2)
    end
    rand(LogNormal(log(mean),0.2))
end

"""
generate_glucose_icu

k : measurement process
"""
function simulate_glucose_icu(g,k,sigma_m=0)

    ## parameters
    # fixed
    
    # random
    pθ = Normal(0.5,0.01)  # mean reversion strength of process
    pσ = Normal(20.0,2)  # variance of process
    pᵤ = Normal(140,5)  # mean of process
    p₀ = Normal(140,20)  # initial value
    pᵦ = Normal(50,5)  # insulin effect
    pm = Normal(0,sigma_m)  # measurement error scaling

    ## time period
    dt = 1e-2
    maxtime = 24.0
    n_iter = Int(maxtime / dt) + 1
    saveat = 1:Int(0.1 / dt):n_iter
    
    ## output container
    sol = DataFrame(t = 0:0.1:maxtime,
                    obs = zeros(Bool,length(saveat)),
                    x_true = zeros(length(saveat)),
                    x = zeros(length(saveat)),
                    m = zeros(length(saveat)),
                    g = zeros(length(saveat)))

    # time 0
    # random
    x₀ = rand(p₀, 1)[1]
    β = -rand(pᵦ, 1)[1]
    σ = rand(pσ, 1)[1]
    μ = rand(pᵤ, 1)[1]
    θ = rand(pθ, 1)[1]
    # fixed
    x = x₀
    x_obs = x₀  + sigma_m * randn() * log(x₀)
    iter = 1
    save_step = 1
    iter_m = 1
    obs = false

    ## loop through time steps
    while iter <= n_iter
        if (iter == iter_m)
            obs = true
            global gₜ = g(iter * 1e-2)
            global mₜ = m(x_obs)   
            Δt = k(x_obs,mₜ)
            iter_m = min(iter + Int(round(Δt/dt,digits=-1)),n_iter)
        else
            obs = false
        end

        ## process update step
        dW = rand(Normal(0,dt),1)[1]
        dx = (θ*(μ - x) + β*mₜ + gₜ)*dt + sqrt(2*θ*σ^2)*dW
        x = x + dx
        x_obs = x + sigma_m * randn() * log(x)

        if iter ∈ saveat
            sol.x_true[save_step] = x
            sol.x[save_step] = x_obs
            sol.obs[save_step] = obs
            sol.m[save_step] = mₜ
            sol.g[save_step] = gₜ
            save_step += 1
        end
        iter += 1
    end
    return sol
end


# model
function m(x)
    dose_hr = 0.0*(x > 0 && x < 140) + 3.0*(x >= 140 && x < 160) + 10.0*(x >= 160 && x < 200) + 20.0*(x >= 200)
    dose_hr / 60.0 
end
function g(t)
    rand() > 0.9 ? glucose_mg_min = 180.0 + round(randn()*10) : glucose_mg_min = 0.0
    glucose_mg_min / 50.0
end
function next_obs(x,m) 
    
    m > 0 ? base = 3 : base = 5
    
    if x < 80
        mean = base .* exp.(-((x .- 120)./50) .^2)
    else
        mean = base .* exp.(-((x .- 120)./140) .^2)
    end
    rand(LogNormal(log(mean),0.2))
end

function simulate_ensemble(N::Int,error)
    # generate
    df_sim = DataFrame(id=Int64[],
                        t=Float64[],
                        obs=Int64[],
                        x_true=Float64[],
                        x=Float64[],
                        m=Float64[],
                        g=Float64[])
    for i in 1:N
        df_i = simulate_glucose_icu(g,next_obs,error);
        df_i.id = repeat([i],size(df_i,1))
        append!(df_sim,df_i)
    end
    df_sim            
end

parsed_args = parse_args(ARGS, s)
N = parsed_args["N"]
error = parsed_args["error"]
seed = parsed_args["seed"]
Random.seed!(seed)
data_OU_1 = simulate_ensemble(Int(N*1.2),error)
CSV.write("./data/simulation.csv",data_OU_1)