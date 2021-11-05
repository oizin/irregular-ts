using DifferentialEquations
using CSV, DataFrames
using Distributions,Random
using ArgParse

s = ArgParseSettings()
@add_arg_table s begin
    "N"
        help = "number of examples"
        arg_type = Int
        required = true
    "seed"
        help = "random seed"
        arg_type = Int
        required = true
end

function generate_OU_1(N::Int,p=0.1)
    """
    Generate from Ornstein Uhlenbeck process

    N: number of time series
    p: probability of observing data point
    
    
    """
    # parameters
    θ=1.0
    μ=1.0
    σ=0.3

    # model
    pₓ₀ = Normal(0,2)
    f(x,p,t) = θ*(μ - x)
    g(x,p,t) = σ

    # solver
    dt = 1e-2
    maxtime = 10.0
    tspan = (0.0,10.0)

    # generate
    df_sim = DataFrame(id=Int64[],timestamp=Float64[],
                        value=Float64[],obs=Int64[])
    for i in 1:N
        x₀=rand(pₓ₀, 1)[1]
        
        prob = SDEProblem(f,g,x₀,tspan)
        sol = solve(prob,EM(),dt=dt,saveat=0:0.1:maxtime)
        df_i = DataFrame(sol)
        df_i.obs = repeat([0],size(df_i,1))
        df_i.id = repeat([i],size(df_i,1))

        # sampling times
        pₙ=Bernoulli(p)
        tobs=rand(pₙ,size(df_i,1))
        tobs[1] = 1
        tobs[size(df_i,1)] = 1
        df_i.obs[tobs] .= 1
        append!(df_sim,df_i)
    end

    return df_sim
end

parsed_args = parse_args(ARGS, s)
N = parsed_args["N"]
seed = parsed_args["seed"]
Random.seed!(seed)
data_OU_1 = generate_OU_1(Int(N*1.2))
CSV.write("./data/simulation.csv",data_OU_1)