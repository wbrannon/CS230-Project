using AutomotiveDrivingModels
using POMDPs
using POMDPModels
using Parameters

include("mdp_definitions.jl")
include("action_space.jl")
include("lane_change_env.jl")

# this should be where the magic happens - where the states, transitions, rewards, etc. are actually called 
@with_kw mutable struct laneChangeMDP
    env::laneChangeEnvironment
    discount_factor::Float64 = 0.95
    timesteps_allowed::Int = 100 # with a timestep of 0.1, this is 10 seconds - this will decrement per time step
    terminal_state::Bool = false # this changes after we reach a terminal state (reach goal lane or crash) or we time out (timesteps_allowed reaches zero)
    collision::Bool = false # figure out a collision function
end

POMDPs.gen(mdp::laneChangeMDP)
POMDPs.reward()
POMDPs.isterminal()
POMDPs.transition()
