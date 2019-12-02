using POMDPs
using Flux
using DeepQLearning
using POMDPModels

include("lane_change_mdp.jl")

mdp = laneChangeMDP()
model = Chain(Dense(18, 32), Dense(32, n_actions(mdp)))
solver = DeepQLearningSolver(qnetwork=model, max_steps=10000, learning_rate=0.005, log_freq=500, recurrence=false, double_q=true, dueling=true, prioritized_replay=true)
policy = solve(solver, mdp)

sim = RolloutSimulator(max_steps=30)
r_tot = simulate(sim, mdp, policy)
println("Total discounted reward for 1 simulation: $r_tot")