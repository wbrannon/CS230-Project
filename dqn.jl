using POMDPs
using Flux
using DeepQLearning
using POMDPModels
using AutomotiveDrivingModels

include("lane_change_mdp.jl")

mdp = laneChangeMDP()
model = Chain(Dense(mdp.num_features, 32, relu), Dense(32,32, relu), Dense(32,32, relu), Dense(32,32, relu), Dense(32, 32, relu), Dense(32, 32, relu), Dense(32, 32, relu), Dense(32, n_actions(mdp), sigmoid))
solver = DeepQLearningSolver(qnetwork=model, max_steps=100000, max_episode_length=2000, learning_rate=0.01, log_freq=500, recurrence=false, double_q=false, dueling=true, prioritized_replay=true)
policy = solve(solver, mdp)

env = MDPEnvironment(mdp)
DeepQLearning.evaluation(solver.evaluation_policy, 
                policy, env,                                  
                solver.num_ep_eval,
                solver.max_episode_length,
                solver.verbose)


# sim = RolloutSimulator(max_steps=30)
# sim = 
# r_tot = simulate(sim, mdp, policy)
# println("Total discounted reward for 1 simulation: $r_tot")
