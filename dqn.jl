using POMDPs
using Flux
using DeepQLearning
using POMDPModels
using AutomotiveDrivingModels
using BSON: @load, @save

include("lane_change_mdp.jl")

mdp = laneChangeMDP()
model = Chain(Dense(mdp.num_features, 32, relu), Dense(32,32, relu), Dense(32,32, relu), Dense(32,32, relu), Dense(32, 32, relu), Dense(32, 32, relu), Dense(32, 32, relu), Dense(32,32, relu), Dense(32,32, relu), Dense(32, n_actions(mdp)))
solver = DeepQLearningSolver(qnetwork=model, max_steps=1000000, eval_freq = 10000, max_episode_length=200, learning_rate=0.0005, log_freq=1000, target_update_freq = 10000, recurrence=false, double_q=false, dueling=true, prioritized_replay=true)
policy = solve(solver, mdp)

@save "policy1.bson" policy


# env = MDPEnvironment(mdp)
# DeepQLearning.evaluation(solver.evaluation_policy, 
#                 policy, env,                                  
#                 solver.num_ep_eval,
#                 solver.max_episode_length,
#                 solver.verbose)               

# sim = RolloutSimulator(max_steps=30)
# sim = 
# r_tot = simulate(sim, mdp, policy)
# println("Total discounted reward for 1 simulation: $r_tot")
