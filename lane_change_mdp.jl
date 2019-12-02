using AutomotiveDrivingModels
using POMDPs
using POMDPModels
using Random
using Parameters

include("mdp_definitions.jl")
include("action_space.jl")
include("lane_change_env.jl")
include("lat_lon_driver.jl")

# will probably have to tinker with these
GOAL_LANE_REWARD = 10.
COLLISION_REWARD = -50.
WAITING_REWARD = -1. # think of better name for this one
TIMEOUT_REWARD = -10
EGO_ID = 1

# this should be where the magic happens - where the states, transitions, rewards, etc. are actually called 
@with_kw mutable struct laneChangeMDP <: MDP{Scene, Int64} # figure out what to put within the MDP type
    env::laneChangeEnvironment = laneChangeEnvironment()
    discount_factor::Float64 = 0.95
    timesteps_allowed::Int = 100 # with a timestep of 0.1, this is 10 seconds - this will decrement per time step
    terminal_state::Bool = false # this changes after we reach a terminal state (reach goal lane or crash) or we time out (timesteps_allowed reaches zero)
    collision::Bool = false # figure out a collision function
    starting_velocity::Float64 = 20.0
    timestep::Float64 = 0.1
    model::lat_lon_driver = lat_lon_driver(starting_velocity, timestep)
    # action_space::action_space = action_space()
    driver_models::Dict{Int, DriverModel} = Dict{Int, DriverModel}(EGO_ID => model)
end

# this needs to return the next state and reward
function POMDPs.gen(mdp::laneChangeMDP, s::Scene, a::Int64, rng::AbstractRNG)
    scene = deepcopy(s)
    # define actionmap function that maps an integer to an action model
    veh_idx = findfirst(EGO_ID, scene)
    action, direction = actionmap(mdp, a)
    # propagate the ego vehicle first, make sure this doesn't cause any issues
    new_ego_state = propogate(scene[vehicle_idx], action, EGO_ID, mdp.env.roadway, mdp.timestep)
    scene[EGO_ID] = Entity(new_ego_state, scene[EGO_ID].def, scene[EGO_ID].id)
    acts = Vector{LaneFollowingAccel}(undef, length(scene))

    # get the actions of all the other vehicles, this is taken from the get_actions! function in simulate.jl
    for (i, veh) in enumerate(scene)
        if i != EGO_ID
            model = mdp.driver_models[veh.id]
            observe!(model, scene, mdp.env.roadway, veh.id)
            acts[i] = rand(model)
        end
    end

    # next, propogate the scene for everyone else, this is taken from the tick! function in simulate.jl
    for i in EGO_ID+1:length(scene)
        veh = scene[i]
        new_state = propagate(veh, actions[i], mdp.env.roadway, mdp.timestep)
        scene[i] = Entity(new_state, veh.def, veh.id)
    end
    # update mdp scene
    mdp.env.scene = scene
    return (sp = scene, r=POMDPs.reward(mdp, s, a, scene))
    
end

POMDPs.discount(mdp::laneChangeMDP) = mdp.discount_factor
POMDPs.actions(mdp::laneChangeMDP) = 1:9
POMDPs.n_actions(mdp::laneChangeMDP) = 9
POMDPs.actionindex(mdp::laneChangeMDP, a::Int64) = a
# function POMDPs.n_actions(mdp::laneChangeMDP)
#     action_space_dict = get_action_space_dict(mdp.action_space, mdp.model, env.scene, env.roadway, env.ego_idx)
#     return length(collect(keys(action_space_dict))) # expecting this line to simply return the length of the dictionary
# end

# create an initial scene with all assigned behavioral models - details regarding the HVs are taken care of in lane_change_env.jl
function POMDPs.initialstate(mdp::laneChangeMDP, rng::AbstractRNG)
    mdp.env.scene, mdp.env.roadway = create_env(mdp.env.ncars, mdp.env.nlanes, mdp.env.road_length, mdp.env.roadway, mdp.env.scene)
    mdp.driver_models::Dict{Int, DriverModel} = Dict{Int, DriverModel}(EGO_ID => mdp.model)
    # assign behavioral models, for now just go with IDM - the create_env function takes care of assigning velocities randomly
    for i in EGO_ID+1:mdp.env.ncars
        mdp.driver_models[i] = IntelligentDriverModel(v_des = mdp.env.scene[i].state.v)
    end
    # not sure if I need to add a burn in period - keep this in mind
    return mdp.env.scene
end

# define reward function, mainly based on whether we reached the goal lane, there was a collision, or we are still going
function POMDPs.reward(mdp::laneChangeMDP, s::Scene, a::Int64, sp::Scene)
    # check if we collide BEFORE we check if we're in the goal lane; otherwise, we might crash to get into the goal lane
    # there is a collision_checker(scene, egoid function)
    # first, check if there is a collision
    mdp.env.collision = collision_checker(sp, EGO_ID)
    # next, get the lane that the ego vehicle is in
    ego_lane = sp[EGO_ID].state.posF.roadind.tag.lane
    r = 0.
    if mdp.env.collision
        mdp.env.terminal_state = true
        r += COLLISION_REWARD
    elseif ego_lane == mdp.env.desired_lane
        mdp.env.terminal_state = true
        r += GOAL_LANE_REWARD
    elseif mdp.timesteps_allowed == 0 # timed out - not sure if this is a good way to do this but let's give it a shot!
        mdp.env.terminal_state = true
        r += TIMEOUT_REWARD
    else
        r += WAITING_REWARD
    end
    return r
end

# the reward function changes the isterminal function, and I believe this should work just fine
POMDPs.isterminal(mdp::laneChangeMDP) = mdp.env.terminal_state

# define a function that returns a vector of features for input into the NN
# for now, define the feature vector as the x and y coordinates of each car, along with their velocities
function POMDPs.convert_s(::Type{V}, s::Scene, mdp::laneChangeMDP) where V<:AbstractArray
    env = mdp.env
    features = ones(3 + 3 * env.ncars)
    ego_idx = findfirst(EGO_ID, s)
    ego_veh = scene[ego_idx]
    features[1] = ego_veh.state.posG.x
    features[2] = ego_veh.state.posG.y
    features[3] = ego_veh.state.v
    veh_idx = EGO_ID+1
    feature_idx = 1
    while veh_idx < env.ncars
        veh = scene[veh_idx]
        features[3+feature_idx:6+feature_idx] = veh.state.posG.x, veh.state.posG.y, veh.state.v
        feature_idx += 3
        veh_idx +=1
    end
    return features
end

# not sure if I need to define the transition function - shouldn't need to since the gen and transition function are redundant
# POMDPs.transition()

# define function that takes in an integer (1-9) and returns an action
function action_map(mdp::laneChangeMDP, a::Int64)
    # get safe actions first
    all_actions = action_space()
    safe_action_space = get_action_space_dict(all_actions, mdp.model, mdp.env.scene, mdp.env.roadway, findfirst(EGO_ID, mdp.env.scene))
    # assign 1-3 to left, 4-6 to straight, and 7-9 to right
    action_string = "slow_straight"
    direction = 0
    if a == 1
        action_string = "slow_left"
        direction = -1
        act = action_space.slow_left
    elseif a == 2
        action_string = "normal_left"
        direction = -1
        act = action_space.normal_left
    elseif a == 3
        action_string = "speed_left"
        direction = -1
        act = action_space.speed_left
    elseif a == 4
        action_string = "slow_straight"
        direction = 0
        act = action_space.slow_straight
    elseif a == 5
        action_string = "normal_straight"
        direction = 0
        act = action_space.straight
    elseif a == 6
        action_string = "speed_straight"
        direction = 0
        act = action_space.speed_straight
    elseif a == 7
        action_string = "slow_right"
        direction = 1
        act = action_space.slow_right
    elseif a == 8
        action_string = "normal_right"
        direction = 1
        act = action_space.normal_right
    elseif a == 9
        action_string = "speed_right"
        direction = 1
        act = action_space.speed_right
    end
    # check if the proposed action is contained within the safe action space - if not, just return straight and it should hopefully work
    if !safe_action_space[action_string]
        direction = 0
        act = action_space.slow_straight
    end

    return act, direction
end
