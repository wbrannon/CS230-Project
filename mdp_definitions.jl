using AutomotiveDrivingModels
include("lat_lon_driver.jl")
include("action_space.jl")
include("lane_change_env.jl")

MAX_LONG_ACCEL = 4. # m/s^2
MAX_LAT_ACCEL = 0.4 # m/s^2
MAX_LAT_VEL = 4. # m/s

# for now, don't account for partial observability
# need to define custom state and custom state transition function

# FOR POMDPS.jl, NEED TO DEFINE THE FOLLOWING:
# STATE S: AgentState definition
# STATE TRANSITION S': propagate function
# ACTION SPACE A: action_space.jl - will likely have to figure out how to transfer dictionary to a useful format for POMDPs.jl
# OBSERVATION O: < might be implicitly defined within ADM.jl > - THIS IS ONLY NEEDED FOR POMDP, WE ARE STARTING WITH MDP
# REWARD R: reward_fn
# DISCOUNT FACTOR γ: discount_factor in mdp 
# state is defined by vehicle state (x, y, z) and lateral and longitudinal accelerations
mutable struct AgentState
    state::VehicleState
    # long_accel::Float64
    # lat_accel::Float64
    # side_slip::Float64 # use this to keep track of velocities and theta
end

function AutomotiveDrivingModels.propagate(vehicle::Entity{VehicleState, VehicleDef, Int64}, action::LatLonAccel, egoid::Int, roadway::Roadway, timestep::Float64, lat_accel::Float64, long_accel::Float64, side_slip::Float64)
    agent = vehicle.state # should pick up the AgentState here
    x = agent.posG.x
    y = agent.posG.y
    θ = agent.posG.θ
    vel = agent.v 
    # long_accel = agent.long_accel           # current longitudinal acceleration
    # lat_accel = agent.lat_accel             # current lateral acceleration
    new_long_accel = long_accel + action.a_lon

    # @show x, y, θ
    # clip to make sure that our acceleration stays within bounds, as is true in reality
    if new_long_accel > MAX_LONG_ACCEL
        new_long_accel = MAX_LONG_ACCEL
    elseif new_long_accel < -MAX_LONG_ACCEL
        new_long_accel = -MAX_LONG_ACCEL
    end

    new_lat_accel = lat_accel + action.a_lat
    if new_lat_accel > MAX_LAT_ACCEL
        new_lat_accel = MAX_LAT_ACCEL
    elseif new_lat_accel < -MAX_LAT_ACCEL
        new_lat_accel = -MAX_LAT_ACCEL
    end
    # @show new_long_accel
    # @show new_lat_accel
    # for now, getting a lot of this math from lat_lon_accel.jl

    ϕ = agent.posF.ϕ                      # lane relative heading
    curr_long_vel = vel * cos(side_slip)                # longitudinal velocity
    curr_long_vel = max(0., curr_long_vel)      # prevents reversing

    curr_lat_vel = vel * sin(side_slip)                 # lateral velocity


    new_long_vel = curr_long_vel + new_long_accel * timestep    # need to update velocities
    new_lat_vel = curr_lat_vel + new_lat_accel * timestep
    if new_lat_vel > MAX_LAT_VEL
        new_lat_vel = MAX_LAT_VEL
    elseif new_lat_vel < -MAX_LAT_VEL
        new_lat_vel = -MAX_LAT_VEL
    end

    curr_vel = sqrt(curr_long_vel^2 + curr_lat_vel^2)
    new_vel = sqrt(new_long_vel^2 + new_lat_vel^2)  # magnitude of velocity vector

    new_x = x + curr_vel * cos(θ) * timestep
    new_y = y + curr_vel * sin(θ) * timestep

    # this is from lat_lon_accel.jl
    # ds = curr_long_vel * timestep + new_long_accel * timestep^2
    # roadind = move_along(posf(agent).roadind, roadway, ds)
    # footpoint = roadway[roadind]
    # @show footpoint
    

    # the comment right below might be relevant only if something closer to steering angle actually becomes part of the action space
    # math for new theta - might just have to add the lateral accleration (this is steering rate pretty much) * timestep squared

    new_θ = θ + side_slip #atan(new_lat_vel, new_long_vel) # keep an eye on this, this is taken from lat_lon_accel.jl
    
    new_side_slip = atan(new_lat_vel, new_long_vel)

    # @show new_lat_vel
    # @show new_long_vel
    # @show new_side_slip

    posG = VecSE2(new_x, new_y, new_θ)
    # @show posG
    new_vehicle_state = VehicleState(posG, roadway, new_long_vel)

    # return AgentState(new_vehicle_state, new_long_accel, new_lat_accel)
    return new_vehicle_state, new_lat_accel, new_long_accel, new_side_slip
end