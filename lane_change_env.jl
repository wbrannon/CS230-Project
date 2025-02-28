# generate environment with random human driven vehicles, all controlled via IDM
# may explore controlling human driven vehicles with lat_long_driver.jl behavioral model so they act a little smarter

using AutomotiveDrivingModels
using AutoViz
using Random
using Parameters

# for use with MDP formulation
# this may or may not make it easier
# use populate_env! function to populate roadway
@with_kw mutable struct laneChangeEnvironment
    nlanes::Int = 4
    starting_lane::Int = 1
    current_lane::Int = starting_lane
    desired_lane::Int = nlanes
    ncars::Int = 1                
    road_length::Float64 = 500.
    roadway::Roadway = gen_straight_roadway(nlanes, road_length)
    scene::Scene = Scene()
    ego_idx::Int = 1
    collision::Bool = false
    terminal_state::Bool = false
    num_steps::Int = 0
    max_steps::Int = 200
end

# check if the random spot chosen in populate_env is already currently taken; if it is, this will return false. If it is available,
# then the spots_taken tuple will be appended with the new spot 
function lane_available!(des_spot::Tuple{Int, Float64}, spots_taken::Array{Tuple{Int, Float64}})
    # use conditional statement to check if the space is available
    if any([des_spot == spots_taken[i] for i=1:length(spots_taken)])
        # spot is already taken up
        # @show "FALSE"
        return false
    else
        # @show "true"
        # @show spots_taken
        # push!(spots_taken, des_spot)
        # @show spots_taken
        return true
    end
end

    

# randomly populate road with a specified amount of human driven vehicles. For now, don't assume that the ego vehicle starts from
# the same spot every time
function populate_env!(ncars::Int, nlanes::Int, road_length::Float64, roadway::Roadway, scene::Scene)
    # initialize spots_taken
    spots_taken = [(1, 0.)] # make array of tuples to determine if a spot is taken up - this first one is where we want AV to be 
    curve = roadway[1].lanes[1].curve
    min_speed = 10.0 # m/s
    max_speed = 30.0 # m/s 
    number_car_placed = 2
    for i in 1:ncars
        car_placed = false
        while car_placed == false
            lane = rand(collect(1:nlanes))
            x_pos = rand(collect(0.0:10.0:road_length))
            # y_pos = 0.0 # place cars in Frenet frame, initialize at middle of designated lane
            # theta = 0.0 # place cars in Frenet frame, initialize as facing in the correct direction
            curr_vel = rand(collect(min_speed:max_speed))
            # des_vel = rand(collect(min_speed:max_speed)) # add desired velocity later
            des_spot = (lane, x_pos)
            # for now, just quickly make sure that we don't place a HV where the AV is supposed to start
            if lane_available!(des_spot, spots_taken)
                car_placed = true
                posG = VecSE2(x_pos, 0., 0.)
                lane = Lane(LaneTag(1, lane), curve)
                posF = Frenet(posG, lane, roadway)
                car_initial_state = VehicleState(posF, roadway, curr_vel)
                car = Vehicle(car_initial_state, VehicleDef(), number_car_placed)
                push!(scene, car)
                number_car_placed += 1
            end
            
        end
    end
end


function create_env(env::laneChangeEnvironment)
    # roadway = gen_straight_roadway(nlanes, road_length)
    # scene = Scene()
    populate_env!(env.ncars, env.nlanes, env.road_length, env.roadway, env.scene)
    return env.scene, env.roadway
end