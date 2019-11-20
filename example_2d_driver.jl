# going off of Tim Wheeler's 2d driver model for practice
# this is the driver model in which a lot of inspiration for CS230 product will be used

# longitudinal model: IntelligentDriverModel()
# lateral mdoel: ProportionalLaneTracker()
# lane changing model (decision making, I think): TimLaneChanger
using AutomotiveDrivingModels
using Distributions
using Random

include("example_lane_changer.jl")
# include("./.julia/packages/AutomotiveDrivingModels/src/behaviors/lateral_driver_models.jl")

mutable struct Example2DDriver <: DriverModel{LatLonAccel}
    rec::SceneRecord # don't use in final CS230 project, use list of scenes instead
    mlon::LaneFollowingDriver
    mlat::LateralDriverModel
    mlane::LaneChangeModel

    function Example2DDriver(
        timestep::Float64;
        mlon::LaneFollowingDriver=IntelligentDriverModel(),
        mlat::LateralDriverModel=ProportionalLaneTracker(),
        mlane::LaneChangeModel=ExampleLaneChanger(timestep),
        rec::SceneRecord = SceneRecord(1, timestep)
    )
        model = new()
        model.rec = rec
        model.mlon = mlon
        model.mlat = mlat
        model.mlane = mlane

    return model
    end
end

get_name(::Example2DDriver) = "Example2DDriver"

function set_desired_speed!(model::Example2DDriver, v_des::Float64) # changes speed according to whatever is in set_desired_speed! for each model
    set_desired_speed!(model.mlon, v_des)
    set_desired_speed!(model.mlane, v_des)
    return model
end

function track_longitudinal!(driver::LaneFollowingDriver, scene::Frame{Entity{VehicleState, D, I}}, roadway::Roadway, vehicle_index::I, fore::NeighborLongitudinalResult) where {D, I}
    v_ego = scene[vehicle_index].state.v 
    if fore.ind != nothing
        headway, v_oth = fore.Δs, scene[fore.ind].state.v
    else
        headway, v_oth = NaN, NaN
    end
    return track_longitudinal!(driver, v_ego, v_oth, headway)
end

function AutomotiveDrivingModels.observe!(driver::Example2DDriver, scene::Frame{Entity{S, D, I}}, roadway::Roadway, egoid::I) where {S, D, I}
    update!(driver.rec, scene)
    observe!(driver.mlane, scene, roadway, egoid)

    vehicle_index = findfirst(egoid, scene)
    lane_change_action = rand(driver.mlane) # outputs decision of whether to change lane
    laneoffset = get_lane_offset(lane_change_action, driver.rec, roadway, vehicle_index)
    lateral_speed = convert(Float64, get(VELFT, driver.rec, roadway, vehicle_index))

    # check information concerning other vehicles in relation to the lane change choice
    if lane_change_action.dir == DIR_MIDDLE
        fore = get_neighbor_fore_along_lane(scene, vehicle_index, roadway, VehicleTargetPointFront(), VehicleTargetPointRear(), VehicleTargetPointFront())
    elseif lane_change_action.dir == DIR_LEFT
        fore = get_neighbor_fore_along_left_lane(scene, vehicle_index, roadway, VehicleTargetPointFront(), VehicleTargetPointRear(), VehicleTargetPointFront())
    else
        @assert(lane_change_action.dir == DIR_RIGHT)
        fore = get_neighbor_fore_along_right_lane(scene, vehicle_index, roadway, VehicleTargetPointFront(), VehicleTargetPointRear(), VehicleTargetPointFront())
    end
    AutomotiveDrivingModels.track_lateral!(driver.mlat, laneoffset, lateral_speed)
    AutomotiveDrivingModels.track_longitudinal!(driver.mlon, scene, roadway, vehicle_index, fore)

    return driver
end

Base.rand(rng::AbstractRNG, driver::Example2DDriver) = LatLonAccel(rand(rng, driver.mlat), rand(rng, driver.mlon).a)
Distributions.pdf(driver::Example2DDriver, a::LatLonAccel) = pdf(driver.mlat, a.a_lat) * pdf(driver.mlon, a.a_lon)
Distributions.logpdf(driver::Example2DDriver, a::LatLonAccel) = logpdf(driver.mlat, a.a_lat) * logpdf(driver.mlon, a.a_lon)
    
