import gymnasium as gym
import sys
import os
import shutil
import logging
import numpy as np
from importlib import resources

from dynfabrics.forwardkinematics.xmlFks.generic_xml_fk import GenericXMLFk
from dynfabrics.urdfenvs.generic_mujoco.generic_mujoco_env import GenericMujocoEnv
from dynfabrics.urdfenvs.generic_mujoco.generic_mujoco_robot import GenericMujocoRobot
from dynfabrics.urdfenvs.sensors.full_sensor import FullSensor
from dynfabrics.mpscenes.goals.goal_composition import GoalComposition
from dynfabrics.mpscenes.obstacles.sphere_obstacle import SphereObstacle
from dynfabrics.mpscenes.obstacles.box_obstacle import BoxObstacle
from dynfabrics.planner.parameterized_planner import ParameterizedFabricPlanner

from mjmanip.robot.panda_nohand import PANDA_XML_PATH


def initalize_environment(render=True):
    """
    Initializes the simulation environment.

    Adds obstacles and goal visualizaion to the environment based and
    steps the simulation once.
    """

    # xml_file = robot_model.get_xml_path()
    robots = [GenericMujocoRobot(xml_file=PANDA_XML_PATH)]
    home_config = np.array([0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04])

    # Sensors
    full_sensor = FullSensor(
        goal_mask=["position", "velocity", "weight"],
        obstacle_mask=[],
        variance=0.0,
        physics_engine_name="mujoco"
    )

    # Obstacles
    static_obst_dict_1 = {
        "type": "sphere",
        "geometry": {"position": [0.1, 0.5, 0.7], "radius": 0.1},
    }
    static_obst1 = SphereObstacle(name="static_obst1", content_dict=static_obst_dict_1)
    static_obst_dict_2 = {
        "type": "sphere",
        "geometry": {"position": [-0.4, 0.2, 0.5], "radius": 0.1},
    }
    static_obst2 = SphereObstacle(name="static_obst2", content_dict=static_obst_dict_2)

    movable_obstacle_dict = {
        'type': 'box',
        'geometry': {
            'position': [0.1, 0.5, 0.2],
            'orientation': [0.923, 0, 0, -0.38],
            'width': 0.2,
            'height': 0.2,
            'length': 0.2,
        },
        'movable': True,
        'high': {
            'position': [1.0, 0.5, 0.2],
            'width': 0.2,
            'height': 0.2,
            'length': 0.2,
        },
        'low': {
            'position': [0.0, 0.0, 0.2],
            'width': 0.2,
            'height': 0.2,
            'length': 0.2,
        }
    }
    movable_box_obstacle = BoxObstacle(name="movable_box", content_dict=movable_obstacle_dict)
    obstacles = [static_obst1, static_obst2, movable_box_obstacle]

    # Definition of the goal.
    goal_dict = {
        "subgoal0": {
            "weight": 0.4,
            "is_primary_goal": True,
            "indices": [0, 1, 2],
            "parent_link": "link0",
            "child_link": "attachment",
            "trajectory": ["0.1 * sp.cos(0.2 * t)", "-0.5 * sp.sin(0.2 * t)", "0.5 + -0.1 * sp.sin(0.2 * t)"],
            "epsilon": 0.02,
            "type": "analyticSubGoal",
        },
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    env = GenericMujocoEnv(robots=robots, goals=goal.sub_goals(),
                           sensors=[full_sensor],  # fsd_sensor
                           width=1024,
                           height=960,
                           obstacles=obstacles,
                           render=render,
                           enforce_real_time=False)
    env.reset(pos=home_config)
    return (env, goal, obstacles)


def set_planner(goal: GoalComposition, degrees_of_freedom: int = 7):
    """
    Initializes the fabric planner for the panda robot.

    This function defines the forward kinematics for collision avoidance,
    and goal reaching. These components are fed into the fabrics planner.

    In the top section of this function, an example for optional reconfiguration
    can be found. Commented by default.

    Params
    ----------
    goal: GoalComposition
        The goal to the motion planning problem. The goal can be composed
        of several subgoals.
    degrees_of_freedom: int
        Degrees of freedom of the robot (default = 7)
    """
    with open(PANDA_XML_PATH, "r", encoding="utf-8") as file:
        xml = file.read()
    forward_kinematics = GenericXMLFk(
        xml,
        root_link="link0",
        end_links=["attachment"]
    )
    planner = ParameterizedFabricPlanner(
        degrees_of_freedom,
        forward_kinematics,
    )
    # The planner hides all the logic behind the function set_components.
    collision_links = ['link7']
    self_collision_pairs = {}
    # No need to add limits for joint fingers regardless
    panda_limits = [
        [-2.8973, 2.8973],
        [-1.7628, 1.7628],
        [-2.8973, 2.8973],
        [-3.0718, -0.0698],
        [-2.8973, 2.8973],
        [-0.0175, 3.7525],
        [-2.8973, 2.8973]
    ]
    planner.set_components(
        collision_links=collision_links,
        self_collision_pairs=self_collision_pairs,
        goal=goal,
        number_obstacles=3,
        number_plane_constraints=1,
        limits=panda_limits
    )
    planner.concretize()
    return planner


def run_panda_trajectory_example(n_steps=5000, render=True, dynamic_fabric: bool = True):
    (env, goal, obstacles) = initalize_environment(render)
    planner = set_planner(goal)
    ob, *_ = env.step(np.zeros(7))

    sub_goal_0_acceleration = np.zeros(3)
    logging.warning(f"Running example with dynamic fabrics? {dynamic_fabric}")
    for _ in range(n_steps):
        ob_robot = ob['robot_0']
        sub_goal_0_position = ob_robot['FullSensor']['goals'][0]['position']
        sub_goal_0_velocity = ob_robot['FullSensor']['goals'][0]['velocity']
        if not dynamic_fabric:
            sub_goal_0_velocity *= 0
        action = planner.compute_action(
            q=ob_robot["joint_state"]["position"],
            qdot=ob_robot["joint_state"]["velocity"],
            x_ref_goal_0_leaf=sub_goal_0_position,
            xdot_ref_goal_0_leaf=sub_goal_0_velocity,
            xddot_ref_goal_0_leaf=sub_goal_0_acceleration,
            weight_goal_0=ob_robot['FullSensor']['goals'][0]['weight'],
            radius_body_link7=0.02,
            radius_obst_0=obstacles[0].size()[0],
            x_obst_0=obstacles[0].position(),
            constraint_0=np.array([0.0, 0.0, 1.0, -0.1]),
            radius_obst_1=obstacles[1].size()[0],
            x_obst_1=obstacles[1].position(),
            radius_obst_2=obstacles[2].size()[0],
            x_obst_2=obstacles[2].position(),
        )
        print("ACTION", action)
        ob, *_ = env.step(action[:7])
    env.close()
    return {}


if __name__ == "__main__":
    dynamic_fabric = True
    res = run_panda_trajectory_example(n_steps=5000, dynamic_fabric=dynamic_fabric)
