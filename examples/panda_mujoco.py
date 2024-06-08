import os
import numpy as np

import mujoco as mj

from dynfabrics.forwardkinematics.xmlFks.generic_xml_fk import GenericXMLFk
from dynfabrics.urdfenvs.generic_mujoco.generic_mujoco_env import GenericMujocoEnv
from dynfabrics.urdfenvs.generic_mujoco.generic_mujoco_robot import GenericMujocoRobot
from dynfabrics.mpscenes.goals.goal_composition import GoalComposition
from dynfabrics.mpscenes.goals.sub_goal import SubGoalType
from dynfabrics.mpscenes.obstacles.sphere_obstacle import SphereObstacle
from dynfabrics.mpscenes.obstacles.box_obstacle import BoxObstacle
from dynfabrics.planner.parameterized_planner import ParameterizedFabricPlanner

from mjmanip.mj_utils import mj_model_joints_qids, MJ_GEOM_TYPES
from mjmanip.robot.panda_nohand import PANDA_DYNFABRICS_XML_PATH


def initialize_environment(render=True):
    """
    Initializes the simulation environment.

    Adds obstacles and goal visualizaion to the environment based and
    steps the simulation once.
    """
    robots = [GenericMujocoRobot(xml_file=PANDA_DYNFABRICS_XML_PATH)]
    home_config = np.array([0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04])

    # Definition of the obstacles
    static_obst_dict_1 = {
        "type": MJ_GEOM_TYPES[mj.mjtGeom.mjGEOM_SPHERE],
        "geometry": {"position": [0.5, 0.2, 0.2], "radius": 0.1},
    }
    static_obst1 = SphereObstacle(name="static_obst1", content_dict=static_obst_dict_1)
    static_obst_dict_2 = {
        "type": MJ_GEOM_TYPES[mj.mjtGeom.mjGEOM_SPHERE],
        "geometry": {"position": [0.2, 0.2, 0.35], "radius": 0.1},
    }
    static_obst2 = SphereObstacle(name="static_obst2", content_dict=static_obst_dict_2)

    movable_obstacle_dict = {
        'type': MJ_GEOM_TYPES[mj.mjtGeom.mjGEOM_BOX],
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

    # Definition of the goal.
    goal_dict = {
        "subgoal0": {
            "weight": 2.0,
            "is_primary_goal": True,
            "indices": [0, 1, 2],
            "parent_link": "link0",
            "child_link": "attachment",
            "desired_pose": [0.5, 0.6, 0.3, 1, 0, 0, 0],
            "epsilon": 0.05,
            "type": SubGoalType.STATIC,
        },
        "subgoal1": {
            "weight": 1.0,
            "is_primary_goal": False,
            "indices": [0, 1, 2],
            "parent_link": "link7",
            "child_link": "attachment",
            "desired_pose": [0.0, 0.2, 0.5, 1, 0, 0, 0],
            "epsilon": 0.05,
            "type": SubGoalType.STATIC,
        }
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    obstacles = [static_obst1, static_obst2, movable_box_obstacle]
    env = GenericMujocoEnv(robots=robots, obstacles=obstacles, goals=goal.sub_goals(), render=render)
    env.reset(pos=home_config)

    return (env, goal, obstacles)


def set_planner(env, goal: GoalComposition, obstacles, dt: float):
    """
    Initializes the fabric planner for the panda robot.

    This function defines the forward kinematics for collision avoidance,
    and goal reaching. These components are fed into the fabrics planner.

    In the top section of this function, an example for optional reconfiguration
    can be found. Commented by default.

    Params
    ----------
    goal: StaticSubGoal
        The goal to the motion planning problem.
    degrees_of_freedom: int
        Degrees of freedom of the robot (default = 7)
    """

    ## Optional reconfiguration of the planner
    # base_inertia = 0.03
    # attractor_potential = "20 * ca.norm_2(x)**4"
    # damper = {
    #     "alpha_b": 0.5,
    #     "alpha_eta": 0.5,
    #     "alpha_shift": 0.5,
    #     "beta_distant": 0.01,
    #     "beta_close": 6.5,
    #     "radius_shift": 0.1,
    # }
    # planner = ParameterizedFabricPlanner(
    #     degrees_of_freedom,
    #     kindyn,
    #     base_inertia=base_inertia,
    #     attractor_potential=attractor_potential,
    #     damper=damper,
    # )
    collision_geometry: str = "-4.5 / (x ** 1) * (-0.5 * (ca.sign(xdot) - 1)) * xdot ** 2"
    geometry_plane_constraint: str = (
        "-10.0 / (x ** 1) * (-0.5 * (ca.sign(xdot) - 1)) * xdot ** 2"
    )
    finsler_plane_constraint: str = (
        "1.0/(x**1) * xdot**2"
    )
    env.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
    env.joint_ids = [env.robot_model.joint(name).id for name in env.joint_names]
    # kindyn = KinDynComputations(urdfstring=env.robot_model, joints_name_list=joint_names)
    kindyn = GenericXMLFk(PANDA_DYNFABRICS_XML_PATH,
                          root_link="link0", end_links=['attachment'])
    planner = ParameterizedFabricPlanner(
        len(env.joint_names),
        kindyn,
        collision_geometry=collision_geometry,
        geometry_plane_constraint=geometry_plane_constraint,
    )
    collision_links = ['attachment']
    self_collision_pairs = {}
    # The planner hides all the logic behind the function set_components.
    planner.set_components(
        collision_links=collision_links,
        self_collision_pairs=self_collision_pairs,
        goal=goal,
        number_obstacles_sphere=len(obstacles),
        number_plane_constraints=1,
        limits=env.robot_model.jnt_range[env.joint_ids].tolist(),
    )
    planner.concretize(mode="vel", time_step=dt)
    return planner


def forward_simulate(planner: ParameterizedFabricPlanner,
                     q0: np.ndarray,
                     arguments: dict,
                     n_steps: int,
                     dt: float) -> np.ndarray:
    trajectory = np.zeros((n_steps, q0.size))
    dq = np.zeros_like(q0)
    q = q0.copy()
    alpha = 0.43
    alpha = 0
    for i in range(n_steps):
        trajectory[i] = q.copy()
        dq_new = planner.compute_action(q=q, qdot=dq, **arguments)
        dq = alpha * dq + (1 - alpha) * dq_new
        q += dq * dt

    return trajectory


def run_panda_example(n_steps=10000, render=True):
    (env, goal, obstacles) = initialize_environment(render)
    planner = set_planner(env, goal, obstacles, env.dt)
    joint_ids = env.joint_ids
    # planner.export_as_c("planner.c")
    action = np.zeros(len(joint_ids))
    ob, *_ = env.step(action)
    """
    body_links={1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 7: 0.1}
    for body_link, radius in body_links.items():
        env.add_collision_link(0, body_link, shape_type='sphere', size=[radius])
    """
    subgoals = goal.sub_goals()
    arguments = {
        **{f"x_goal_{i}": g.position() for i, g in enumerate(subgoals)},
        **{f"weight_goal_{i}": g.weight() for i, g in enumerate(subgoals)},
        **{f"radius_obst_{i}": o.size()[0] for i, o in enumerate(obstacles)},
        **{f"x_obst_{i}": o.position() for i, o in enumerate(obstacles)},
        "radius_body_attachment": 0.01,
        "constraint_0": np.array([0.0, 0.0, 1.0, -0.1]),
    }

    # q0 = ob['robot_0']['joint_state']['position'][joint_ids]
    # trajectory_forward = forward_simulate(planner, q0, arguments, n_steps, env.dt)
    # trajectory_actual = np.zeros_like(trajectory_forward)

    robot_qpos_ids = mj_model_joints_qids(env.scene_model, env.joint_names, is_qpos=True)
    robot_qdof_ids = mj_model_joints_qids(env.scene_model, env.joint_names, is_qpos=False)

    for i in range(n_steps):
        # trajectory_actual[i] = q
        action = planner.compute_action(q=env.scene_data.qpos[robot_qpos_ids],
                                        qdot=env.scene_data.qvel[robot_qdof_ids], **arguments)
        ob, reward, terminated, truncated, info = env.step(action)

        """
        if terminated or truncated:
            print(info)
            break
        """
    env.close()
    return {}


if __name__ == "__main__":
    res = run_panda_example(render=True)
