import sys
import numpy as np
import time
import logging
from typing import List, Optional, Union

import gymnasium as gym
from gymnasium import Env, utils
import mujoco as mj
# from dm_control import mjcf
from dynfabrics.urdfenvs.sensors.mujoco.lidar import LidarMujoco as Lidar
from dynfabrics.urdfenvs.sensors.sensor import Sensor

from dynfabrics.urdfenvs.urdf_common.helpers import (
    check_observation,
    WrongObservationError,
)

from dynfabrics.urdfenvs.generic_mujoco.generic_mujoco_robot import GenericMujocoRobot
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from dynfabrics.mpscenes.obstacles.collision_obstacle import CollisionObstacle
from dynfabrics.mpscenes.goals.sub_goal import SubGoal

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": -1,
    "distance": 4.0,
}

DEFAULT_SIZE = 480


class LinkIdNotFoundError(Exception):
    pass


class GenericMujocoEnv(Env):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }
    _t: float
    _number_movable_obstacles: int

    def __init__(
            self,
            robots: List[GenericMujocoRobot],
            obstacles: List[CollisionObstacle],
            goals: List[SubGoal],
            sensors: Optional[List[Sensor]] = None,
            render: Optional[Union[str, bool]] = None,
            dt: float = 0.01,
            n_sub_steps: int = 1,
            width: int = DEFAULT_SIZE,
            height: int = DEFAULT_SIZE,
            camera_id: Optional[int] = None,
            camera_name: Optional[str] = None,
            enforce_real_time: Optional[bool] = False,
    ) -> None:

        utils.EzPickle.__init__(self)
        self._xml_file = robots[0].xml_file
        self._done: bool = False
        self.joints_name_list: list[str] = []
        self.joint_ids: list[int] = []
        self._obstacles = obstacles
        self._goals = goals
        if not sensors:
            self._sensors = []
        else:
            self._sensors = sensors
        self._number_movable_obstacles = 0
        self._t = 0.0

        render_mode = render
        if render is True:
            render_mode = "human"
        if render is False:
            render_mode = None
        if enforce_real_time and render_mode == "human":
            self._enforce_real_time = True
        else:
            self._enforce_real_time = False
        self._sleep_offset = 0.0

        self.width = width
        self.height = height

        self._dt = dt
        self._n_sub_steps = n_sub_steps
        self._initialize_simulation()

        assert self.metadata["render_modes"] == [
            "human",
            "rgb_array",
            "depth_array",
        ], self.metadata["render_modes"]
        self.observation_space = self.get_observation_space()
        self.action_space = self.get_action_space()

        self.render_mode = render_mode
        self.camera_name = camera_name
        self.camera_id = camera_id

        self.mujoco_renderer = MujocoRenderer(
            self.scene_model,
            self.scene_data,
            DEFAULT_CAMERA_CONFIG,
            height=DEFAULT_SIZE,
            width=DEFAULT_SIZE,
        )
        self._end_last_step_time = time.time()

    def render(self):
        return self.mujoco_renderer.render(self.render_mode)

    def get_observation_space(self) -> gym.spaces.Dict:
        """Get observation space."""
        observation_space_as_dict = dict(
            gym.spaces.Dict(
                {
                    "joint_state": gym.spaces.Dict(
                        {
                            "position": gym.spaces.Box(
                                low=self.joint_limits()[:, 0],
                                high=self.joint_limits()[:, 1],
                                dtype=float,
                            ),
                            "velocity": gym.spaces.Box(
                                low=np.ones_like(self.joint_limits()[:, 0])
                                    * -10.0,
                                high=np.ones_like(self.joint_limits()[:, 0])
                                     * 10.0,
                                dtype=float,
                            ),
                        }
                    )
                }
            )
        )
        for sensor in self._sensors:
            observation_space_as_dict.update(
                sensor.get_observation_space(
                    self.obstacles_dict, self.goals_dict
                )
            )
        observation_space = gym.spaces.Dict(
            {"robot_0": gym.spaces.Dict(observation_space_as_dict)}
        )
        return observation_space

    def get_action_space(self) -> np.ndarray:
        return gym.spaces.Box(
            low=self.actuator_limits()[:, 0],
            high=self.actuator_limits()[:, 1],
            dtype=float,
        )

    def joint_limits(self) -> np.ndarray:
        return self.scene_model.jnt_range[: self.nq]

    def actuator_limits(self) -> np.ndarray:
        return self.scene_model.actuator_ctrlrange

    def velocity_limits(self) -> np.ndarray:
        return self.scene_model.actuator_ctrlrange

    def _initialize_simulation(self):
        self.model_spec = mj.MjSpec.from_file(self._xml_file)
        self.robot_model = self.model_spec.compile()
        self.add_scene()
        self.scene_model = self.model_spec.compile()
        self.scene_model.body_pos[0] = [0, 1, 1]
        self.scene_model.vis.global_.offwidth = self.width
        self.scene_model.vis.global_.offheight = self.height
        self.scene_model.opt.timestep = self._dt / self._n_sub_steps
        self.scene_data = mj.MjData(self.scene_model)

        for sensor in self._sensors:
            sensor.set_data(self.scene_data)

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def t(self) -> float:
        return self._t

    @property
    def obstacles_dict(self) -> dict:
        return {k: v for k, v in enumerate(self._obstacles)}

    @property
    def goals_dict(self) -> dict:
        return {k: v for k, v in enumerate(self._goals)}

    def update_obstacles_position(self):

        non_movable_obstacles = [
            obstacle for obstacle in self._obstacles if not obstacle.movable()
        ]
        for i, obstacle in enumerate(non_movable_obstacles):
            self.scene_data.mocap_pos[i] = obstacle.position(t=self.t)

    def update_goals_position(self):
        for i, goal in enumerate(self._goals):
            self.scene_data.site_xpos[i] = goal.position(t=self.t)

    def step(self, action: np.ndarray):
        target_end_step_time = self._end_last_step_time + self.dt

        self._t += self.dt
        truncated = False
        info = {}
        reward = 0
        if not self.action_space.contains(action):
            self._done = True
            info = {"action_limits": f"{action} not in {self.action_space}"}

        self.do_simulation(action)
        for contact in self.scene_data.contact:
            body1 = self.scene_model.geom(contact.geom1).name
            body2 = self.scene_model.geom(contact.geom2).name
            if "movable" in body1 or "movable" in body2:
                continue

            message = (
                f"Collision occured at {round(self.t, 2)} "
                f"between {body1} and obstacle "
                f"with id {body2}"
            )
            info = {"Collision": message}
            self._done = True
        self.update_obstacles_position()
        self.update_goals_position()
        if self.render_mode == "human":
            self.render()

        ob = self._get_obs()
        if not self.observation_space.contains(ob):
            try:
                check_observation(self.observation_space, ob)
            except WrongObservationError as e:
                self._done = True
                info = {"observation_limits": str(e)}
        time_before_sleep = time.time()
        sleep_time = target_end_step_time - time_before_sleep - self._sleep_offset
        if self._enforce_real_time and sleep_time > 0:
            time.sleep(sleep_time)
        time_after_sleep = time.time()
        # Compute the real-time factor (RTF)
        real_time_step = time_after_sleep - self._end_last_step_time
        rtf = self.dt / (real_time_step)
        if real_time_step < self.dt:
            self._sleep_offset -= 0.0001
        else:
            self._sleep_offset += 0.0001
        logging.info(f"Real time factor {rtf:.4f}")
        self._end_last_step_time = time.time()
        return (
            ob,
            reward,
            self._done,
            truncated,
            info,
        )

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[dict] = None,
              pos: Optional[np.ndarray] = None,
              vel: Optional[np.ndarray] = None):
        super().reset(seed=seed, options=options)
        if options and options.get("randomize_obstacles", False):
            for obstacle in self._obstacles:
                obstacle.shuffle()
        if options and options.get("randomize_goals", False):
            for goal in self._goals:
                goal.shuffle()
        if pos is not None:
            qpos = pos
        else:
            qpos = (self.joint_limits()[:, 1] + self.joint_limits()[:, 0]) / 2
        if vel is not None:
            qvel = vel
        else:
            qvel = np.zeros(self.nv)
        self.set_state(qpos, qvel)
        self._t = 0.0
        self._end_last_step_time = time.time()
        return self._get_obs(), {}

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.nq,) and qvel.shape == (self.nv,)
        self.scene_data.qpos[: self.nq] = np.copy(qpos)
        self.scene_data.qvel[: self.nv] = np.copy(qvel)
        if self.scene_model.na == 0:
            self.scene_data.act[:] = None
        mj.mj_forward(self.scene_model, self.scene_data)

    @property
    def nu(self) -> int:
        return self.scene_model.nu

    @property
    def nq(self) -> int:
        return self.scene_model.nq - 7 * self._number_movable_obstacles

    @property
    def nv(self) -> int:
        return self.scene_model.nv - 6 * self._number_movable_obstacles

    def do_simulation(self, ctrl) -> None:
        # Check control input is contained in the action space
        if np.array(ctrl).shape != (self.scene_model.nu,):
            raise ValueError(
                f"Action dimension mismatch. Expected {(self.scene_model.nu,)}, found {np.array(ctrl).shape}"
            )
        self._step_mujoco_simulation(ctrl)

    def _step_mujoco_simulation(self, ctrl):
        self.scene_data.ctrl[:] = ctrl
        mj.mj_step(self.scene_model, self.scene_data, nstep=self._n_sub_steps)

    def add_scene(self) -> None:
        for obstacle in self._obstacles:
            self.add_obstacle(obstacle)
        for sub_goal in self._goals:
            self.add_sub_goal(sub_goal)
        for sensor in self._sensors:
            if isinstance(sensor, Lidar):
                self.add_range_finder(sensor)

    def add_range_finder(self, sensor: Lidar) -> None:
        try:
            lidar_body = self.scene_model.body(sensor._link_name)
        except Exception as e:
            print(e)
            raise LinkIdNotFoundError(
                f"Link '{sensor._link_name}' not found in xml. It might be in an include."
            )

        for i in range(sensor._nb_rays):
            angle = i / sensor._nb_rays * np.pi * 2
            start_position = np.array(
                [
                    np.cos(angle) * 0.00,
                    np.sin(angle) * 0.00,
                    0.0,
                ]
            )
            end_position = np.array(
                [
                    np.cos(angle) * 0.01,
                    np.sin(angle) * 0.01,
                    0.0,
                ]
            )
            fromto_string = " ".join(map(str, start_position))
            fromto_string += " " + " ".join(map(str, end_position))

            site_values = {
                "name": f"{sensor.name()}_rf_{i}",
                "type": "capsule",
                "size": "0.01",
                "fromto": fromto_string,
            }
            lidar_body.add("site", **site_values)

            rangefinder_values = {
                "name": f"{sensor.name()}_{i}",
                "site": f"{sensor.name()}_rf_{i}",
                "cutoff": str(sensor._ray_length + 0.01 / 2),
            }
            self.model_spec.add_sensor(name=f"{sensor.name()}_{i}", site=f"{sensor.name()}_rf_{i}",
                                       cutoff=str(sensor._ray_length + 0.01 / 2))

    def add_obstacle(self, obst: CollisionObstacle) -> None:
        obstacle_body = self.model_spec.worldbody.add_body(
            name=obst.name(),
            pos=obst.position().tolist(),
            mocap=not obst.movable(),
        )
        if obst.movable():
            obstacle_body.add_freejoint(name=f"freejoint_{obst.name()}", align=True)
            self._number_movable_obstacles += 1
        obst_type = obst.type()
        obstacle_body.add_geom(name=obst.name(),
                               type=mj.mjtGeom.mjGEOM_SPHERE if obst_type == "sphere" else mj.mjtGeom.mjGEOM_BOX,
                               rgba=obst.rgba(),
                               size=obst.size())

    def add_sub_goal(self, sub_goal: SubGoal) -> None:
        position = np.array(sub_goal.position())
        self.model_spec.worldbody.add_site(name=sub_goal.name(),
                                           type=mj.mjtGeom.mjGEOM_SPHERE,
                                           rgba=[0, 1, 0, 0.3],
                                           pos=position,
                                           quat=sub_goal.orientation(),
                                           size=sub_goal.epsilon())

    def _get_obs(self):
        observation = {
            "joint_state": {
                "position": self.scene_data.qpos[0: self.nq],
                "velocity": self.scene_data.qvel[0: self.nv],
            }
        }
        for sensor in self._sensors:
            observation.update(
                {
                    sensor.name(): sensor.sense(
                        0, self.obstacles_dict, self.goals_dict, self.t
                    )
                }
            )
        return {"robot_0": observation}

    def close(self):
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()

    def xml(self) -> str:
        """Return the xml string of the model.

        As the model from the xml file may be changed during the simulation,
        by adding obstacles, goals, sensors, etc., this method ensures that
        the updated model can be saved as an xml file.

        """
        return self.model_spec.to_xml()
