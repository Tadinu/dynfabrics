"""Turns a chain from a MuJoCo XML (MJCF) into casadi expressions, using the MjSpec API.

Counterpart of urdfparser.py (URDFparser). Mapping:
    URDF link / joint origin   -> MJCF body (pos + orientation); named geoms, geom meshes
                                  and sites are exposed as extra link frames
    revolute / continuous      -> hinge  (rotation about the joint pos)
    prismatic                  -> slide
    symbolic_parameters        -> keyed by body name or joint name, keys x,y,z,roll,pitch,yaw

The XML is loaded as an ``mj.MjSpec`` and compiled, so MuJoCo itself resolves
<compiler> angle units, eulerseq, <default> classes, quat/axisangle/euler/xyaxes/zaxis
orientations and geom fromto. Everything below reads the resolved values.
"""
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import casadi as ca
import numpy as np

import mujoco as mj

WORLD = "world"
casadiOrNumpy = Union[ca.SX, np.ndarray]

_JOINT_TYPES = {
    int(mj.mjtJoint.mjJNT_FREE): "free",
    int(mj.mjtJoint.mjJNT_BALL): "ball",
    int(mj.mjtJoint.mjJNT_SLIDE): "slide",
    int(mj.mjtJoint.mjJNT_HINGE): "hinge",
}


class LinkNotInXMLError(Exception):
    pass


# --------------------------------------------------------------------------- #
# Small transformation helpers (numpy / casadi agnostic)
# --------------------------------------------------------------------------- #
def _sx(v) -> ca.SX:
    return v if isinstance(v, ca.SX) else ca.SX(float(v))


def homogeneous_transformation(rotation=None, translation=None) -> casadiOrNumpy:
    sym = isinstance(rotation, ca.SX) or isinstance(translation, ca.SX) or (
            isinstance(translation, (list, tuple)) and any(isinstance(t, ca.SX) for t in translation)
    )
    if sym:
        T = ca.SX.eye(4)
        if rotation is not None:
            T[:3, :3] = rotation if isinstance(rotation, ca.SX) else ca.SX(np.asarray(rotation, float))
        if translation is not None:
            T[:3, 3] = translation if isinstance(translation, ca.SX) else ca.vertcat(*[_sx(t) for t in translation])
        return T
    T = np.eye(4)
    if rotation is not None:
        T[:3, :3] = rotation
    if translation is not None:
        T[:3, 3] = np.asarray(translation, float).reshape(3)
    return T


def quat2rotMat(quat) -> np.ndarray:
    """(w, x, y, z) quaternion -> rotation matrix."""
    R = np.zeros(9)
    mj.mju_quat2Mat(R, np.asarray(quat, float) / np.linalg.norm(quat))
    return R.reshape(3, 3)


def pose2T(pos, quat) -> np.ndarray:
    return homogeneous_transformation(quat2rotMat(quat), pos)


def rpy_rotation(roll, pitch, yaw) -> casadiOrNumpy:
    """URDF-style (ZYX) rotation from roll/pitch/yaw."""
    sym = any(isinstance(v, ca.SX) for v in (roll, pitch, yaw))
    cos, sin = (ca.cos, ca.sin) if sym else (np.cos, np.sin)
    cr, sr, cp, sp, cy, sy = cos(roll), sin(roll), cos(pitch), sin(pitch), cos(yaw), sin(yaw)
    rows = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]]
    if sym:
        return ca.vertcat(*[ca.horzcat(*[_sx(v) for v in r]) for r in rows])
    return np.array(rows, dtype=float)


def rotmat2rpy(R: np.ndarray) -> Tuple[float, float, float]:
    cp = np.hypot(R[0, 0], R[1, 0])
    pitch = np.arctan2(-R[2, 0], cp)
    if cp > 1e-9:
        return float(np.arctan2(R[2, 1], R[2, 2])), float(pitch), float(np.arctan2(R[1, 0], R[0, 0]))
    return float(np.arctan2(-R[1, 2], R[1, 1])), float(pitch), 0.0


def axis2rotMat(axis, theta: casadiOrNumpy) -> casadiOrNumpy:
    """Right-handed rotation of theta rad about axis (Rodrigues)."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    if isinstance(theta, ca.SX):
        return (ca.cos(theta) * ca.SX.eye(3) + (1 - ca.cos(theta)) * ca.SX(np.outer(a, a))
                + ca.sin(theta) * ca.SX(K))
    th = float(theta)
    return np.cos(th) * np.eye(3) + (1 - np.cos(th)) * np.outer(a, a) + np.sin(th) * K


@dataclass
class Joint:
    name: str
    type: str
    axis: np.ndarray
    pos: np.ndarray
    body: str


@dataclass
class Body:
    name: str
    rot: np.ndarray
    pos: np.ndarray
    parent: Optional[str] = None
    joints: List[Joint] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
class MJCFparser(object):
    """Class that turns a chain from MJCF to casadi expressions."""

    actuated_types = ["hinge", "slide"]

    def __init__(self, xml: str, root_link: str = WORLD, end_links: list = None):
        self._root_link = root_link
        self._end_links = [end_links] if isinstance(end_links, str) else end_links
        self.spec: mj.MjSpec = None
        self.model: mj.MjModel = None
        if xml.endswith('.xml') or xml.endswith('.mjcf'):
            self.load_from_file(xml)
        else:
            self.load_from_string(xml)

    # ----------------------------- loading ------------------------------ #
    def load_from_file(self, filename: str):
        """Uses an MJCF file to get robot description."""
        self.load_from_spec(mj.MjSpec.from_file(filename))

    def load_from_string(self, xmlstring: str, assets: Optional[Dict[str, bytes]] = None):
        """Uses an MJCF string to get robot description. `assets` maps file names to bytes."""
        self.load_from_spec(mj.MjSpec.from_string(xmlstring, assets=assets))

    def load_from_spec(self, spec: "mj.MjSpec"):
        """Uses an existing MjSpec (e.g. one you edited programmatically)."""
        self.spec = spec
        self.model = spec.compile()
        self.extract_information()

    def extract_information(self):
        self._actuated_joints = []
        self._active_joints = set()
        self._degrees_of_freedom = 0
        self._parse_model()
        self.detect_link_names()
        self._root_body = self._body_of(self._root_link)
        self._end_bodies = (
            [self._body_of(e) for e in self._end_links]
            if self._end_links is not None else self._leaf_bodies()
        )
        self.set_active_joints()
        self.set_actuated_joints()
        self.extract_degrees_of_freedom()

    # --------------------------- model parsing -------------------------- #
    def _name(self, objtype, i: int, prefix: str) -> str:
        return mj.mj_id2name(self.model, objtype, i) or f"{prefix}_{i}"

    def _parse_model(self):
        m = self.model
        obj = mj.mjtObj
        self._bodies: Dict[str, Body] = {}
        body_names = [WORLD if b == 0 else self._name(obj.mjOBJ_BODY, b, "body") for b in range(m.nbody)]
        for b in range(m.nbody):
            parent = None if b == 0 else body_names[m.body_parentid[b]]
            body = Body(name=body_names[b], parent=parent, rot=quat2rotMat(m.body_quat[b]), pos=m.body_pos[b].copy())
            first, num = m.body_jntadr[b], m.body_jntnum[b]
            for j in range(first, first + num) if num > 0 else []:
                body.joints.append(Joint(name=self._name(obj.mjOBJ_JOINT, j, "joint"),
                                         type=_JOINT_TYPES[int(m.jnt_type[j])],
                                         axis=m.jnt_axis[j].copy(),
                                         pos=m.jnt_pos[j].copy(), body=body_names[b]))
            self._bodies[body.name] = body

        # link name -> (owning body, local 4x4 transform w.r.t. that body)
        self._frames: Dict[str, Tuple[str, np.ndarray]] = {n: (n, np.eye(4)) for n in body_names}
        for g in range(m.ngeom):
            body = body_names[m.geom_bodyid[g]]
            T = pose2T(m.geom_pos[g], m.geom_quat[g])
            names = [mj.mj_id2name(m, obj.mjOBJ_GEOM, g)]
            if m.geom_type[g] == mj.mjtGeom.mjGEOM_MESH:
                mesh = m.geom_dataid[g]
                # the compiler re-centres mesh geoms on the mesh frame; undo it so the frame
                # matches the pos/quat written in the XML
                T = T @ np.linalg.inv(pose2T(m.mesh_pos[mesh], m.mesh_quat[mesh]))
                names.append(mj.mj_id2name(m, obj.mjOBJ_MESH, mesh))
            for n in names:
                if n:
                    self._frames.setdefault(n, (body, T))
        for s in range(m.nsite):
            n = mj.mj_id2name(m, obj.mjOBJ_SITE, s)
            if n:
                self._frames.setdefault(n, (body_names[m.site_bodyid[s]], pose2T(m.site_pos[s], m.site_quat[s])))

    def _body_of(self, link: Optional[str]) -> str:
        if link is None:
            return WORLD
        if link not in self._frames:
            raise LinkNotInXMLError(f"Link {link} is not in the xml. Possible links are {self.link_names()}")
        return self._frames[link][0]

    def _leaf_bodies(self) -> List[str]:
        parents = {b.parent for b in self._bodies.values()}
        return [n for n in self._bodies if n not in parents]

    def _ancestors(self, body: str) -> List[str]:
        """Bodies from the absolute root down to `body`."""
        chain = []
        while body is not None:
            chain.append(body)
            body = self._bodies[body].parent
        return chain[::-1]

    # --------------------------- joint bookkeeping ---------------------- #
    def set_active_joints(self) -> None:
        self._active_bodies = set()
        for end in self._end_bodies:
            b = end
            while b is not None and b != self._root_body:
                self._active_bodies.add(b)
                b = self._bodies[b].parent
        for name in self._active_bodies:
            self._active_joints.update(j.name for j in self._bodies[name].joints)

    def set_actuated_joints(self) -> None:
        for body in self._bodies.values():  # body / joint id order
            for j in body.joints:
                if j.name not in self._active_joints:
                    continue
                if j.type not in self.actuated_types:
                    raise NotImplementedError(
                        f"Joint {j.name} of type '{j.type}' is not supported (only hinge and slide).")
                self._actuated_joints.append(j.name)

    def extract_degrees_of_freedom(self) -> None:
        self._degrees_of_freedom = len(self._actuated_joints)

    def set_joint_variable_map(self) -> None:
        self._joint_map = {name: i for i, name in enumerate(self._actuated_joints)}

    def degrees_of_freedom(self) -> int:
        return self._degrees_of_freedom

    def actuated_joints(self) -> list:
        return self._actuated_joints

    def active_joints(self) -> set:
        return self._active_joints

    def link_names(self) -> list:
        return self._link_names

    def detect_link_names(self) -> list:
        self._link_names = list(self._frames.keys())
        return self._link_names

    def get_joint_info(self, root, tip) -> list:
        """Active joints on the chain to `tip`, ordered from the root."""
        return [j for b in self._active_chain(tip) for j in self._bodies[b].joints]

    def _active_chain(self, tip: str) -> List[str]:
        return [b for b in self._ancestors(self._body_of(tip)) if b in self._active_bodies]

    # ------------------------- forward kinematics ----------------------- #
    def _body_frame(self, body: Body, symbolic_parameters: dict) -> casadiOrNumpy:
        R, p = body.rot, list(body.pos)
        override = {}
        for key in [body.name] + [j.name for j in body.joints]:
            override.update(symbolic_parameters.get(key, {}))
        for i, k in enumerate(["x", "y", "z"]):
            if k in override:
                p[i] = override[k]
        if any(k in override for k in ("roll", "pitch", "yaw")):
            rpy = list(rotmat2rpy(body.rot))
            for i, k in enumerate(["roll", "pitch", "yaw"]):
                if k in override:
                    rpy[i] = override[k]
            R = rpy_rotation(*rpy)
        return homogeneous_transformation(R, p)

    @staticmethod
    def _joint_frame(joint: Joint, qi) -> casadiOrNumpy:
        if joint.type == "hinge":
            R = axis2rotMat(joint.axis, qi)
            if isinstance(R, ca.SX):
                t = ca.SX(joint.pos) - ca.mtimes(R, ca.SX(joint.pos))
            else:
                t = joint.pos - R @ joint.pos
            return homogeneous_transformation(R, t)
        # slide
        t = ca.SX(joint.axis) * qi if isinstance(qi, ca.SX) else joint.axis * qi
        return homogeneous_transformation(translation=t)

    def get_forward_kinematics(self,
                               root,
                               tip,
                               q,
                               link_transformation=np.eye(4),
                               symbolic_parameters: Dict[str, Dict[str, ca.SX]] = None) -> Dict[str, ca.SX]:
        """Returns the forward kinematics as a casadi expression."""
        if symbolic_parameters is None:
            symbolic_parameters = {}
        if tip not in self._frames:
            raise LinkNotInXMLError(f"Link {tip} is not in the xml. Possible links are {self.link_names()}")
        T_fk = ca.SX.eye(4)
        for name in self._active_chain(tip):
            body = self._bodies[name]
            T_fk = ca.mtimes(T_fk, self._body_frame(body, symbolic_parameters))
            for joint in body.joints:
                T_fk = ca.mtimes(T_fk, self._joint_frame(joint, q[self._joint_map[joint.name]]))
        T_fk = ca.mtimes(T_fk, self._frames[tip][1])
        T_fk = ca.mtimes(T_fk, link_transformation)
        return {"T_fk": T_fk}
