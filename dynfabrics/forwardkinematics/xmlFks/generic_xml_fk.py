from typing import List, Union

import casadi as ca
import numpy as np

from dynfabrics.forwardkinematics.fksCommon.fk import ForwardKinematics
from dynfabrics.forwardkinematics.xmlFks.mjcf_parser import MJCFparser, LinkNotInXMLError


class GenericXMLFk(ForwardKinematics):
    """Forward kinematics from MuJoCo XML; same interface as URDFForwardKinematics."""

    def __init__(self,
                 xml: str,
                 root_link: str,
                 end_links: List[str],
                 base_type: str = "holonomic"):
        super().__init__()
        self._desc = xml
        self._root_link = root_link
        self._end_links = end_links
        self._base_type = base_type
        self._mjcf = self.read_xml()
        self._n = self._mjcf.degrees_of_freedom()
        self._q_ca = ca.SX.sym("q", self._n)
        self._mount_transformation = np.identity(4)
        if base_type in ["diffdrive"]:
            self._q_base = ca.SX.sym("q_base", 3)
        self.generate_functions()

    def n(self) -> int:
        if self._base_type == "diffdrive":
            return self._n + 3
        return self._n

    def read_xml(self) -> MJCFparser:
        mjcf = MJCFparser(xml=self._desc, root_link=self._root_link, end_links=self._end_links)
        mjcf.detect_link_names()
        mjcf.set_joint_variable_map()
        return mjcf

    def generate_functions(self):
        self._fks = {}
        for link in self._mjcf.link_names():
            if self._base_type in ["diffdrive"]:
                q = ca.vcat([self._q_base, self._q_ca])
            else:
                q = self._q_ca
            fname = "fk" + "".join(c if c.isalnum() else "_" for c in link)
            self._fks[link] = ca.Function(fname, [q], [self.casadi(q, link)])

    def casadi(self,
               q: ca.SX,
               child_link: str,
               parent_link: Union[str, None] = None,
               link_transformation=np.eye(4),
               position_only=False,
               symbolic_parameters=None):
        if symbolic_parameters is None:
            symbolic_parameters = {}
        if parent_link is None:
            parent_link = self._root_link
        if child_link not in self._mjcf.link_names():
            raise LinkNotInXMLError(
                f"""The link you have requested, {child_link}, is not in the xml.
                    Possible links are  {self._mjcf.link_names()}"""
            )
        if self._base_type in ["diffdrive"]:
            fk = self._mjcf.get_forward_kinematics(
                parent_link, child_link, q[3:], link_transformation, symbolic_parameters
            )["T_fk"]
            c = ca.cos(q[2])
            s = ca.sin(q[2])
            T_base = ca.vcat(
                [
                    ca.hcat([c, -s, 0, q[0]]),
                    ca.hcat([s, c, 0, q[1]]),
                    ca.hcat([0, 0, 1, 0]),
                    ca.hcat([0, 0, 0, 1]),
                ]
            )
            fk = ca.mtimes(T_base, fk)
        else:
            fk = self._mjcf.get_forward_kinematics(
                parent_link, child_link, q, link_transformation, symbolic_parameters
            )["T_fk"]
            fk = ca.mtimes(self._mount_transformation, fk)

        if position_only:
            fk = fk[:3, 3]
        return fk

    def numpy(self,
              q: np.ndarray,
              child_link: str,
              parent_link: Union[str, None] = None,
              link_transformation=np.eye(4),
              position_only=False):
        if child_link not in self._fks and child_link != self._root_link:
            raise LinkNotInXMLError(
                f"""The link you have requested, {child_link}, is not in the urdf.
                    Possible links are  {self.robot.link_names()}"""
            )
        if parent_link is None:
            parent_link = self._root_link
        if parent_link not in self._fks and parent_link != self._root_link:
            raise LinkNotInXMLError(
                f"""The link you have requested, {parent_link}, is not in the urdf.
                    Possible links are  {self.robot.link_names()}"""
            )
        if parent_link == self._root_link:
            fk_parent = np.identity(4)
        else:
            fk_parent = self._fks[parent_link](q)
        if child_link == self._root_link:
            fk_child = np.identity(4)
        else:
            fk_child = self._fks[child_link](q)
        tf_parent_child = np.dot(np.linalg.inv(fk_parent), fk_child)
        tf_parent_child = np.dot(
            link_transformation, tf_parent_child
        )  # ToDo check if correct
        if position_only:
            return tf_parent_child[0:3, 3]
        else:
            return tf_parent_child

    def fk(self, q, child_link, parent_link=None, link_transformation=np.eye(4), position_only=False):
        """Backwards compatible entry point: casadi for symbolic q, numpy otherwise."""
        if isinstance(q, ca.SX):
            return self.casadi(q, child_link, parent_link, link_transformation, position_only)
        return self.numpy(q, child_link, parent_link, link_transformation, position_only)
