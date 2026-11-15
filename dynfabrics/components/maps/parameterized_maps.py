from typing import List
import casadi as ca
from dynfabrics.diffGeometry.diffMap import (
    DifferentialMap,
)
from dynfabrics.helpers.distances import (capsule_to_sphere, cuboid_to_sphere,
                                          sphere_to_plane, cuboid_to_capsule)
from dynfabrics.helpers.variables import Variables


class ParameterizedGoalMap(DifferentialMap):
    def __init__(self, var, fk, reference_variable):
        phi = fk - reference_variable
        super().__init__(phi, var)


class ParameterizedGeometryMap(DifferentialMap):
    pass


class SphereSphereMap(ParameterizedGeometryMap):
    def __init__(
            self,
            var: Variables,
            sphere_1_position: ca.SX,
            sphere_2_position: ca.SX,
            sphere_1_radius: ca.SX,
            sphere_2_radius: ca.SX,
    ):
        phi = (
                ca.norm_2(sphere_1_position - sphere_2_position)
                / (sphere_1_radius + sphere_2_radius)
                - 1
        )
        super().__init__(phi, var)


class CapsuleSphereMap(ParameterizedGeometryMap):
    def __init__(
            self,
            var: Variables,
            capsule_centers: List[ca.SX],
            sphere_center: ca.SX,
            capsule_radius: ca.SX,
            sphere_radius: ca.SX,
    ):
        phi = capsule_to_sphere(
            capsule_centers, sphere_center, capsule_radius, sphere_radius
        )
        super().__init__(phi, var)


class CapsuleCuboidMap(ParameterizedGeometryMap):
    def __init__(
            self,
            var: Variables,
            capsule_centers: List[ca.SX],
            cuboid_center: ca.SX,
            capsule_radius: ca.SX,
            cuboid_size: ca.SX,
    ):
        phi = cuboid_to_capsule(
            cuboid_center, capsule_centers, cuboid_size, capsule_radius
        )
        super().__init__(phi, var)


class PlaneSphereMap(ParameterizedGeometryMap):
    def __init__(
            self,
            var: Variables,
            sphere_center: ca.SX,
            sphere_radius: ca.SX,
            constraint: ca.SX,
    ):
        phi = sphere_to_plane(sphere_center, constraint, sphere_radius)

        super().__init__(phi, var)


class CuboidSphereMap(ParameterizedGeometryMap):
    def __init__(
            self,
            var: Variables,
            sphere_center: ca.SX,
            cuboid_center: ca.SX,
            sphere_radius: ca.SX,
            cuboid_size: ca.SX,
    ):
        phi = cuboid_to_sphere(cuboid_center, sphere_center, cuboid_size, sphere_radius)

        super().__init__(phi, var)


class ParameterizedPlaneConstraintMap(ParameterizedGeometryMap):
    def __init__(
            self,
            var: Variables,
            fk,
            constraint_variable,
            radius_body_variable,
    ):
        pos = fk
        normal = constraint_variable[0:3]
        offset = constraint_variable[3]

        phi = ca.fabs(ca.dot(normal, pos) + offset) / ca.norm_2(normal) - radius_body_variable

        super().__init__(phi, var)
