import numpy as np
from ..utils.constants import *
from ..utils.vector3 import vec3
from ..geometry import Primitive, Collider


class Sphere(Primitive):
    def __init__(self, center, material, radius, max_ray_depth=5, shadow=True):
        super().__init__(center, material, max_ray_depth, shadow=shadow)
        self.collider_list += [
            Sphere_Collider(assigned_primitive=self, center=center, radius=radius)
        ]
        self.bounded_sphere_radius = radius

    def get_uv(self, hit):
        return hit.collider.get_uv(hit)


class Sphere_Collider(Collider):
    def __init__(self, radius, **kwargs):
        super().__init__(**kwargs)
        self.radius = radius

    def intersect(self, O, D):
        """
        Computes the intersection of a ray with the sphere.

        Args:
        - O: A vec3 object representing the origins of the rays.
        - D: A vec3 object representing the directions of the rays.

        Returns:
        - A NumPy array of shape (2, num_ray) containing
          the ray distance to the intersection point (row 0)
          and the orientation of the intersection point (row 1).
        """
        raise NotImplementedError("TODO")

    def get_Normal(self, hit):
        # M = intersection point
        return (hit.point - self.center) * (1.0 / self.radius)

    def get_uv(self, hit):
        M_C = (hit.point - self.center) / self.radius
        phi = np.arctan2(M_C.z, M_C.x)
        theta = np.arcsin(M_C.y)
        u = (phi + np.pi) / (2 * np.pi)
        v = (theta + np.pi / 2) / np.pi
        return u, v
