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
        # raise NotImplementedError("TODO")
        
        # R(t) = O +tD , (P-C)^2 - r^2 = 0
        # t^2 (D⋅D)+2t(D⋅(O−C))+(O−C)⋅(O−C)−r^2 =0
        # t = -b +- sqrt(discriminat) / 2a
        a = D.dot(D)
        O_C = O-self.center
        b = 2*D.dot(O_C)
        c = (O_C).dot(O_C) - self.radius**2
        discriminant = b**2 - 4*a*c
        
        # if disc < 0, ray does not intersect
        # if disc > 0, intersect with 2
        # if disc = 0, ray meets sphere 1
        # let disc > 0.
        r1 = (-b - np.sqrt(np.abs(discriminant))) / (2*a)
        r2 = (-b + np.sqrt(np.abs(discriminant))) / (2*a)
        r= np.where((r1 > 0) & (r1 < r2), r1, r2)
        M = O + D*r # intersection point
        NdotD = ((M-self.center)/self.radius).dot(D)
        
        hit = (discriminant > 0) & (r>0)
        hit_UPWARDS = NdotD < 0
        hit_UPDOWN = np.logical_not(hit_UPWARDS)

        pred1 = hit & hit_UPWARDS
        pred2 = hit & hit_UPDOWN
        pred3 = True
        return np.select(
            [pred1, pred2, pred3],
            [
                [r, np.tile(UPWARDS, r.shape)],
                [r, np.tile(UPDOWN, r.shape)],
                FARAWAY,
            ],
        )

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
