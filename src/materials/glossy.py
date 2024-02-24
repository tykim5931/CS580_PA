from ..utils.constants import *
from ..utils.vector3 import vec3, rgb, extract
from functools import reduce as reduce
from ..ray import Ray, get_raycolor
from .. import lights
import numpy as np
from . import Material
from ..textures import *


class Glossy(Material):
    def __init__(self, diff_color, roughness, spec_coeff, diff_coeff, n, **kwargs):
        super().__init__(**kwargs)

        if isinstance(diff_color, vec3):
            self.diff_texture = solid_color(diff_color)
        elif isinstance(diff_color, texture):
            self.diff_texture = diff_color

        self.roughness = roughness
        self.diff_coeff = diff_coeff
        self.spec_coeff = spec_coeff
        self.n = n  # index of refraction

    def get_color(self, scene, ray, hit):
        """
        Computes the color of glossy surface intersected by the given ray.

        Args:
        - scene: A Scene object containing the list of objects in the scene.
        - ray: A Ray object containing the origin, direction, and other information of the rays.
        - hit: A Hit object containing the information of the ray-surface intersection.

        Returns:
        - A vec3 object containing the color of the diffuse surface intersected by the given ray.
        """

        hit.point = ray.origin + ray.dir * hit.distance  # intersection point
        N = hit.material.get_Normal(hit)  # normal

        diff_color = self.diff_texture.get_color(hit) * self.diff_coeff

        # Ambient color
        color = scene.ambient_color * diff_color
        V = ray.dir * -1.0
        nudged = hit.point + N * 0.000001  # M nudged to avoid itself

        for light in scene.Light_list:

            L = light.get_L()  # direction to light
            dist_light = light.get_distance(hit.point)  # distance to light
            NdotL = np.maximum(N.dot(L), 0.0)
            lv = light.get_irradiance(
                dist_light, NdotL
            )  # amount of intensity that falls on the surface

            H = (L + V).normalize()  # Half-way vector

            # Shadow: find if the point is shadowed or not.
            # This amounts to finding out if M can see the light
            # Shoot a ray from M to L and check what object is the nearest
            if not scene.shadowed_collider_list == []:
                inters = [s.intersect(nudged, L) for s in scene.shadowed_collider_list]
                light_distances, light_hit_orientation = zip(*inters)
                light_nearest = reduce(np.minimum, light_distances)
                seelight = light_nearest >= dist_light
            else:
                seelight = 1.0

            # Lambert shading (diffuse)
            color += diff_color * lv * seelight

            if self.roughness != 0.0:
                # TODO: Compute ray color using the Cook-Torrance model
                raise NotImplementedError("TODO")

        # Reflection
        if ray.depth < hit.surface.max_ray_depth:
            # TODO: Compute color contribution from the reflected ray
            raise NotImplementedError("TODO")

        return color
