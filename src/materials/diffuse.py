from ..utils.constants import *
from ..utils.vector3 import vec3, rgb, extract
from ..utils.random import spherical_caps_pdf, cosine_pdf, mixed_pdf
from functools import reduce as reduce
from ..ray import Ray, get_raycolor
from .. import lights
import numpy as np
from . import Material
from ..textures import *


class Diffuse(Material):
    def __init__(self, diff_color, diffuse_rays=20, ambient_weight=0.5, **kwargs):
        super().__init__(**kwargs)

        if isinstance(diff_color, vec3):
            self.diff_texture = solid_color(diff_color)
        elif isinstance(diff_color, texture):
            self.diff_texture = diff_color

        self.diffuse_rays = diffuse_rays
        self.max_diffuse_reflections = 2
        self.ambient_weight = ambient_weight

    def get_color(self, scene, ray, hit) -> vec3:
        """
        Computes the color of diffuse surface intersected by the given ray.

        Args:
        - scene: A Scene object containing the list of objects in the scene.
        - ray: A Ray object containing the origin, direction, and other information of the rays.
        - hit: A Hit object containing the information of the ray-surface intersection.

        Returns:
        - A vec3 object containing the color of the diffuse surface intersected by the given ray.
        """

        hit.point = ray.origin + ray.dir * hit.distance  # intersection point
        N = hit.material.get_Normal(hit)  # normal

        diff_color = self.diff_texture.get_color(hit)

        color = rgb(0.0, 0.0, 0.0)

        # TODO: If the ray intersects the surface for the first time,
        # we generate multiple secondary rays to solve the rendering equation.
        if ray.diffuse_reflections < 1:
            raise NotImplementedError("TODO")
        # TODO: If the ray intersected with diffuse material more than once,
        # we generate only one secondary ray.
        elif ray.diffuse_reflections < self.max_diffuse_reflections:
            raise NotImplementedError("TODO")
        # TODO: Stop tracing if the recursion depth exceeds the maximum depth.
        else:
            return color
