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
            # raise NotImplementedError("TODO")
            nudged = hit.point + N * 0.000001  # M nudged to avoid itself

            # To parallelize for loop, make as repeated matrix (sample at once!!)
            nudged_20 = nudged.repeat(self.diffuse_rays)
            N_20 = N.repeat(self.diffuse_rays)
            ray_n_20 = ray.n if ray.n.shape() == 1 else ray.n.repeat(self.diffuse_rays) # if no refraction we're okay but should be handled in case of diffuse

            pdf = cosine_pdf(N_20.shape()[0], N_20)
            reflected_rays_dir = pdf.generate() # already normalized
            pdf_val = pdf.value(reflected_rays_dir)
            
            reflected_ray = Ray(
                nudged_20,
                reflected_rays_dir,
                ray.depth + 1,
                ray_n_20,
                ray.reflections + 1,
                ray.transmissions,
                ray.diffuse_reflections + 1
            )
            N_dot_L_20 = np.clip(reflected_rays_dir.dot(N_20), 0., 1.)
            every_color = get_raycolor(reflected_ray, scene) * N_dot_L_20 / pdf_val
            mean_c_sample = every_color.reshape(N.shape()[0], self.diffuse_rays).mean(1)

            # color_temp = rgb(0.,0.,0.)
            # for i in range(self.diffuse_rays):
            #     pdf = cosine_pdf(N.shape()[0], N)
            #     reflected_rays_dir = pdf.generate()
            #     pdf_val = pdf.value(reflected_rays_dir)
            #     reflected_ray = Ray(
            #         nudged,
            #         reflected_rays_dir,
            #         ray.depth + 1,
            #         ray.n,
            #         ray.reflections + 1,
            #         ray.transmissions,
            #         ray.diffuse_reflections + 1
            #     )
            #     N_dot_L = np.clip(reflected_rays_dir.dot(N), 0., 1.)
            #     color_temp += get_raycolor(reflected_ray, scene) * N_dot_L / pdf_val
            # mean_c_sample = color_temp / self.diffuse_rays / np.pi
            color += diff_color * mean_c_sample / np.pi

            return color
            
        # TODO: If the ray intersected with diffuse material more than once,
        # we generate only one secondary ray.
        elif ray.diffuse_reflections < self.max_diffuse_reflections:
            # raise NotImplementedError("TODO")
            nudged = hit.point + N * 0.000001  # M nudged to avoid itself
            
            pdf = cosine_pdf(nudged.shape()[0], N)
            reflected_rays_dir = pdf.generate()
            pdf_val = pdf.value(reflected_rays_dir)
            reflected_ray = Ray(
                nudged,
                reflected_rays_dir,
                ray.depth + 1,
                ray.n,
                ray.reflections + 1,
                ray.transmissions,
                ray.diffuse_reflections + 1
            )
            N_dot_L = np.clip(reflected_rays_dir.dot(N), 0., 1.)
            c = get_raycolor(reflected_ray, scene)* N_dot_L/pdf_val / np.pi
            color += diff_color * c
            return color
            
        # TODO: Stop tracing if the recursion depth exceeds the maximum depth.
        else:
            return color