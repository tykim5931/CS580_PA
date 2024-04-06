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
            # nudged = hit.point + N * 0.000001  # M nudged to avoid itself

            N_20 = N.repeat(self.diffuse_rays)
            hit_20 = hit.point.repeat(self.diffuse_rays)
            nudged_20 = hit_20 + N_20 * .000001
            ray_n_20 = ray.n.repeat(self.diffuse_rays)
            
            pdf = cosine_pdf(nudged_20.shape()[0], N_20)
            reflected_rays_dir = pdf.generate()
            pdf_val = pdf.value(reflected_rays_dir)
            
            
            # pdf2 = spherical_caps_pdf(nudged_20.shape()[0], nudged_20, scene.importance_sampled_list)
            # s_pdf = None
            # if scene.importance_sampled_list == []:
            #     s_pdf = pdf
            # else:
            #     s_pdf = mixed_pdf(nudged_20.shape()[0], pdf, pdf2, self.ambient_weight)
            # reflected_rays_dir = s_pdf.generate()
            # pdf_val = s_pdf.value(reflected_rays_dir)
            
            reflected_ray = Ray(
                nudged_20,
                reflected_rays_dir,
                ray.depth + 1,
                ray.n,
                ray.reflections + 1,
                ray.transmissions,
                ray.diffuse_reflections + 1
            )
            N_dot_L = np.clip(reflected_rays_dir.dot(N_20), 0., 1.)
            
            c = get_raycolor(reflected_ray, scene)/pdf_val * N_dot_L
            mean_c_sample = c.reshape(N.shape()[0], self.diffuse_rays).mean(1)
            color += diff_color / np.pi * mean_c_sample
            return color
            
        # TODO: If the ray intersected with diffuse material more than once,
        # we generate only one secondary ray.
        elif ray.diffuse_reflections < self.max_diffuse_reflections:
            # raise NotImplementedError("TODO")
            nudged = hit.point + N * 0.000001  # M nudged to avoid itself
            
            pdf = cosine_pdf(nudged.shape()[0], N)
            reflected_rays_dir = pdf.generate()
            pdf_val = pdf.value(reflected_rays_dir)
            
            
            # pdf2 = spherical_caps_pdf(nudged.shape()[0], nudged, scene.importance_sampled_list)
            # s_pdf = None
            # if scene.importance_sampled_list == []:
            #     s_pdf = pdf
            # else:
            #     s_pdf = mixed_pdf(nudged.shape()[0], pdf, pdf2, self.ambient_weight)
            # reflected_rays_dir = s_pdf.generate()
            # pdf_val = s_pdf.value(reflected_rays_dir)
            
            
            reflected_ray = Ray(
                nudged,
                reflected_rays_dir,
                ray.depth + 1,
                ray.n,
                ray.reflections + 1,
                ray.transmissions,
                ray.diffuse_reflections + 1
            )
            N_dot_L = np.clip(reflected_rays_dir.dot(N), 0, 1)
            
            c = get_raycolor(reflected_ray, scene)/pdf_val * N_dot_L
            color += diff_color / np.pi * c
            return color
            
        # TODO: Stop tracing if the recursion depth exceeds the maximum depth.
        else:
            return color
