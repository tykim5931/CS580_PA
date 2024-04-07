from ..utils.constants import *
from ..utils.vector3 import vec3, rgb, extract
from functools import reduce as reduce
from ..ray import Ray, get_raycolor
from .. import lights
import numpy as np
from . import Material


class Refractive(Material):
    def __init__(self, n, **kwargs):
        super().__init__(**kwargs)

        self.n = n  # index of refraction

        # Instead of defining a index of refraction (n) for each wavelenght (computationally expensive) we aproximate defining the index of refraction
        # using a vec3 for red = 630 nm, green 555 nm, blue 475 nm, the most sensitive wavelenghts of human eye.

        # Index a refraction is a complex number.
        # The real part is involved in how much light is reflected and model refraction direction via Snell Law.
        # The imaginary part of n is involved in how much light is reflected and absorbed. For non-transparent materials like metals is usually between (0.1j,3j)
        # and for transparent materials like glass is  usually between (0.j , 1e-7j)

    def get_color(self, scene, ray, hit):
        """
        Computes the color of refractive surface intersected by the given ray.

        Args:
        - scene: A Scene object containing the list of objects in the scene.
        - ray: A Ray object containing the origin, direction, and other information of the rays.
        - hit: A Hit object containing the information of the ray-surface intersection.

        Returns:
        - A vec3 object containing the color of the diffuse surface intersected by the given ray.
        """

        hit.point = ray.origin + ray.dir * hit.distance  # intersection point
        N = hit.material.get_Normal(hit)  # normal

        color = rgb(0.0, 0.0, 0.0)

        V = ray.dir * -1.0  # direction to ray origin
        nudged = hit.point + N * 0.000001  # M nudged to avoid itself
        
        # compute reflection and refraction
        # NOTE: Refer to the following material for the formulation.
        # https://graphics.stanford.edu/courses/cs148-10-summer/docs/2006--degreve--reflection_refraction.pdf
        if ray.depth < hit.surface.max_ray_depth:
            n1 = ray.n
            n2 = vec3.where(hit.orientation == UPWARDS, self.n, scene.n)

            n1_div_n2 = vec3.real(n1) / vec3.real(n2)
            cosθi = V.dot(N)
            sin2θt = (n1_div_n2) ** 2 * (1.0 - cosθi**2)

            # TODO: Compute complete fresnel term
            cosθt = vec3.sqrt(1. - n1/n2 ** 2 * (1.0 - cosθi**2)) # for reflection we need to include the imag term
            rs = (n1*cosθi - n2*cosθt) / (n1*cosθi + n2*cosθt)
            rp = (n1*cosθt - n2*cosθi) / (n1*cosθt + n2*cosθi)
            F = np.abs((rs**2 + rp**2) / 2.)
            
            # # TODO: Add the contribution of the reflected ray
            # # color += ...  # the color of the reflected ray
            # r = i+ 2 * cosθi * n
            r_dir = (ray.dir + N * 2. * cosθi).normalize()
            reflected_ray = Ray(
                nudged,
                r_dir,
                ray.depth + 1,
                n1,
                ray.reflections + 1,
                ray.transmissions,
                ray.diffuse_reflections
            )
            color += get_raycolor(reflected_ray, scene) * F

            # # TODO: Compute refraction
            # # color += ... # the color of the refracted ray
            cond = sin2θt.average()<=1
            if np.any(cond):
                # we have candidate to sum up as refraction.
                nudged = hit.point - N * 0.000001  # M nudged to avoid itself
                t_dir = (
                    ray.dir * n1_div_n2 +  N * (
                        n1_div_n2 * cosθi - np.sqrt(1-sin2θt.clip(0.,1.))
                    )
                ).normalize()   # we use only real part for refraction direction
                refracted_ray = Ray(
                    nudged,
                    t_dir,
                    ray.depth + 1,
                    n2,
                    ray.reflections,
                    ray.transmissions + 1,
                    ray.diffuse_reflections
                )
                T = (1. - F)
                color += get_raycolor(refracted_ray, scene) * T

            # # TODO: Compute absorption effect
            extinction_coeff = vec3.imag(ray.n) # index of refraction.
            absorption_coefficient = extinction_coeff * 4. * np.pi / vec3(630,550,475)
            concentration = 1e9
            color *= vec3.exp(-absorption_coefficient * concentration * hit.distance)  # the absorption effect
            # color = color *vec3.exp(-2.*vec3.imag(ray.n)*2.*np.pi/vec3(630,550,475) * 1e9* hit.distance)
        return color

