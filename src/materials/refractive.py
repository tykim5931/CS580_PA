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
            F = None  # the computed fresnel term
            
            # TODO: Add the contribution of the reflected ray
            # color += ...  # the color of the reflected ray

            # TODO: Compute refraction
            # color += ... # the color of the refracted ray

            # TODO: Compute absorption effect
            # color *= ...  # the absorption effect
        return color
