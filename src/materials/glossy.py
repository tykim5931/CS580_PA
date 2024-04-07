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
                
                # microfacet BRDF
                # f_r(v,l) = fd + fs
                # fd = lo_d / pi <- normalized phong BRDF (lambert shading (diffuse))
                # fs = Frensel * normal distribution * geoemtry term / 4 n.dot(l) n.dot(v)
                NdotH = np.clip(N.dot(H), 0.0, 1.)
                VdotH = np.clip(V.dot(H), 0.0, 1.)
                NdotV = np.clip(N.dot(V), 0.0, 1.)
                
                # F: Fresnel: Schlick's Approximation
                F0 = np.abs((ray.n-self.n)/(ray.n+self.n))**2
                F = F0 + (1. - F0) * (1.- VdotH)**5

                # D: normal distribution
                power = 2./(self.roughness**2.) - 2.    # bling-phong
                D_blinn = np.power(NdotH, power) /np.pi/self.roughness**2
                # D_ggx = self.roughness**2 / (np.pi * ((NdotH)**2 * (self.roughness**2-1.) + 1.)**2)
    
                # G: geometry term (schlick's approximation)
                G = NdotV / (NdotV * (1 - self.roughness / 2.) + self.roughness / 2.)
                # G: fraction of microfacets which are neither occluded or shadowed
                # number from 0 to 1 which indicates the proportion of light that is not blocked by either of these effects
                # G = np.minimum(1., np.minimum(2.*NdotH*NdotV/VdotH, 2.*NdotH*NdotL/VdotH)
                
                color_rs = F * G * D_blinn * self.spec_coeff / 4. / np.clip(NdotV * NdotL, 0.001, 1.) # avoid zerodivision
                color += color_rs * lv * seelight

        # Reflection
        if ray.depth < hit.surface.max_ray_depth:
            # TODO: Compute color contribution from the reflected ray
            # raise NotImplementedError("TODO")
            # get color of reflected ray * weight down according to reflection ratio.
            
            # Fresnel Reflection (Schlick’s approximation)
            # theta: angle between the direction from which the incident light is coming & normal of the interface between two media -> NdotV
            # n1, n2 indices of refraction of the two media. (here, air & surface)
            F0 = np.abs((scene.n - self.n)/(scene.n  + self.n))**2
            NdotV = np.clip(N.dot(V), 0.0, 1.)
            F = F0 + (1. - F0) * (1.- NdotV)**5
            
            # Get color of reflected ray
            # R_r = R_i -2H(R_i*N)
            out_dir = (ray.dir - N * 2. * ray.dir.dot(N)).normalize()
            reflected_ray = Ray(
                nudged,  # M nudged to avoid itself
                out_dir,
                ray.depth + 1,
                ray.n,
                ray.reflections + 1,
                ray.transmissions,
                ray.diffuse_reflections
            )
            reflected_ray_color = get_raycolor(reflected_ray, scene)
            color += F*reflected_ray_color

        return color
