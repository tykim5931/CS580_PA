from .utils.constants import *
from .utils.vector3 import vec3, extract, rgb
import numpy as np
from functools import reduce as reduce


class Ray:
    """Info of the ray and the media it's travelling"""

    def __init__(
        self, origin, dir, depth, n, reflections, transmissions, diffuse_reflections
    ):

        self.origin = origin  # the point where the ray comes from
        self.dir = dir  # direction of the ray
        self.depth = depth  # ray_depth is the number of the refrections + transmissions/refractions, starting at zero for camera rays
        self.n = n  # ray_n is the index of refraction of the media in which the ray is travelling

        # Instead of defining a index of refraction (n) for each wavelenght (computationally expensive) we aproximate defining the index of refraction
        # using a vec3 for red = 630 nm, green 555 nm, blue 475 nm, the most sensitive wavelenghts of human eye.

        # Index a refraction is a complex number.
        # The real part is involved in how much light is reflected and model refraction direction via Snell Law.
        # The imaginary part of n is involved in how much light is reflected and absorbed. For non-transparent materials like metals is usually between (0.1j,3j)
        # and for transparent materials like glass is  usually between (0.j , 1e-7j)

        self.reflections = reflections  # reflections is the number of the refrections, starting at zero for camera rays
        self.transmissions = transmissions  # transmissions is the number of the transmissions/refractions, starting at zero for camera rays
        self.diffuse_reflections = diffuse_reflections  # reflections is the number of the refrections, starting at zero for camera rays

    def extract(self, hit_check):
        return Ray(
            self.origin.extract(hit_check),
            self.dir.extract(hit_check),
            self.depth,
            self.n.extract(hit_check),
            self.reflections,
            self.transmissions,
            self.diffuse_reflections,
        )


class Hit:
    """Info of the ray-surface intersection"""

    def __init__(self, distance, orientation, material, collider, surface):
        self.distance = distance
        self.orientation = orientation
        self.material = material
        self.collider = collider
        self.surface = surface
        self.u = None
        self.v = None
        self.N = None
        self.point = None

    def get_uv(self):
        if self.u is None:  # this is for prevent multiple computations of u,v
            self.u, self.v = self.collider.assigned_primitive.get_uv(self)
        return self.u, self.v

    def get_normal(self):
        if self.N is None:  # this is for prevent multiple computations of normal
            self.N = self.collider.get_N(self)
        return self.N


def get_raycolor(ray, scene) -> vec3:
    """
    Computes the color of the ray after it intersects with the scene.

    Args:
    - ray: A Ray object containing the origin, direction, and other information of the rays.
    - scene: A Scene object containing the list of objects in the scene.

    Returns:
    - A vec3 object containing the color of the ray after it intersects with the scene.
    """

    # raise NotImplementedError("TODO")
    # ray: dir, n, origin, reflections, transmissions
    # scene:light_list ambient_color, camera, collider_list, importance_sampled list(empty), n, scene_primitives, shadowed_collider_list

    # performing a ray-object intersection check 
    # and initiating recursive ray tracing for reflection and refraction, 
    # depending on the material characteristic of the surface.

    intersections = [c.intersect(ray.origin, ray.dir) for c in scene.collider_list]
    distances, orientations = list(zip(*intersections))
    
    # find first object ray is intersecting
    first_hit_distance = reduce(np.minimum, distances)
    
    # initiate color to accumulate
    color = rgb(0., 0., 0.)

    # for all objects collided in scene
    for (d, o, c) in zip(distances, orientations, scene.collider_list):
        # mask out rays that it actually collides & which is first collision
        hit_mask = (first_hit_distance!=FARAWAY) & (d==first_hit_distance)
        if not np.any(hit_mask):
            pass

        first_hit = Hit(extract(hit_mask,d), extract(hit_mask,o), c.assigned_primitive.material, c, c.assigned_primitive)
        cumulated_color = c.assigned_primitive.material.get_color(scene, ray.extract(hit_mask), first_hit) #recursively get material & color
        color += cumulated_color.place(hit_mask)
        
    return color

def get_distances(
    ray, scene
):  # Used for debugging ray-surface collisions. Return a grey map of objects distances.

    inters = [s.intersect(ray.origin, ray.dir) for s in scene.collider_list]
    distances, hit_orientation = zip(*inters)
    # get the shortest distance collision
    nearest = reduce(np.minimum, distances)

    max_r_distance = 10
    r_distance = np.where(nearest <= max_r_distance, nearest, max_r_distance)
    norm_r_distance = r_distance / max_r_distance
    return rgb(norm_r_distance, norm_r_distance, norm_r_distance)
