import numpy as np
from ..utils.constants import *
from ..utils.vector3 import vec3
from ..geometry import Primitive, Collider


class Cuboid(Primitive):
    def __init__(
        self, center, material, width, height, length, max_ray_depth=5, shadow=True
    ):
        super().__init__(center, material, max_ray_depth, shadow=shadow)
        self.width = width
        self.height = height
        self.length = length
        self.bounded_sphere_radius = np.sqrt(
            (self.width / 2) ** 2 + (self.height / 2) ** 2 + (self.length / 2) ** 2
        )

        self.collider_list += [
            Cuboid_Collider(
                assigned_primitive=self,
                center=center,
                width=width,
                height=height,
                length=length,
            )
        ]

    def get_uv(self, hit):
        u, v = hit.collider.get_uv(hit)
        u, v = u / 4, v / 3
        return u, v


"""        
        This was the old approach, but remplaced by a Box collider that is more efficient
        #we model a cuboid as six planes
        
        
        #BOTTOM                                                                                                                                       #BOTTOM
        self.collider_list += [Plane_Collider(assigned_primitive = self, center = center + vec3(0.0,-h, 0.0), u_axis = vec3(1.0, 0.0, 0.0), v_axis = vec3(0.0, 0.0, 1.0), w = w, h = l, uv_shift = (1,0))]
        #TOP                                                                                                                                       #TOP
        self.collider_list += [Plane_Collider(assigned_primitive = self, center = center + vec3(0.0,h, 0.0), u_axis = vec3(1.0, 0.0, 0.0), v_axis = vec3(0.0, 0.0, -1.0), w = w, h = l, uv_shift= (1,2))]
        #RIGHT                                                                                                                                       #RIGHT
        self.collider_list += [Plane_Collider(assigned_primitive = self, center = center + vec3(w,0.0, 0.0), u_axis = vec3(0.0, 0.0,  -1.0), v_axis = vec3(0.0, 1.0, 0.0), w = l, h = h, uv_shift= (2,1))]
        #LEFT                                                                                                                                       #LEFT
        self.collider_list += [Plane_Collider(assigned_primitive = self, center = center + vec3(-w,0.0, 0.0), u_axis = vec3(0.0, 0.0,  1.0), v_axis = vec3(0.0, 1.0, 0.0), w = l, h = h, uv_shift= (0,1))]
        #FRONT                                                                                                                                       #FRONT
        self.collider_list += [Plane_Collider(assigned_primitive = self, center = center + vec3(0,0, l), u_axis = vec3(1.0, 0.0, 0.0), v_axis = vec3(0.0, 1.0, 0.0), w = w, h = h, uv_shift= (1,1))]
        #BACK                                                                                                                                       #BACK
        self.collider_list += [Plane_Collider(assigned_primitive = self, center = center + vec3(0,0, -l), u_axis = vec3(-1.0, 0.0, 0.0), v_axis = vec3(0.0, 1.0, 0.0), w = w, h = h, uv_shift= (3,1))]
"""


class Cuboid_Collider(Collider):
    def __init__(self, width, height, length, **kwargs):
        super().__init__(**kwargs)

        self.lb = self.center - vec3(width / 2, height / 2, length / 2)
        self.rt = self.center + vec3(width / 2, height / 2, length / 2)

        self.lb_local_basis = self.lb
        self.rt_local_basis = self.rt

        self.width = width
        self.height = height
        self.length = length

        # basis vectors
        self.ax_w = vec3(1.0, 0.0, 0.0)
        self.ax_h = vec3(0.0, 1.0, 0.0)
        self.ax_l = vec3(0.0, 0.0, 1.0)

        self.inverse_basis_matrix = np.array(
            [
                [self.ax_w.x, self.ax_h.x, self.ax_l.x],
                [self.ax_w.y, self.ax_h.y, self.ax_l.y],
                [self.ax_w.z, self.ax_h.z, self.ax_l.z],
            ]
        )

        self.basis_matrix = self.inverse_basis_matrix.T

    def rotate(self, M, center):
        self.ax_w = self.ax_w.matmul(M)
        self.ax_h = self.ax_h.matmul(M)
        self.ax_l = self.ax_l.matmul(M)

        self.inverse_basis_matrix = np.array(
            [
                [self.ax_w.x, self.ax_h.x, self.ax_l.x],
                [self.ax_w.y, self.ax_h.y, self.ax_l.y],
                [self.ax_w.z, self.ax_h.z, self.ax_l.z],
            ]
        )

        self.basis_matrix = self.inverse_basis_matrix.T

        self.lb = center + (self.lb - center).matmul(M)
        self.rt = center + (self.rt - center).matmul(M)

        self.lb_local_basis = self.lb.matmul(self.basis_matrix)
        self.rt_local_basis = self.rt.matmul(self.basis_matrix)

    def intersect(self, O, D):
        """
        Computes the intersection of a ray with the cuboid.

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

        P = (hit.point - self.center).matmul(self.basis_matrix)
        absP = vec3(1.0 / self.width, 1.0 / self.height, 1.0 / self.length) * np.abs(P)
        Pmax = np.maximum(np.maximum(absP.x, absP.y), absP.z)
        P.x = np.where(Pmax == absP.x, np.sign(P.x), 0.0)
        P.y = np.where(Pmax == absP.y, np.sign(P.y), 0.0)
        P.z = np.where(Pmax == absP.z, np.sign(P.z), 0.0)

        return P.matmul(self.inverse_basis_matrix)

    def get_uv(self, hit):
        hit.N = self.get_Normal(hit)
        M_C = hit.point - self.center

        BOTTOM = hit.N == vec3(0.0, -1.0, 0.0)
        TOP = hit.N == vec3(0.0, 1.0, 0.0)
        RIGHT = hit.N == vec3(1.0, 0.0, 0.0)
        LEFT = hit.N == vec3(-1.0, 0.0, 0.0)
        FRONT = hit.N == vec3(0.0, 0.0, 1.0)
        BACK = hit.N == vec3(0.0, 0.0, -1.0)

        # 0.985 to avoid corners
        u = np.select(
            [BOTTOM, TOP, RIGHT, LEFT, FRONT, BACK],
            [
                ((self.ax_w.dot(M_C) / self.width * 2 * 0.985 + 1) / 2 + 1),
                ((self.ax_w.dot(M_C) / self.width * 2 * 0.985 + 1) / 2 + 1),
                ((self.ax_l.dot(M_C) / self.width * 2 * 0.985 + 1) / 2 + 2),
                (((self.ax_l * -1).dot(M_C) / self.width * 2 * 0.985 + 1) / 2 + 0),
                (((self.ax_w * -1).dot(M_C) / self.width * 2 * 0.985 + 1) / 2 + 3),
                ((self.ax_w.dot(M_C) / self.width * 2 * 0.985 + 1) / 2 + 1),
            ],
        )
        v = np.select(
            [BOTTOM, TOP, RIGHT, LEFT, FRONT, BACK],
            [
                (((self.ax_l * -1).dot(M_C) / self.width * 2 * 0.985 + 1) / 2 + 0),
                ((self.ax_l.dot(M_C) / self.width * 2 * 0.985 + 1) / 2 + 2),
                ((self.ax_h.dot(M_C) / self.width * 2 * 0.985 + 1) / 2 + 1),
                (((self.ax_h).dot(M_C) / self.width * 2 * 0.985 + 1) / 2 + 1),
                (((self.ax_h).dot(M_C) / self.width * 2 * 0.985 + 1) / 2 + 1),
                ((self.ax_h.dot(M_C) / self.width * 2 * 0.985 + 1) / 2 + 1),
            ],
        )
        return u, v
