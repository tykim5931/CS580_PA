from jaxtyping import Shaped

import numpy as np
from ..utils.constants import *
from ..utils.vector3 import vec3
from ..geometry import Primitive, Collider


class Plane(Primitive):
    def __init__(
        self,
        center,
        material,
        width,
        height,
        u_axis,
        v_axis,
        max_ray_depth=5,
        shadow=True,
    ):
        super().__init__(center, material, max_ray_depth, shadow=shadow)
        self.collider_list += [
            Plane_Collider(
                assigned_primitive=self,
                center=center,
                u_axis=u_axis,
                v_axis=v_axis,
                w=width / 2,
                h=height / 2,
            )
        ]
        self.width = width
        self.height = height
        self.bounded_sphere_radius = np.sqrt((width / 2) ** 2 + (height / 2) ** 2)

    def get_uv(self, hit):
        return hit.collider.get_uv(hit)


class Plane_Collider(Collider):
    def __init__(self, u_axis, v_axis, w, h, uv_shift=(0.0, 0.0), **kwargs):
        super().__init__(**kwargs)
        self.normal = u_axis.cross(v_axis).normalize()

        self.w = w
        self.h = h
        self.u_axis = u_axis
        self.v_axis = v_axis
        self.uv_shift = uv_shift
        self.inverse_basis_matrix = np.array(
            [
                [self.u_axis.x, self.v_axis.x, self.normal.x],
                [self.u_axis.y, self.v_axis.y, self.normal.y],
                [self.u_axis.z, self.v_axis.z, self.normal.z],
            ]
        )
        self.basis_matrix = self.inverse_basis_matrix.T

    def intersect(
        self,
        O: vec3,
        D: vec3,
    ) -> Shaped[np.ndarray, "2 num_ray"]:
        """
        Computes the intersection of a ray with the plane.

        Args:
        - O: A vec3 object representing the origins of the rays.
        - D: A vec3 object representing the directions of the rays.

        Returns:
        - A NumPy array of shape (2, num_ray) containing
          the ray distance to the intersection point (row 0)
          and the orientation of the intersection point (row 1).
        """
        # raise NotImplementedError("TODO")
        N = self.normal

        NdotD = N.dot(D)
        NdotD = np.where(NdotD == 0.0, NdotD + 0.0001, NdotD)  # avoid zero division

        NdotC_O = N.dot(self.center - O)
        d = D * NdotC_O / NdotD
        M = O + d  # intersection point
        dis = d.length()
        M_C = M - self.center
        
        # plane collider => intersection point is inside plane boundary & frontal way
        # should be absolute, since if negative, it will be always count to be hitted!!! > cornell box problem
        hit_inside = (
            (np.abs(self.u_axis.dot(M_C)) <= self.w)
            & (np.abs(self.v_axis.dot(M_C)) <= self.h)
            & (NdotC_O * NdotD > 0)
        )
        hit_UPWARDS = NdotD < 0
        hit_UPDOWN = np.logical_not(hit_UPWARDS)

        pred1 = hit_inside & hit_UPWARDS
        pred2 = hit_inside & hit_UPDOWN
        pred3 = True
        return np.select(
            [pred1, pred2, pred3],
            [
                [dis, np.tile(UPWARDS, dis.shape)],
                [dis, np.tile(UPDOWN, dis.shape)],
                FARAWAY,
            ],
        )

    def rotate(self, M, center):
        self.u_axis = self.u_axis.matmul(M)
        self.v_axis = self.v_axis.matmul(M)
        self.normal = self.normal.matmul(M)
        self.center = center + (self.center - center).matmul(M)

    def get_uv(self, hit):
        M_C = hit.point - self.center
        u = (self.u_axis.dot(M_C) / self.w + 1) / 2 + self.uv_shift[0]
        v = (self.v_axis.dot(M_C) / self.h + 1) / 2 + self.uv_shift[1]
        return u, v

    def get_Normal(self, hit):
        return self.normal
