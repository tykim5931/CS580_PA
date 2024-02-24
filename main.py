"""
main.py

A script for running the ray tracer.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from jaxtyping import Shaped, jaxtyped
from typeguard import typechecked
import tyro

from src import *


@dataclass
class Args:
    scene_type: Literal[
        "scene0",
        "scene0-skybox",
        "scene1-diffuse",
        "scene1-glossy",
        "scene1-refractive",
        "scene2-diffuse",
        "scene2-glossy",
        "scene2-refractive",
        "scene3",
        "scene4",
        "cornell_box",
    ] = "cornell_box"
    """The type of scene to render."""
    spp: int = 25
    """The number of samples per pixel."""
    show_pbar: bool = True
    """A flag for showing progress bar."""


@jaxtyped(typechecker=typechecked)
def main(args: Args) -> None:

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    print(f"Results will be stored under {str(out_dir)}")

    # setup scene
    scene = build_scene(args)

    # render
    img = scene.render(
        samples_per_pixel=args.spp,
        progress_bar=args.show_pbar,
    )

    # show and save
    img.save(out_dir / f"{args.scene_type}.png")


@jaxtyped(typechecker=typechecked)
def build_scene(args: Args) -> Scene:

    if args.scene_type == "cornell_box":
        scene = Scene(ambient_color=rgb(0.00, 0.00, 0.00))

        scene.add_Camera(
            screen_width=600,
            screen_height=600,
            look_from=vec3(278, 278, 800),
            look_at=vec3(278, 278, 0),
            focal_distance=1.0,
            field_of_view=40,
        )

        # materials
        green_diffuse = Diffuse(diff_color=rgb(0.12, 0.45, 0.15))
        red_diffuse = Diffuse(diff_color=rgb(0.65, 0.05, 0.05))
        white_diffuse = Diffuse(diff_color=rgb(0.73, 0.73, 0.73))
        emissive_white = Emissive(color=rgb(15.0, 15.0, 15.0))
        blue_glass = Refractive(n=vec3(1.5 + 0.05e-8j, 1.5 + 0.02e-8j, 1.5 + 0.0j))

        scene.add(
            Plane(
                material=emissive_white,
                center=vec3(213 + 130 / 2, 554, -227.0 - 105 / 2),
                width=130.0,
                height=105.0,
                u_axis=vec3(1.0, 0.0, 0),
                v_axis=vec3(0.0, 0, 1.0),
            ),
            importance_sampled=False,
            # importance_sampled=False,
        )

        scene.add(
            Plane(
                material=white_diffuse,
                center=vec3(555 / 2, 555 / 2, -555.0),
                width=555.0,
                height=555.0,
                u_axis=vec3(0.0, 1.0, 0),
                v_axis=vec3(1.0, 0, 0.0),
            )
        )

        scene.add(
            Plane(
                material=green_diffuse,
                center=vec3(-0.0, 555 / 2, -555 / 2),
                width=555.0,
                height=555.0,
                u_axis=vec3(0.0, 1.0, 0),
                v_axis=vec3(0.0, 0, -1.0),
            )
        )

        scene.add(
            Plane(
                material=red_diffuse,
                center=vec3(555.0, 555 / 2, -555 / 2),
                width=555.0,
                height=555.0,
                u_axis=vec3(0.0, 1.0, 0),
                v_axis=vec3(0.0, 0, -1.0),
            )
        )

        scene.add(
            Plane(
                material=white_diffuse,
                center=vec3(555 / 2, 555, -555 / 2),
                width=555.0,
                height=555.0,
                u_axis=vec3(1.0, 0.0, 0),
                v_axis=vec3(0.0, 0, -1.0),
            )
        )

        scene.add(
            Plane(
                material=white_diffuse,
                center=vec3(555 / 2, 0.0, -555 / 2),
                width=555.0,
                height=555.0,
                u_axis=vec3(1.0, 0.0, 0),
                v_axis=vec3(0.0, 0, -1.0),
            )
        )

        cb = Cuboid(
            material=white_diffuse,
            center=vec3(182.5, 165, -285 - 160 / 2),
            width=165,
            height=165 * 2,
            length=165,
            shadow=False,
        )
        cb.rotate(θ=15, u=vec3(0, 1, 0))
        scene.add(cb)

        scene.add(
            Sphere(
                material=blue_glass,
                center=vec3(370.5, 165 / 2, -65 - 185 / 2),
                radius=165 / 2,
                shadow=False,
                max_ray_depth=3,
            ),
            # importance_sampled=True,
            importance_sampled=False,
        )

        return scene
    elif args.scene_type == "scene0":
        # define materials to use
        floor = Glossy(
            diff_color=image("checkered_floor.png", repeat=2.0),
            roughness=0.2,
            spec_coeff=0.3,
            diff_coeff=0.7,
            n=vec3(2.2, 2.2, 2.2),
        )  # n = index of refraction

        # Set Scene
        scene = Scene()
        scene.add_Camera(
            look_from=vec3(0.0, 0.25, 1.0),
            look_at=vec3(0.0, 0.25, -3.0),
            screen_width=400,
            screen_height=300,
        )

        scene.add_DirectionalLight(Ldir=vec3(0.0, 0.5, 0.5), color=rgb(0.5, 0.5, 0.5))

        scene.add(
            Plane(
                material=floor,
                center=vec3(0, -0.5, -3.0),
                width=6.0,
                height=6.0,
                u_axis=vec3(1.0, 0, 0),
                v_axis=vec3(0, 0, -1.0),
                max_ray_depth=5,
            )
        )

        # see src/backgrounds
        return scene
    elif args.scene_type == "scene0-skybox":
        # define materials to use
        floor = Glossy(
            diff_color=image("checkered_floor.png", repeat=2.0),
            roughness=0.2,
            spec_coeff=0.3,
            diff_coeff=0.7,
            n=vec3(2.2, 2.2, 2.2),
        )  # n = index of refraction

        # Set Scene
        scene = Scene()
        scene.add_Camera(
            look_from=vec3(0.0, 0.25, 1.0),
            look_at=vec3(0.0, 0.25, -3.0),
            screen_width=400,
            screen_height=300,
        )

        scene.add_DirectionalLight(Ldir=vec3(0.0, 0.5, 0.5), color=rgb(0.5, 0.5, 0.5))

        scene.add(
            Plane(
                material=floor,
                center=vec3(0, -0.5, -3.0),
                width=6.0,
                height=6.0,
                u_axis=vec3(1.0, 0, 0),
                v_axis=vec3(0, 0, -1.0),
                max_ray_depth=5,
            )
        )

        # see src/backgrounds
        scene.add_Background("stormydays.png")
        return scene
    elif args.scene_type == "scene1-diffuse":
        # define materials to use
        floor = Glossy(
            diff_color=image("checkered_floor.png", repeat=2.0),
            roughness=0.2,
            spec_coeff=0.3,
            diff_coeff=0.7,
            n=vec3(2.2, 2.2, 2.2),
        )  # n = index of refraction
        green_diffuse = Diffuse(diff_color=rgb(0.0, 0.1, 0.0))

        # Set Scene
        scene = Scene()
        scene.add_Camera(
            look_from=vec3(0.0, 0.25, 1.0),
            look_at=vec3(0.0, 0.25, -3.0),
            screen_width=400,
            screen_height=300,
        )

        scene.add_DirectionalLight(Ldir=vec3(0.0, 0.5, 0.5), color=rgb(0.5, 0.5, 0.5))

        scene.add(
            Plane(
                material=floor,
                center=vec3(0, -0.5, -3.0),
                width=6.0,
                height=6.0,
                u_axis=vec3(1.0, 0, 0),
                v_axis=vec3(0, 0, -1.0),
                max_ray_depth=5,
            )
        )

        cb = Cuboid(
            material=green_diffuse,
            center=vec3(0.00, 0.0001, -0.8),
            width=0.9,
            height=1.0,
            length=0.4,
            shadow=False,
            max_ray_depth=5,
        )
        cb.rotate(θ=30, u=vec3(0, 1, 0))
        scene.add(cb)

        # see src/backgrounds
        scene.add_Background("stormydays.png")
        return scene
    elif args.scene_type == "scene1-glossy":
        # define materials to use
        floor = Glossy(
            diff_color=image("checkered_floor.png", repeat=2.0),
            roughness=0.2,
            spec_coeff=0.3,
            diff_coeff=0.7,
            n=vec3(2.2, 2.2, 2.2),
        )  # n = index of refraction
        green_glossy = Glossy(
            diff_color=rgb(0.0, 0.5, 0.0),
            n=vec3(1.3 + 1.91j, 1.3 + 1.91j, 1.4 + 2.91j),
            roughness=0.2,
            spec_coeff=0.3,
            diff_coeff=0.8,
        )

        # Set Scene
        scene = Scene()
        scene.add_Camera(
            look_from=vec3(0.0, 0.25, 1.0),
            look_at=vec3(0.0, 0.25, -3.0),
            screen_width=400,
            screen_height=300,
        )

        scene.add_DirectionalLight(Ldir=vec3(0.0, 0.5, 0.5), color=rgb(0.5, 0.5, 0.5))

        scene.add(
            Plane(
                material=floor,
                center=vec3(0, -0.5, -3.0),
                width=6.0,
                height=6.0,
                u_axis=vec3(1.0, 0, 0),
                v_axis=vec3(0, 0, -1.0),
                max_ray_depth=5,
            )
        )

        cb = Cuboid(
            material=green_glossy,
            center=vec3(0.00, 0.0001, -0.8),
            width=0.9,
            height=1.0,
            length=0.4,
            shadow=False,
            max_ray_depth=5,
        )
        cb.rotate(θ=30, u=vec3(0, 1, 0))
        scene.add(cb)

        # see src/backgrounds
        scene.add_Background("stormydays.png")
        return scene
    elif args.scene_type == "scene1-refractive":
        # define materials to use
        floor = Glossy(
            diff_color=image("checkered_floor.png", repeat=2.0),
            roughness=0.2,
            spec_coeff=0.3,
            diff_coeff=0.7,
            n=vec3(2.2, 2.2, 2.2),
        )  # n = index of refraction
        green_glass = Refractive(n=vec3(1.5 + 4e-8j, 1.5 + 0.0j, 1.5 + 4e-8j))

        # Set Scene
        scene = Scene()
        scene.add_Camera(
            look_from=vec3(0.0, 0.25, 1.0),
            look_at=vec3(0.0, 0.25, -3.0),
            screen_width=400,
            screen_height=300,
        )

        scene.add_DirectionalLight(Ldir=vec3(0.0, 0.5, 0.5), color=rgb(0.5, 0.5, 0.5))

        scene.add(
            Plane(
                material=floor,
                center=vec3(0, -0.5, -3.0),
                width=6.0,
                height=6.0,
                u_axis=vec3(1.0, 0, 0),
                v_axis=vec3(0, 0, -1.0),
                max_ray_depth=5,
            )
        )

        cb = Cuboid(
            material=green_glass,
            center=vec3(0.00, 0.0001, -0.8),
            width=0.9,
            height=1.0,
            length=0.4,
            shadow=False,
            max_ray_depth=5,
        )
        cb.rotate(θ=30, u=vec3(0, 1, 0))
        scene.add(cb)

        # see src/backgrounds
        scene.add_Background("stormydays.png")
        return scene
    elif args.scene_type == "scene2-diffuse":
        green_diffuse = Diffuse(diff_color=rgb(0.0, 0.1, 0.0))

        floor = Glossy(
            diff_color=image("checkered_floor.png", repeat=80.0),
            n=vec3(1.2 + 0.3j, 1.2 + 0.3j, 1.1 + 0.3j),
            roughness=0.2,
            spec_coeff=0.3,
            diff_coeff=0.9,
        )

        # Set Scene
        scene = Scene(ambient_color=rgb(0.05, 0.05, 0.05))

        angle = -np.pi / 2 * 0.3
        scene.add_Camera(
            look_from=vec3(2.5 * np.sin(angle), 0.25, 2.5 * np.cos(angle) - 1.5),
            look_at=vec3(0.0, 0.25, -3.0),
            screen_width=400,
            screen_height=300,
        )

        scene.add_DirectionalLight(
            Ldir=vec3(0.52, 0.45, -0.5), color=rgb(0.15, 0.15, 0.15)
        )

        scene.add(
            Sphere(
                material=green_diffuse,
                center=vec3(-0.75, 0.1, -2.0),
                radius=0.6,
                max_ray_depth=3,
            )
        )

        scene.add(
            Plane(
                material=floor,
                center=vec3(0, -0.5, -3.0),
                width=120.0,
                height=120.0,
                u_axis=vec3(1.0, 0, 0),
                v_axis=vec3(0, 0, -1.0),
                max_ray_depth=3,
            )
        )

        # see src/backgrounds
        scene.add_Background("stormydays.png")
        return scene
    elif args.scene_type == "scene2-glossy":
        green_glossy = Glossy(
            diff_color=rgb(0.0, 0.5, 0.0),
            n=vec3(1.3 + 1.91j, 1.3 + 1.91j, 1.4 + 2.91j),
            roughness=0.2,
            spec_coeff=0.3,
            diff_coeff=0.8,
        )

        floor = Glossy(
            diff_color=image("checkered_floor.png", repeat=80.0),
            n=vec3(1.2 + 0.3j, 1.2 + 0.3j, 1.1 + 0.3j),
            roughness=0.2,
            spec_coeff=0.3,
            diff_coeff=0.9,
        )

        # Set Scene
        scene = Scene(ambient_color=rgb(0.05, 0.05, 0.05))

        angle = -np.pi / 2 * 0.3
        scene.add_Camera(
            look_from=vec3(2.5 * np.sin(angle), 0.25, 2.5 * np.cos(angle) - 1.5),
            look_at=vec3(0.0, 0.25, -3.0),
            screen_width=400,
            screen_height=300,
        )

        scene.add_DirectionalLight(
            Ldir=vec3(0.52, 0.45, -0.5), color=rgb(0.15, 0.15, 0.15)
        )

        scene.add(
            Sphere(
                material=green_glossy,
                center=vec3(-0.75, 0.1, -2.0),
                radius=0.6,
                max_ray_depth=3,
            )
        )

        scene.add(
            Plane(
                material=floor,
                center=vec3(0, -0.5, -3.0),
                width=120.0,
                height=120.0,
                u_axis=vec3(1.0, 0, 0),
                v_axis=vec3(0, 0, -1.0),
                max_ray_depth=3,
            )
        )

        # see src/backgrounds
        scene.add_Background("stormydays.png")
        return scene
    elif args.scene_type == "scene2-refractive":
        green_glass = Refractive(n=vec3(1.5 + 4e-8j, 1.5 + 0.0j, 1.5 + 4e-8j))

        floor = Glossy(
            diff_color=image("checkered_floor.png", repeat=80.0),
            n=vec3(1.2 + 0.3j, 1.2 + 0.3j, 1.1 + 0.3j),
            roughness=0.2,
            spec_coeff=0.3,
            diff_coeff=0.9,
        )

        # Set Scene
        scene = Scene(ambient_color=rgb(0.05, 0.05, 0.05))

        angle = -np.pi / 2 * 0.3
        scene.add_Camera(
            look_from=vec3(2.5 * np.sin(angle), 0.25, 2.5 * np.cos(angle) - 1.5),
            look_at=vec3(0.0, 0.25, -3.0),
            screen_width=400,
            screen_height=300,
        )

        scene.add_DirectionalLight(
            Ldir=vec3(0.52, 0.45, -0.5), color=rgb(0.15, 0.15, 0.15)
        )

        scene.add(
            Sphere(
                material=green_glass,
                center=vec3(-0.75, 0.1, -2.0),
                radius=0.6,
                max_ray_depth=3,
            )
        )

        scene.add(
            Plane(
                material=floor,
                center=vec3(0, -0.5, -3.0),
                width=120.0,
                height=120.0,
                u_axis=vec3(1.0, 0, 0),
                v_axis=vec3(0, 0, -1.0),
                max_ray_depth=3,
            )
        )

        # see src/backgrounds
        scene.add_Background("stormydays.png")
        return scene
    elif args.scene_type == "scene3":
        gold_metal = Glossy(
            diff_color=rgb(1.0, 0.572, 0.184),
            n=vec3(0.15 + 3.58j, 0.4 + 2.37j, 1.54 + 1.91j),
            roughness=0.0,
            spec_coeff=0.2,
            diff_coeff=0.8,
        )
        bluish_metal = Glossy(
            diff_color=rgb(0.0, 0, 0.1),
            n=vec3(1.3 + 1.91j, 1.3 + 1.91j, 1.4 + 2.91j),
            roughness=0.2,
            spec_coeff=0.5,
            diff_coeff=0.3,
        )

        floor = Glossy(
            diff_color=image("checkered_floor.png", repeat=80.0),
            n=vec3(1.2 + 0.3j, 1.2 + 0.3j, 1.1 + 0.3j),
            roughness=0.2,
            spec_coeff=0.3,
            diff_coeff=0.9,
        )

        # Set Scene
        scene = Scene(ambient_color=rgb(0.05, 0.05, 0.05))

        angle = -np.pi / 2 * 0.3
        scene.add_Camera(
            look_from=vec3(2.5 * np.sin(angle), 0.25, 2.5 * np.cos(angle) - 1.5),
            look_at=vec3(0.0, 0.25, -3.0),
            screen_width=400,
            screen_height=300,
        )

        scene.add_DirectionalLight(
            Ldir=vec3(0.52, 0.45, -0.5), color=rgb(0.15, 0.15, 0.15)
        )

        scene.add(
            Sphere(
                material=gold_metal,
                center=vec3(-0.75, 0.1, -3.0),
                radius=0.6,
                max_ray_depth=3,
            )
        )
        scene.add(
            Sphere(
                material=bluish_metal,
                center=vec3(1.25, 0.1, -3.0),
                radius=0.6,
                max_ray_depth=3,
            )
        )

        scene.add(
            Plane(
                material=floor,
                center=vec3(0, -0.5, -3.0),
                width=120.0,
                height=120.0,
                u_axis=vec3(1.0, 0, 0),
                v_axis=vec3(0, 0, -1.0),
                max_ray_depth=3,
            )
        )

        # see src/backgrounds
        scene.add_Background("stormydays.png")
        return scene
    elif args.scene_type == "scene4":
        # materials
        blue_glass = Refractive(
            n=vec3(1.5 + 4e-8j, 1.5 + 4e-8j, 1.5 + 0.0j)
        )  # n = index of refraction
        green_glass = Refractive(n=vec3(1.5 + 4e-8j, 1.5 + 0.0j, 1.5 + 4e-8j))
        red_glass = Refractive(n=vec3(1.5 + 0.0j, 1.5 + 5e-8j, 1.5 + 5e-8j))

        floor = Glossy(
            diff_color=image("checkered_floor.png", repeat=80.0),
            n=vec3(1.2 + 0.3j, 1.2 + 0.3j, 1.1 + 0.3j),
            roughness=0.2,
            spec_coeff=0.3,
            diff_coeff=0.9,
        )

        # Set Scene
        scene = Scene(ambient_color=rgb(0.05, 0.05, 0.05))

        angle = np.pi / 2 * 0.3
        scene.add_Camera(
            look_from=vec3(2.5 * np.sin(angle), 0.25, 2.5 * np.cos(angle) - 1.5),
            look_at=vec3(0.0, 0.25, -1.5),
            screen_width=400,
            screen_height=300,
        )

        scene.add_DirectionalLight(
            Ldir=vec3(0.52, 0.45, -0.5), color=rgb(0.15, 0.15, 0.15)
        )

        scene.add(
            Sphere(
                material=blue_glass,
                center=vec3(-1.2, 0.0, -1.5),
                radius=0.5,
                shadow=False,
                max_ray_depth=3,
            )
        )
        scene.add(
            Sphere(
                material=green_glass,
                center=vec3(0.0, 0.0, -1.5),
                radius=0.5,
                shadow=False,
                max_ray_depth=3,
            )
        )
        scene.add(
            Sphere(
                material=red_glass,
                center=vec3(1.2, 0.0, -1.5),
                radius=0.5,
                shadow=False,
                max_ray_depth=3,
            )
        )

        scene.add(
            Plane(
                material=floor,
                center=vec3(0, -0.5, -3.0),
                width=120.0,
                height=120.0,
                u_axis=vec3(1.0, 0, 0),
                v_axis=vec3(0, 0, -1.0),
                max_ray_depth=3,
            )
        )

        # see src/backgrounds
        scene.add_Background("miramar.jpeg")
        return scene

if __name__ == "__main__":
    main(tyro.cli(Args))
