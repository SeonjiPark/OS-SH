"""Shared COLMAP preprocessing for LLFF and Mip-NeRF 360."""

from argparse import ArgumentParser
from contextlib import closing
from pathlib import Path
import shutil
import sqlite3
import subprocess


def pipeline(scene, base_path, n_views, max_num_features, overwrite=False):
    import numpy as np

    scene_path = (Path(base_path).expanduser().resolve() / scene).resolve()
    if not scene_path.is_dir():
        raise FileNotFoundError(f"Scene directory does not exist: {scene_path}")
    if n_views < 1:
        raise ValueError("n_views must be positive")
    view_path = scene_path / f"{n_views}_views"
    if view_path.exists():
        if not overwrite:
            raise FileExistsError(f"{view_path} already exists; use --overwrite to regenerate it")
        shutil.rmtree(view_path)
    for name in ("created", "triangulated", "images"):
        (view_path / name).mkdir(parents=True, exist_ok=True)

    def colmap(command, *options):
        subprocess.run(["colmap", command, *map(str, options)], cwd=view_path, check=True)

    colmap("model_converter", "--input_path", "../sparse/0/",
           "--output_path", "../sparse/0/", "--output_type", "TXT")

    images = {}
    with (scene_path / "sparse/0/images.txt").open() as source:
        for line in source:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split(maxsplit=9)
            images[fields[9]] = fields[1:]
            next(source, None)  # Skip the corresponding 2D point observations.

    train_images = [name for index, name in enumerate(sorted(images)) if index % 8 != 0]
    if n_views > len(train_images):
        raise ValueError(f"Requested {n_views} views, but {scene} has {len(train_images)} training images")
    indices = {round(index) for index in np.linspace(0, len(train_images) - 1, n_views)}
    train_images = [name for index, name in enumerate(train_images) if index in indices]
    for name in train_images:
        destination = view_path / "images" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(scene_path / "images" / name, destination)

    shutil.copyfile(scene_path / "sparse/0/cameras.txt", view_path / "created/cameras.txt")
    (view_path / "created/points3D.txt").touch()
    colmap("feature_extractor", "--database_path", "database.db", "--image_path", "images",
           "--SiftExtraction.max_image_size", 4032,
           "--SiftExtraction.max_num_features", max_num_features,
           "--SiftExtraction.estimate_affine_shape", 1,
           "--SiftExtraction.domain_size_pooling", 1)
    colmap("exhaustive_matcher", "--database_path", "database.db",
           "--SiftMatching.guided_matching", 1, "--SiftMatching.max_num_matches", 32768)

    with closing(sqlite3.connect(view_path / "database.db")) as database:
        db_images = database.execute("SELECT image_id, name FROM images ORDER BY image_id").fetchall()
    with (view_path / "created/images.txt").open("w") as target:
        for image_id, name in db_images:
            target.write(f"{image_id} " + " ".join(images[name]) + "\n\n")

    colmap("point_triangulator", "--database_path", "database.db", "--image_path", "images",
           "--input_path", "created", "--output_path", "triangulated",
           "--Mapper.ba_local_max_num_iterations", 40, "--Mapper.ba_local_max_refinements", 3,
           "--Mapper.ba_global_max_num_iterations", 100)
    colmap("model_converter", "--input_path", "triangulated",
           "--output_path", "triangulated", "--output_type", "TXT")
    colmap("image_undistorter", "--image_path", "images", "--input_path", "triangulated",
           "--output_path", "dense")
    colmap("patch_match_stereo", "--workspace_path", "dense")
    colmap("stereo_fusion", "--workspace_path", "dense", "--output_path", "dense/fused.ply")


def main(dataset_path, scenes, n_views, max_num_features):
    parser = ArgumentParser(description="Generate sparse-view dense point clouds with COLMAP")
    parser.add_argument("--dataset_path", default=dataset_path, help="Dataset root directory")
    parser.add_argument("--n_views", type=int, default=n_views, help="Number of training views")
    parser.add_argument("--scenes", nargs="+", default=scenes, help="Scene directory names")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing <n>_views directories")
    args = parser.parse_args()
    if args.n_views < 1:
        parser.error("--n_views must be positive")
    base_path = Path(args.dataset_path).expanduser().resolve()
    for scene in args.scenes:
        pipeline(scene, base_path, args.n_views, max_num_features, overwrite=args.overwrite)
