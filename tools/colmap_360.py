"""Prepare sparse-view dense point clouds for Mip-NeRF 360."""

if __package__:
    from .colmap_common import main
else:
    from colmap_common import main


if __name__ == "__main__":
    main(
        dataset_path="dataset/mipnerf360",
        scenes=['bicycle', 'bonsai', 'counter', 'garden', 'kitchen', 'room', 'stump'],
        n_views=12,
        max_num_features=16384,
    )
