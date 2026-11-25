"""Prepare sparse-view dense point clouds for LLFF."""

if __package__:
    from .colmap_common import main
else:
    from colmap_common import main


if __name__ == "__main__":
    main(
        dataset_path="dataset/nerf_llff_data",
        scenes=['fern', 'flower', 'fortress', 'horns', 'leaves', 'orchids', 'room', 'trex'],
        n_views=3,
        max_num_features=32768,
    )
