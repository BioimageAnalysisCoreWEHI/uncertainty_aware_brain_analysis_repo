# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 15:14:11 2026

@author: zaman.i
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)

from pathlib import Path
import argparse
import shutil


TIFF_EXTENSIONS = (".tif", ".tiff")


def find_file(folder, names=None, pattern=None):
    """Return the first matching TIFF file in a folder."""

    if names is not None:
        for name in names:
            path = folder / name
            if path.exists():
                return path

    if pattern is not None:
        matches = sorted(
            path
            for path in folder.glob(pattern)
            if path.is_file()
            and path.suffix.lower() in TIFF_EXTENSIONS
        )

        if matches:
            return matches[0]

    return None


def copy_file(source, destination, overwrite=False):
    if destination.exists() and not overwrite:
        print(f"[SKIP] Already exists: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)

    print(f"[COPIED] {source} -> {destination}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Collect downsampled images, registered atlases and "
            "downsampled standard images from registration folders."
        )
    )

    parser.add_argument(
        "--source-root",
        default=(
            P(r"<SCRATCH>"
            r"\jom_review_sweep")
        ),
    )

    parser.add_argument(
        "--output-root",
        default=P(r"<IMAGING>\Ishrat\JoM\post_review_analysis\results\registration_images_collected"),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)

    if not source_root.exists():
        raise FileNotFoundError(
            f"Source folder does not exist: {source_root}"
        )

    # Expected:
    # source_root / sample_id / input_type / registration_folder
    for sample_dir in sorted(source_root.iterdir()):
        if not sample_dir.is_dir():
            continue

        sample_id = sample_dir.name

        for input_dir in sorted(sample_dir.iterdir()):
            if not input_dir.is_dir():
                continue

            input_type = input_dir.name

            registration_dirs = sorted(
                path
                for path in input_dir.iterdir()
                if path.is_dir()
            )

            if not registration_dirs:
                continue

            print("\n" + "=" * 80)
            print(f"{sample_id} | {input_type}")
            print("=" * 80)

            destination_dir = (
                output_root
                / sample_id
                / input_type
            )

            # ------------------------------------------------------
            # Copy one downsampled and one TH image
            # ------------------------------------------------------

            downsampled_copied = False
            th_copied = False

            for registration_dir in registration_dirs:
                if not downsampled_copied:
                    downsampled_path = find_file(
                        registration_dir,
                        names=[
                            "downsampled.tiff",
                            "downsampled.tif",
                        ],
                    )

                    if downsampled_path is not None:
                        copy_file(
                            downsampled_path,
                            destination_dir / downsampled_path.name,
                            overwrite=args.overwrite,
                        )
                        downsampled_copied = True

                if not th_copied:
                    th_path = find_file(
                        registration_dir,
                        pattern="downsampled_*_TH_level1.tif*",
                    )

                    if th_path is not None:
                        copy_file(
                            th_path,
                            destination_dir / th_path.name,
                            overwrite=args.overwrite,
                        )
                        th_copied = True

                if downsampled_copied and th_copied:
                    break

            if not downsampled_copied:
                print(
                    f"[WARNING] No downsampled.tif/.tiff found for "
                    f"{sample_id} | {input_type}"
                )

            if not th_copied:
                print(
                    f"[WARNING] No downsampled TH image found for "
                    f"{sample_id} | {input_type}"
                )

           # ------------------------------------------------------
        # Copy outputs from every registration
        # ------------------------------------------------------
        
        for registration_dir in registration_dirs:
            registration_name = registration_dir.name
        
            # Define the outputs we want to collect
            registration_outputs = {
                "registered_atlas": [
                    "registered_atlas.tiff",
                    "registered_atlas.tif",
                ],
                "registered_hemispheres": [
                    "registered_hemispheres.tiff",
                    "registered_hemispheres.tif",
                ],
                "downsampled_standard": [
                    "downsampled_standard.tiff",
                    "downsampled_standard.tif",
                ],
            }
        
            for output_name, possible_names in registration_outputs.items():
        
                # Check whether either .tif or .tiff version is already collected
                existing_destinations = [
                    destination_dir
                    / f"{registration_name}__{output_name}{ext}"
                    for ext in TIFF_EXTENSIONS
                ]
        
                if not args.overwrite and any(
                    path.exists() for path in existing_destinations
                ):
                    print(
                        f"[SKIP] Already collected: "
                        f"{registration_name}__{output_name}"
                    )
                    continue
        
                # Only search the registration folder if we actually need the file
                source_file = find_file(
                    registration_dir,
                    names=possible_names,
                )
        
                if source_file is None:
                    print(
                        f"[WARNING] Missing {output_name}: "
                        f"{registration_dir}"
                    )
                    continue
        
                destination = destination_dir / (
                    f"{registration_name}"
                    f"__{output_name}"
                    f"{source_file.suffix.lower()}"
                )
        
                copy_file(
                    source_file,
                    destination,
                    overwrite=args.overwrite,
                )

    print("\n[DONE] Finished collecting registration images.")


if __name__ == "__main__":
    main()