from __future__ import annotations

import email
import json
import os
from pathlib import Path
import sys
import tarfile
import tomllib
import unittest
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from auto_mlx.schemas import schema_names  # noqa: E402


SCHEMA_NAMES = set(schema_names())


class PackagingTests(unittest.TestCase):
    def test_release_metadata_is_explicit_and_dependency_free(self) -> None:
        with (PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
            config = tomllib.load(handle)
        project = config["project"]
        self.assertEqual(project["name"], "auto-mlx")
        self.assertEqual(project["version"], "0.1.0")
        self.assertEqual(project["readme"], "README.md")
        self.assertEqual(project["requires-python"], ">=3.11")
        self.assertEqual(project["dependencies"], [])
        self.assertEqual(project["license"], "MIT")
        self.assertEqual(project["license-files"], ["LICENSE"])
        self.assertEqual(project["scripts"]["auto-mlx"], "auto_mlx.cli:main")

    def test_schemas_are_packaged_resources_with_a_single_source_of_truth(self) -> None:
        with (PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
            config = tomllib.load(handle)
        setuptools_config = config["tool"]["setuptools"]
        self.assertNotIn(
            "data-files",
            setuptools_config,
            "schemas must ship as auto_mlx.schemas package data, not sys.prefix data-files",
        )
        self.assertEqual(setuptools_config["package-data"]["auto_mlx.schemas"], ["*.json"])

        # The relocated package directory is the only copy of the schemas;
        # the old repo-root schemas/ directory must be gone, not duplicated.
        self.assertFalse((PROJECT_ROOT / "schemas").exists())
        schemas_dir = PROJECT_ROOT / "src" / "auto_mlx" / "schemas"
        self.assertTrue((schemas_dir / "__init__.py").is_file())
        self.assertEqual({path.name for path in schemas_dir.glob("*.json")}, SCHEMA_NAMES)
        for schema_path in schemas_dir.glob("*.json"):
            with schema_path.open(encoding="utf-8") as handle:
                self.assertIsInstance(json.load(handle), dict)

    def test_built_wheel_ships_only_the_package_and_its_schemas(self) -> None:
        raw_dist_dir = os.environ.get("AUTO_MLX_DIST_DIR")
        if raw_dist_dir is None:
            self.skipTest("set AUTO_MLX_DIST_DIR to inspect built artifacts")
        dist_dir = Path(raw_dist_dir)
        wheels = sorted(dist_dir.glob("auto_mlx-0.1.0-*.whl"))
        self.assertEqual(len(wheels), 1)

        with zipfile.ZipFile(wheels[0]) as archive:
            names = set(archive.namelist())
            self.assertTrue(any(name.endswith(".dist-info/METADATA") for name in names))
            metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
            metadata = email.message_from_bytes(archive.read(metadata_name))
            self.assertEqual(metadata["Name"], "auto-mlx")
            self.assertEqual(metadata["Version"], "0.1.0")
            self.assertEqual(metadata["Requires-Python"], ">=3.11")
            self.assertIsNone(metadata.get("Requires-Dist"))
            self.assertIn("# Auto MLX", metadata.get_payload())
            entry_points_name = next(name for name in names if name.endswith(".dist-info/entry_points.txt"))
            self.assertIn("auto-mlx = auto_mlx.cli:main", archive.read(entry_points_name).decode("utf-8"))

            wheel_schema_names = {
                Path(name).name for name in names if name.startswith("auto_mlx/schemas/") and name.endswith(".json")
            }
            self.assertEqual(wheel_schema_names, SCHEMA_NAMES)

            # Nothing else leaks into the wheel: no legacy .data/data root
            # (the old data-files mechanism), and no repo-only material such
            # as examples/ or docs/ that a user never imports.
            self.assertFalse(any(".data/" in name for name in names))
            self.assertFalse(any(name.startswith("examples/") for name in names))
            self.assertFalse(any(name.startswith("docs/") for name in names))

    def test_built_sdist_retains_the_full_release_surface(self) -> None:
        raw_dist_dir = os.environ.get("AUTO_MLX_DIST_DIR")
        if raw_dist_dir is None:
            self.skipTest("set AUTO_MLX_DIST_DIR to inspect built artifacts")
        dist_dir = Path(raw_dist_dir)
        sdists = sorted(dist_dir.glob("auto_mlx-0.1.0.tar.gz"))
        self.assertEqual(len(sdists), 1)

        with tarfile.open(sdists[0], "r:gz") as archive:
            names = set(archive.getnames())
            prefix = "auto_mlx-0.1.0/"
            self.assertIn(prefix + "pyproject.toml", names)
            self.assertIn(prefix + "MANIFEST.in", names)
            self.assertIn(prefix + "README.md", names)
            self.assertIn(prefix + "LICENSE", names)
            self.assertIn(prefix + "CONTRIBUTING.md", names)
            self.assertIn(prefix + "examples/workload.json", names)
            self.assertIn(prefix + "examples/README.md", names)
            self.assertIn(prefix + "docs/cli.md", names)
            self.assertIn(prefix + "tests/test_packaging.py", names)
            sdist_schema_names = {
                Path(name).name
                for name in names
                if name.startswith(prefix + "src/auto_mlx/schemas/") and name.endswith(".json")
            }
            self.assertEqual(sdist_schema_names, SCHEMA_NAMES)


if __name__ == "__main__":
    unittest.main()
