from __future__ import annotations

import email
import json
import os
from pathlib import Path
import tarfile
import tomllib
import unittest
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_NAMES = {
    "artifact.json",
    "candidate_proposal.json",
    "declarative_provider.json",
    "evaluation_policy.json",
    "frozen_workload.json",
    "knob.json",
    "runtime_identity.json",
}


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

    def test_source_distribution_inputs_are_declared(self) -> None:
        with (PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
            config = tomllib.load(handle)
        data_files = config["tool"]["setuptools"]["data-files"]
        self.assertEqual(data_files["schemas"], ["schemas/*.json"])
        self.assertEqual(data_files["examples"], ["examples/*.json", "examples/README.md"])
        self.assertEqual(data_files["docs"], ["docs/*.md"])
        self.assertEqual({path.name for path in (PROJECT_ROOT / "schemas").glob("*.json")}, SCHEMA_NAMES)
        for schema_path in (PROJECT_ROOT / "schemas").glob("*.json"):
            with schema_path.open(encoding="utf-8") as handle:
                self.assertIsInstance(json.load(handle), dict)

    def test_built_archives_contain_release_surface(self) -> None:
        raw_dist_dir = os.environ.get("AUTO_MLX_DIST_DIR")
        if raw_dist_dir is None:
            self.skipTest("set AUTO_MLX_DIST_DIR to inspect built artifacts")
        dist_dir = Path(raw_dist_dir)
        wheels = sorted(dist_dir.glob("auto_mlx-0.1.0-*.whl"))
        sdists = sorted(dist_dir.glob("auto_mlx-0.1.0.tar.gz"))
        self.assertEqual(len(wheels), 1)
        self.assertEqual(len(sdists), 1)

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
                Path(name).name for name in names if "/schemas/" in name and name.endswith(".json")
            }
            self.assertEqual(wheel_schema_names, SCHEMA_NAMES)
            self.assertTrue(any(name.endswith(".data/data/examples/workload.json") for name in names))
            self.assertTrue(any(name.endswith(".data/data/examples/provider.json") for name in names))
            self.assertTrue(any(name.endswith(".data/data/examples/README.md") for name in names))
            self.assertTrue(any(name.endswith(".data/data/docs/cli.md") for name in names))

        with tarfile.open(sdists[0], "r:gz") as archive:
            names = set(archive.getnames())
            prefix = "auto_mlx-0.1.0/"
            self.assertIn(prefix + "pyproject.toml", names)
            self.assertIn(prefix + "README.md", names)
            self.assertIn(prefix + "examples/workload.json", names)
            self.assertIn(prefix + "docs/cli.md", names)
            sdist_schema_names = {
                Path(name).name for name in names if name.startswith(prefix + "schemas/") and name.endswith(".json")
            }
            self.assertEqual(sdist_schema_names, SCHEMA_NAMES)


if __name__ == "__main__":
    unittest.main()
