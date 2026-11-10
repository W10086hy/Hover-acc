import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import project_files


class ArtifactPathTests(unittest.TestCase):
    def test_directory_relocation_preserves_nested_original_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'cleanup').mkdir();(root/'new/domain').mkdir(parents=True)
            (root/'new/domain/value.json').write_text('{}',encoding='utf-8')
            (root/'cleanup/path_map.json').write_text(json.dumps({'old':'new'}),encoding='utf-8')
            with patch.object(project_files,'ROOT',root):
                self.assertEqual(project_files.resolve_artifact('old/domain',directory=True),root/'new/domain')
                self.assertEqual(project_files.resolve_artifact('old/domain/value.json'),root/'new/domain/value.json')

    def test_relative_parent_escape_cannot_resolve_a_file_outside_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            outer=Path(tmp);root=outer/'project';root.mkdir();(outer/'outside.txt').write_text('fixture')
            with patch.object(project_files,'ROOT',root):
                with self.assertRaises(ValueError):project_files.resolve_artifact('../outside.txt')

    def test_existing_file_still_wins_over_historical_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'value.json').write_text('{}',encoding='utf-8')
            with patch.object(project_files,'ROOT',root):
                self.assertEqual(project_files.resolve_artifact('value.json'),root/'value.json')
