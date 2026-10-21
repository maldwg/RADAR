"""Verify that documentation models describe production classes and real SQL keys."""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('radar_design_models', ROOT/'scripts/generate_design_models.py')
models = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(models)


class DesignModelTests(unittest.TestCase):
    """Keep the committed documentation models aligned with implementation."""
    def test_class_views_cover_the_complete_inventory(self):
        """Include every production class and disambiguate repeated class names."""
        inventory = models.classes(ROOT)
        viewed = {key for _,_,rows in models.model_views(inventory) for row in rows for key in row}
        self.assertEqual(viewed,set(inventory))
        self.assertEqual(len([key for key in inventory if key.endswith('.Generator')]),4)
        self.assertGreater(len(inventory),50)

    def test_protocol_realizations_and_inheritance_are_distinct(self):
        """Check the depicted protocol operations and explicit Python inheritance."""
        inventory = models.classes(ROOT)
        relations,_ = models.relationships(ROOT,inventory)
        for source,target,kind,_ in relations:
            with self.subTest(source=source,target=target):
                self.assertIn(source,inventory)
                self.assertIn(target,inventory)
                if kind == 'realization':
                    self.assertEqual(inventory[target]['kind'],'protocol')
                    self.assertNotIn(inventory[target]['name'],inventory[source]['bases'])
                    self.assertTrue(set(inventory[target]['methods']) <= set(inventory[source]['methods']))
                if kind == 'inheritance':
                    self.assertIn(inventory[target]['name'],inventory[source]['bases'])

    def test_new_classes_and_import_aliases_produce_relationships_without_registration(self):
        """Discover new classes and update held references and bases after source edits."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root/'src/radar'
            package.mkdir(parents=True)
            (package/'model_types.py').write_text('class Entity: pass\nclass Base: pass\nclass OtherBase: pass\ndef build() -> Entity: return Entity()\n')
            source = package/'custom.py'
            text = ('from .model_types import Entity as Item, Base, OtherBase, build\n'
                    'class NewBranch(Base):\n'
                    '    def __init__(self, item: Item):\n'
                    '        self.item = item\n'
                    '        self.extra = build()\n'
                    '    def result(self) -> Item: return Item()\n')
            source.write_text(text)
            inventory = models.classes(root)
            relations,evidence = models.relationships(root,inventory)
            source_key = 'radar.custom.NewBranch'
            target_key = 'radar.model_types.Entity'
            self.assertIn((source_key,target_key,'association','extra, item'),relations)
            self.assertIn((source_key,'radar.model_types.Base','inheritance','extends'),relations)
            self.assertIn(source_key,{key for _,_,rows in models.model_views(inventory) for row in rows for key in row})
            self.assertTrue(evidence[f'{source_key}|{target_key}|association']['item'][0].startswith('src/radar/custom.py:'))
            source.write_text(text.replace('NewBranch(Base)','NewBranch(OtherBase)').replace('self.item = item','self.item = None'))
            changed,_ = models.relationships(root,models.classes(root))
            self.assertIn((source_key,target_key,'association','extra'),changed)
            self.assertIn((source_key,'radar.model_types.OtherBase','inheritance','extends'),changed)
            self.assertNotIn((source_key,'radar.model_types.Base','inheritance','extends'),changed)

    def test_protocol_candidates_require_attributes_and_compatible_call_shapes(self):
        """Avoid drawing protocol arrows from a method-name coincidence alone."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root/'src/radar'
            package.mkdir(parents=True)
            (package/'contract.py').write_text(
                'from typing import Protocol\n'
                'class Contract(Protocol):\n'
                '    descriptor: str\n'
                '    def run(self, item: str, *, mode: str = "x") -> str: ...\n'
                'class Good:\n'
                '    descriptor = "ok"\n'
                '    def run(self, item: str, *, mode: str = "x") -> str: return item\n'
                'class WrongSignature:\n'
                '    descriptor = "ok"\n'
                '    def run(self, item: str, extra: str) -> str: return item\n'
                'class MissingAttribute:\n'
                '    def run(self, item: str, *, mode: str = "x") -> str: return item\n')
            relations,_ = models.relationships(root,models.classes(root))
            candidates = {(source,target) for source,target,kind,_ in relations if kind=='realization'}
            self.assertEqual(candidates,{('radar.contract.Good','radar.contract.Contract')})

    def test_actual_foreign_keys_and_optional_reference_matches(self):
        """Preserve real key constraints, including unmatched evaluation entries."""
        reporting = models.schema(ROOT,'reporting')
        cache = models.schema(ROOT,'cache')
        self.assertEqual(len(reporting),15)
        self.assertEqual(len(cache),3)
        self.assertEqual(sum(len(t['foreign_keys']) for t in reporting.values()),18)
        self.assertFalse(any(t['foreign_keys'] for t in cache.values()))
        fks = {fk['from']:fk for fk in reporting['reference_matches']['foreign_keys']}
        self.assertTrue(fks['entity_id']['nullable'])
        self.assertTrue(fks['decision_id']['nullable'])
        self.assertFalse(fks['evaluation_id']['nullable'])
        self.assertNotIn('reference_key',fks)  # Application join, not a SQL foreign key.
        self.assertFalse(reporting['schema_meta']['foreign_keys'])
        self.assertFalse(reporting['figure_manifests']['foreign_keys'])
        self.assertEqual([(c['name'],c['pk']) for c in cache['decisions']['columns'] if c['pk']],
                         [('namespace',1),('record_key',2),('profile_hash',3),('evidence_hash',4)])
        mermaid = models.er_mermaid(reporting)
        self.assertIn('entities |o..o{ reference_matches',mermaid)
        self.assertIn('evaluations ||--o{ reference_matches',mermaid)
        self.assertIn('TEXT entity_id FK "nullable"',mermaid)

    def test_check_detects_stale_svg_without_overwriting_it(self):
        """Make the CI drift check fail when a committed figure has changed."""
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            models.generate(ROOT,output,pdf=False)
            svg = output/'uml-all-classes.svg'
            svg.write_text('stale diagram')
            result = subprocess.run([sys.executable,str(ROOT/'scripts/generate_design_models.py'),
                                     '--check','--output',str(output)],capture_output=True,text=True)
            self.assertEqual(result.returncode,1)
            self.assertIn('uml-all-classes.svg',result.stderr)
            self.assertEqual(svg.read_text(),'stale diagram')

    def test_generation_is_deterministic_and_never_opens_run_databases(self):
        """Rebuild checked-in models using disposable in-memory schemas only."""
        connect = sqlite3.connect
        opened = []
        def memory_only(path,*args,**kwargs):
            """Reject any attempt to access a real SQLite database during generation."""
            opened.append(path)
            self.assertEqual(path,':memory:')
            return connect(path,*args,**kwargs)
        with tempfile.TemporaryDirectory() as tmp,patch.object(models.sqlite3,'connect',side_effect=memory_only):
            output = Path(tmp)
            result = models.generate(ROOT,output,pdf=False)
            self.assertEqual(opened,[':memory:',':memory:'])
            installed = ROOT/'docs/_static/models'
            for generated in output.iterdir():
                with self.subTest(file=generated.name):
                    self.assertEqual(generated.read_bytes(),(installed/generated.name).read_bytes())
                    if generated.suffix == '.svg':
                        svg = ET.parse(generated).getroot()
                        self.assertEqual(svg.tag,'{http://www.w3.org/2000/svg}svg')
                        self.assertIn('viewBox',svg.attrib)
                        _,_,width,height = map(float,svg.attrib['viewBox'].split())
                        for element in svg.findall('.//{http://www.w3.org/2000/svg}path'):
                            data = element.get('d','')
                            if ' V ' not in data or ' Q ' in data: continue
                            for kind,a,b in re.findall(r'([MHV])\s+(-?\d+(?:\.\d+)?)(?:\s+(-?\d+(?:\.\d+)?))?',data):
                                if kind in {'M','H'}: self.assertTrue(0 <= float(a) <= width)
                                if kind == 'M': self.assertTrue(0 <= float(b) <= height)
                                if kind == 'V': self.assertTrue(0 <= float(a) <= height)
            saved = json.loads((output/'model_inventory.json').read_text())
            self.assertEqual(set(saved['classes']),set(result['classes']))
            full_uml = (output/'uml-all-classes.puml').read_text()
            for key in result['classes']:
                self.assertIn(f'"{key}"',full_uml)
            self.assertIn('builtins.ValueError',full_uml)
            self.assertIn('logging.Formatter',full_uml)


if __name__ == '__main__':
    unittest.main()
