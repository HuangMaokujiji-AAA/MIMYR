"""Checks for the notebook's exact paper split, ID alignment, and result preservation."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / 'notebooks/merfish_pretrained_3fold.ipynb'


def notebook_functions():
    notebook = nbformat.read(BOOK, as_version=4)
    nbformat.validate(notebook)
    namespace = {'Path': Path, 'ast': ast, 'hashlib': hashlib, 'json': json,
                 'np': np, 'pd': pd}
    wanted = {'sha256', 'write_json_new', 'write_csv_new', 'paper_split_from_source',
              'build_protocols', 'ccf_match', 'initialize_run', 'new_attempt', 'verify_completed_case'}
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        compile(cell.source, cell.id, 'exec')
        tree = ast.parse(cell.source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {'train', 'backward', 'step'}:
                    raise AssertionError(f'Training call found in {cell.id}')
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), cell.id, 'exec'), namespace)
    return namespace


class ReconstructionNotebookTests(unittest.TestCase):
    def test_original_split_and_custom_threefold(self):
        ns = notebook_functions()
        split = ns['paper_split_from_source'](ROOT)
        with tempfile.TemporaryDirectory(prefix='reconstruction-split-', dir=ROOT / 'tmp') as tmp:
            folder = Path(tmp)
            # Actual author archive filenames; this fixture only exercises filename/index selection.
            for i in range(1, 74):
                if i not in {24, 29, 38, 43, 64, 68, 69}:
                    (folder / f'sec_{i:03}.h5ad').touch()
            protocols = ns['build_protocols'](folder, split, [31, 32, 33, 34, 35])
            paper = protocols['paper_rq3']
            custom = protocols['custom_3fold']
            self.assertEqual([c['target'] for c in paper['cases']],
                             ['sec_021.h5ad', 'sec_028.h5ad', 'sec_037.h5ad', 'sec_047.h5ad', 'sec_051.h5ad'])
            self.assertEqual(paper['original_split']['train_indices'],
                             ['sec_018.h5ad', 'sec_025.h5ad', 'sec_044.h5ad', 'sec_049.h5ad'])
            self.assertEqual(paper['original_split']['val_indices'], ['sec_034.h5ad'])
            self.assertEqual(paper['cases'][-1]['reference'], ['sec_049.h5ad', 'sec_049.h5ad'])
            self.assertEqual([c['target'] for c in custom['cases']],
                             ['sec_032.h5ad', 'sec_033.h5ad', 'sec_034.h5ad'])
            self.assertEqual(custom['cases'][1]['reference'], ['sec_032.h5ad', 'sec_034.h5ad'])
            for protocol in protocols.values():
                for case in protocol['cases']:
                    self.assertNotIn(case['target'], case['reference'])

    def test_ccf_alignment_and_outputs_are_not_overwritten(self):
        ns = notebook_functions()
        ids = ['212059428959805613780123472670405935608', '101833224400913170993812156276156584394']
        ccf = pd.DataFrame({'x': [2., 1.], 'y': [4., 3.], 'z': [6., 5.]}, index=ids[::-1])
        xyz, valid = ns['ccf_match']([*ids, 'unregistered'], ccf)
        np.testing.assert_array_equal(valid, [True, True, False])
        np.testing.assert_array_equal(xyz[:2, 0], [1., 2.])
        with tempfile.TemporaryDirectory(prefix='reconstruction-preserve-', dir=ROOT / 'tmp') as tmp:
            paper = Path(tmp) / 'paper_rq3' / 'run_1'
            custom = Path(tmp) / 'custom_3fold' / 'run_1'
            identity = {'seed': 42, 'input': 'unchanged'}
            ns['initialize_run'](paper, identity)
            ns['initialize_run'](custom, identity)
            self.assertNotEqual(paper, custom)
            original = (paper / 'experiment.json').read_bytes()
            with self.assertRaises(FileExistsError):
                ns['initialize_run'](paper, identity)
            ns['initialize_run'](paper, identity, resume=True)
            with self.assertRaises(FileExistsError):
                ns['initialize_run'](paper, {'seed': 1}, resume=True)
            self.assertEqual((paper / 'experiment.json').read_bytes(), original)
            first = ns['new_attempt'](paper / 'case_1')
            (first / 'prediction.h5ad').write_bytes(b'existing result')
            second = ns['new_attempt'](paper / 'case_1')
            self.assertNotEqual(first, second)
            self.assertEqual((first / 'prediction.h5ad').read_bytes(), b'existing result')
            ns['write_json_new'](second / 'metrics.json', {'score': 1.0})
            with self.assertRaises(FileExistsError):
                ns['write_json_new'](second / 'metrics.json', {'score': 0.0})
            ns['write_json_new'](paper / 'case_1/completed.json', {
                'attempt': second.name,
                'files': {'metrics.json': ns['sha256'](second / 'metrics.json')}})
            self.assertEqual(ns['verify_completed_case'](paper / 'case_1'), second)
            (second / 'metrics.json').write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError):
                ns['verify_completed_case'](paper / 'case_1')


if __name__ == '__main__':
    unittest.main()
