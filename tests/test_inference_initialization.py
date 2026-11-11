"""Check actual model definitions without importing optional data/transformer integrations."""
import ast
from copy import deepcopy
from dataclasses import dataclass
import itertools
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import anndata as ad
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]


def load_model_definitions():
    """Execute the unchanged class bodies, including their real network construction."""
    namespace = dict(torch=torch, nn=nn, optim=optim, np=np, math=math, itertools=itertools,
                     dataclass=dataclass, deepcopy=deepcopy, DataLoader=DataLoader,
                     TensorDataset=TensorDataset, ad=ad)
    for filename in ['models/diffusion_model.py', 'models/celltype_model.py', 'models/combined_model.py']:
        tree = ast.parse((ROOT / filename).read_text(encoding='utf-8-sig'))
        definitions = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), filename, 'exec'), namespace)
    return namespace


class InferenceInitializationTests(unittest.TestCase):
    def test_optimizer_integrations_are_skipped_until_training(self):
        namespace = load_model_definitions()
        # Simulate an optimizer constructor hitting the server's protobuf failure.
        # Initialization must still construct both original-sized models successfully.
        with patch.object(torch.cuda, 'is_available', return_value=False), \
             patch.object(optim, 'AdamW', side_effect=TypeError('legacy protobuf integration')) as adamw, \
             patch.object(optim, 'Adam', side_effect=TypeError('legacy protobuf integration')) as adam:
            model = namespace['CombinedModel']()
            self.assertIsNone(model.trainer.opt)
            self.assertIsNone(model.celltype_model.optimizer)
            adamw.assert_not_called()
            adam.assert_not_called()

        # Zero-epoch engineering checks execute training initialization without updates.
        # Real original optimizers must retain their original learning rates and decay.
        model.trainer.cfg.epochs = 0
        model.trainer.train()
        self.assertIsInstance(model.trainer.opt, optim.AdamW)
        self.assertEqual(model.trainer.opt.param_groups[0]['lr'], 2e-4)
        self.assertEqual(model.trainer.opt.param_groups[0]['weight_decay'], 0)

        tiny = ad.AnnData(np.zeros((2, 1), dtype=np.float32))
        tiny.obsm['aligned_spatial'] = np.zeros((2, 3), dtype=np.float32)
        tiny.obs['token'] = [0, 1]
        model.celltype_model.fit(tiny, epochs=0)
        self.assertIsInstance(model.celltype_model.optimizer, optim.Adam)
        self.assertEqual(model.celltype_model.optimizer.param_groups[0]['lr'], 0.001)


if __name__ == '__main__':
    unittest.main()
