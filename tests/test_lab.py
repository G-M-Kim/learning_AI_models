"""Meaningful invariants: leakage, gradients, state reset, all supported combinations."""
import unittest
import torch
from ailab.data import VOCAB, datasets, recall
from ailab.models import MODELS, Spike, build
from ailab.run import evaluate
from torch.utils.data import DataLoader, TensorDataset


class LabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_every_model_backward(self):
        for task, names in MODELS.items():
            x, y = datasets(task, length=8, quick=True)[0]
            for name in names:
                with self.subTest(task=task, model=name):
                    model = build(task, name, length=8, vocab=len(VOCAB))
                    logits = model(x[:4])
                    self.assertEqual(logits.shape[:-1], y[:4].shape)
                    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y[:4].reshape(-1))
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss))
                    grads = [p.grad for p in model.parameters() if p.grad is not None]
                    self.assertTrue(grads)
                    self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def test_causal_lm_cannot_read_future(self):
        for name in MODELS['language']:
            model = build('language', name, 12, len(VOCAB)).eval()
            x = torch.randint(0, len(VOCAB), (2,12))
            z = x.clone()
            z[:,6:] = (z[:,6:] + 1) % len(VOCAB)
            with torch.no_grad():
                a, b = model(x), model(z)
            torch.testing.assert_close(a[:,:6], b[:,:6])

    def test_snn_state_reset(self):
        for task, name in [('digits','snn'), ('digits','scnn'), ('recall','srnn')]:
            model = build(task, name, 8).eval()
            x = datasets(task, 8, True)[2][0][:3]
            with torch.no_grad():
                a, b = model(x), model(x)
            torch.testing.assert_close(a, b)

    def test_surrogate_is_trainable(self):
        x = torch.tensor([-.2, .2], requires_grad=True)
        Spike.apply(x).sum().backward()
        self.assertTrue((x.grad > 0).all())

    def test_splits_and_recall_semantics(self):
        splits = datasets('recall', 12)
        self.assertFalse(torch.equal(splits[1][0], splits[2][0]))
        x, y = recall(32,12,1)
        self.assertTrue(torch.equal(x[:,0].argmax(-1), y))
        self.assertTrue((x[:,-1].argmax(-1) == 8).all())
        for task in MODELS:
            a, b = datasets(task, 8, True), datasets(task, 8, True)
            for (x,y), (u,v) in zip(a,b):
                torch.testing.assert_close(x,u)
                torch.testing.assert_close(y,v)

    def test_metric_weights_partial_batch(self):
        class Fixed(torch.nn.Module):
            def forward(self, x):
                return x
        x = torch.tensor([[4.,0.],[0.,4.],[0.,4.]])
        y = torch.tensor([0,1,0])
        loss, acc = evaluate(Fixed(), DataLoader(TensorDataset(x,y), batch_size=2), torch.device('cpu'))
        self.assertAlmostEqual(acc, 2/3)
        self.assertAlmostEqual(loss, torch.nn.functional.cross_entropy(x,y).item(), places=6)


if __name__ == '__main__':
    unittest.main()
