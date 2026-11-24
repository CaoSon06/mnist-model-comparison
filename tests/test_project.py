"""Offline behavioral tests; no MNIST download or previously trained weights."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

from mnist_project.data import normalize_images, split_data, load_data
from mnist_project.models import build_model, load_model, save_model, DEEP_MODELS
from mnist_project.experiments import run_experiment, evaluate_saved, predict_images, raw_predictions
from mnist_project.explain import integrated_gradients, saliency, gradcam, attention_rollout, explain_run, deletion_test


def fixture():
    rng = np.random.default_rng(7)
    x = rng.integers(0, 256, (100, 28, 28), dtype=np.uint8)
    y = np.tile(np.arange(10), 10)
    tx = rng.integers(0, 256, (20, 28, 28), dtype=np.uint8)
    ty = np.tile(np.arange(10), 2)
    return x, y, tx, ty


class DataTests(unittest.TestCase):
    def test_normalization_and_bad_inputs(self):
        x = np.zeros((1, 28, 28), dtype=np.uint8)
        x[0, 0, :2] = [255, 128]
        actual = normalize_images(x)
        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(actual.shape, (1, 28, 28, 1))
        self.assertAlmostEqual(float(actual[0, 0, 0, 0]), 1.)
        self.assertAlmostEqual(float(actual[0, 0, 1, 0]), 128/255)
        np.testing.assert_array_equal(normalize_images(actual), actual)
        for bad in [np.zeros((1, 20, 20)), np.full((1, 28, 28), np.nan), np.full((1, 28, 28), -1.)]:
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                normalize_images(bad)

    def test_disjoint_reproducible_split_preserves_official_test(self):
        x, y, tx, ty = fixture()
        data = split_data(x, y, tx, ty, validation_size=.2, seed=42)
        self.assertEqual((len(data.y_train), len(data.y_val), len(data.y_test)), (80, 20, 20))
        self.assertFalse(set(data.train_indices) & set(data.val_indices))
        self.assertEqual(set(data.train_indices) | set(data.val_indices), set(range(100)))
        np.testing.assert_array_equal(data.x_test, normalize_images(tx))
        np.testing.assert_array_equal(data.y_test, ty)
        again = split_data(x, y, tx, ty, validation_size=.2, seed=42)
        np.testing.assert_array_equal(data.train_indices, again.train_indices)

    def test_npz_and_stratified_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'data.npz'
            x, y, tx, ty = fixture()
            np.savez(path, x_train=x, y_train=y, x_test=tx, y_test=ty)
            data = load_data(path, validation_size=.2, train_limit=40, test_limit=10)
            self.assertEqual(len(data.y_train), 40)
            self.assertEqual(set(data.y_train), set(range(10)))
            self.assertEqual(set(data.y_test), set(range(10)))
            with self.assertRaises(ValueError):
                load_data(path, train_limit=2)

    def test_subsets_retain_rare_classes_and_allow_near_full_size(self):
        from mnist_project.data import _limit
        labels=np.repeat(np.arange(10),[100,2,2,2,2,2,2,2,2,2])
        selected=_limit(np.arange(len(labels)),labels,10,42)
        self.assertEqual(set(labels[selected]),set(range(10)))
        near_full=_limit(np.arange(100),np.tile(np.arange(10),10),99,42)
        self.assertEqual(len(near_full),99)
        self.assertEqual(len(np.unique(near_full)),99)


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = split_data(*fixture(), validation_size=.2)

    def test_all_model_families_train_save_reload_and_evaluate(self):
        for name in ['knn', 'svm', 'random_forest', 'mlp', 'cnn', 'resnet9', 'tinyvit']:
            with self.subTest(model=name), tempfile.TemporaryDirectory() as directory:
                dest = Path(directory)/name
                run_experiment(name, self.data, dest, epochs=1, batch_size=20, verbose=0)
                metrics = json.loads((dest/'metrics.json').read_text())
                self.assertEqual(metrics['n_test'], 20)
                self.assertTrue(0 <= metrics['accuracy'] <= 1)
                self.assertEqual(np.load(dest/'predictions.npy').shape, (20,))
                report = evaluate_saved(dest, self.data.x_test, self.data.y_test)
                self.assertAlmostEqual(report['accuracy'], metrics['accuracy'])
                pred = predict_images(dest, self.data.x_test[:2])
                self.assertEqual(len(pred['predictions']), 2)
                if name=='svm': self.assertNotIn('probabilities', pred)
                self.assertTrue((dest/'confusion_matrix.png').is_file())
                explanation=explain_run(dest,self.data.x_test[:1],Path(directory)/'explanation',steps=4)
                self.assertEqual(explanation['model'],name)
                self.assertTrue((Path(directory)/'explanation'/'explanation.png').is_file())
                with self.assertRaises(FileExistsError):
                    run_experiment(name, self.data, dest, epochs=1)

    def test_vit_embeddings_are_tracked_and_roundtrip(self):
        import tensorflow as tf
        model = build_model('tinyvit', seed=42)
        embed = model.get_layer('position_embedding')
        self.assertEqual(len(embed.trainable_weights), 1)
        before = embed.get_weights()[0].copy()
        model.train_on_batch(self.data.x_train[:10], self.data.y_train[:10])
        self.assertGreater(float(np.max(np.abs(before-embed.get_weights()[0]))), 0)
        with tempfile.TemporaryDirectory() as directory:
            save_model(model, 'tinyvit', Path(directory))
            loaded = load_model('tinyvit', Path(directory))
            np.testing.assert_allclose(model(self.data.x_test[:2]), loaded(self.data.x_test[:2]), atol=1e-6)
        tf.keras.backend.clear_session()

    def test_benchmark_prediction_does_not_compute_optional_scores(self):
        model=build_model('svm')
        model.fit(self.data.x_train.reshape(80,784),self.data.y_train)
        result=raw_predictions(model,'svm',self.data.x_test,include_scores=False)
        self.assertEqual(set(result),{'predictions'})
        self.assertEqual(result['predictions'].shape,(20,))


class CliTests(unittest.TestCase):
    def test_offline_train_evaluate_predict_eda_and_tune(self):
        from mnist_project.cli import main
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root=Path(directory)
            x,y,tx,ty=fixture()
            path=root/'data.npz'
            np.savez(path,x_train=x,y_train=y,x_test=tx,y_test=ty)
            common=['--data',str(path),'--validation-size','.2']
            run=root/'knn'
            main(['train','--model','knn','--output',str(run),*common])
            main(['evaluate','--run',str(run),'--output',str(root/'evaluation.json')])
            images=root/'images.npy'; np.save(images,tx[:2])
            main(['predict','--run',str(run),'--images-npy',str(images),'--output',str(root/'prediction.json')])
            self.assertEqual(len(json.loads((root/'prediction.json').read_text())['predictions']),2)
            main(['eda','--output',str(root/'eda'),*common])
            self.assertTrue((root/'eda'/'summary.json').is_file())
            main(['tune','--model','mlp','--trials','1','--epochs','1','--output',str(root/'tuning'),*common])
            tuned=json.loads((root/'tuning'/'best_params.json').read_text())
            self.assertEqual(tuned['n_train'],80)
            main(['train','--model','mlp','--params',str(root/'tuning'/'best_params.json'),
                  '--batch-size','20','--epochs','1','--output',str(root/'mlp'),*common])
            config=json.loads((root/'mlp'/'config.json').read_text())
            self.assertEqual(config['batch_size'],20)


class ExplainTests(unittest.TestCase):
    def test_deletion_keeps_explicit_nonpredicted_target(self):
        import tensorflow as tf
        inputs=tf.keras.Input((28,28,1))
        output=tf.keras.layers.Dense(10,activation='softmax',use_bias=False)(tf.keras.layers.Flatten()(inputs))
        model=tf.keras.Model(inputs,output)
        weights=np.zeros((784,10),np.float32)
        weights[:,0]=.001; weights[:,1]=-.001
        model.layers[-1].set_weights([weights])
        image=np.full((28,28,1),.5,np.float32)
        rows=deletion_test(model,image,np.ones((28,28)),fractions=(.5,),random_runs=1,target=1)
        self.assertLess(rows[0]['guided_drop'],0)
        self.assertEqual(rows[0]['target'],1)

    def test_integrated_gradients_matches_linear_score_difference(self):
        import tensorflow as tf
        inputs = tf.keras.Input((28,28,1))
        values = tf.keras.layers.Flatten()(inputs)
        output = tf.keras.layers.Dense(10, activation='softmax', use_bias=False)(values)
        model = tf.keras.Model(inputs, output)
        weights = np.zeros((784,10), np.float32)
        weights[:,3] = 2
        model.layers[-1].set_weights([weights])
        image = np.full((28,28,1), .5, np.float32)
        attr = integrated_gradients(model, image, target=3, steps=8)
        np.testing.assert_allclose(attr, np.ones((28,28)), atol=1e-6)
        self.assertAlmostEqual(float(attr.sum()), 784.)

    def test_maps_finite_for_conv_and_transformer(self):
        image = normalize_images(fixture()[0][:1])[0]
        for name in ['cnn','resnet9','tinyvit']:
            with self.subTest(model=name):
                model = build_model(name)
                maps = [saliency(model,image), integrated_gradients(model,image,steps=4)]
                maps += [attention_rollout(model,image)] if name=='tinyvit' else [gradcam(model,image)]
                for result in maps:
                    self.assertEqual(result.shape,(28,28))
                    self.assertTrue(np.isfinite(result).all())


if __name__ == '__main__':
    unittest.main()
