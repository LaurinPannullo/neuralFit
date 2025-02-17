import unittest
import numpy as np
import tensorflow as tf
import keras
from neuralFit import KL_kernel_Momentum, KL_kernel_Position_Vacuum, KL_kernel_Position_FiniteT, KL_kernel_Omega, Di, SpectralNN, networkTrainer, LossCalculator, networkParameters, ParameterHandler, FitRunner, networkTrainer
import os
import json
class TestKernels(unittest.TestCase):

    def setUp(self):
        self.Momentum = np.array([1, 2, 3])
        self.Position = np.array([1, 2, 3])
        self.Omega = np.array([1, 2, 3])


    def test_KL_kernel_Momentum(self):
        result = KL_kernel_Momentum(self.Momentum, self.Omega)
        expected = np.array([[1/(2 * np.pi), 2/(5 * np.pi), 3/(10 * np.pi)], 
                     [1/(5 * np.pi), 1/(4 * np.pi), 3/(13 * np.pi)], 
                     [1/(10 * np.pi), 2/(13 * np.pi), 1/(6 * np.pi)]])
        np.testing.assert_almost_equal(result, expected, decimal=5)

    def test_KL_kernel_Momentum_edge_cases(self):
        Momentum = np.array([0, 0, 0])
        Omega = np.array([1, 2, 3])
        result = KL_kernel_Momentum(Momentum, Omega)
        expected = np.array([[1/np.pi, 1/(2 * np.pi), 1/(3 * np.pi)], 
                     [1/np.pi, 1/(2 * np.pi), 1/(3 * np.pi)], 
                     [1/np.pi, 1/(2 * np.pi), 1/(3 * np.pi)]])
        np.testing.assert_almost_equal(result, expected, decimal=5)

    def test_KL_kernel_Position_Vacuum(self):
        result = KL_kernel_Position_Vacuum(self.Position, self.Omega)
        expected = np.array([[0.36787944, 0.13533528, 0.04978707],
                             [0.13533528, 0.01831564, 0.00247875],
                             [0.04978707, 0.00247875, 0.00012341]])
        np.testing.assert_almost_equal(result, expected, decimal=5)

    def test_KL_kernel_Position_FiniteT(self):
        T = 1.0
        result = KL_kernel_Position_FiniteT(self.Position, self.Omega, T)
        expected = np.array([[np.cosh(1/2) / np.sinh(1/2), np.cosh(1) / np.sinh(1), np.cosh(3/2) / np.sinh(3/2)],
                     [np.cosh(3/2) / np.sinh(1/2), np.cosh(3) / np.sinh(1), np.cosh(9/2) / np.sinh(3/2)],
                     [np.cosh(5/2) / np.sinh(1/2), np.cosh(5) / np.sinh(1), np.cosh(15/2) / np.sinh(3/2)]])
        np.testing.assert_almost_equal(result, expected, decimal=5)

    def test_KL_kernel_Momentum_Omega(self):
        KL = KL_kernel_Momentum
        result = KL_kernel_Omega(KL, self.Position, self.Omega)
        expected = np.array([[0.31830989, 0.15915494, 0.1061033 ],
                             [0.15915494, 0.15915494, 0.1061033 ],
                             [0.1061033 , 0.1061033 , 0.1061033 ]])
        expected = np.array([[1/(2 * np.pi), 4/(5 * np.pi), 9/(10 * np.pi)], 
                     [1/(5 * np.pi), 1/(2 * np.pi), 9/(13 * np.pi)], 
                     [1/(10 * np.pi), 4/(13 * np.pi), 1/(2 * np.pi)]])
        np.testing.assert_almost_equal(result, expected, decimal=5)

    def test_KL_kernel_Position_FiniteT_Omega(self):
        KL = KL_kernel_Position_FiniteT
        T = 1.0
        result = KL_kernel_Omega(KL, self.Position, self.Omega, [T])
        expected = np.array([[np.cosh(1/2) / np.sinh(1/2), 2 * np.cosh(1) / np.sinh(1), 3 * np.cosh(3/2) / np.sinh(3/2)],
                     [np.cosh(3/2) / np.sinh(1/2), 2 * np.cosh(3) / np.sinh(1), 3 * np.cosh(9/2) / np.sinh(3/2)],
                     [np.cosh(5/2) / np.sinh(1/2), 2 * np.cosh(5) / np.sinh(1), 3 * np.cosh(15/2) / np.sinh(3/2)]])
        np.testing.assert_almost_equal(result, expected, decimal=5)


class TestDiFunction(unittest.TestCase):

    def test_Di(self):
        KL = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
        rhoi = tf.constant([0.5, 0.5], dtype=tf.float32)
        delomega = tf.constant(0.1, dtype=tf.float32)
        result = Di(KL, rhoi, delomega)
        expected = tf.constant([0.15, 0.35], dtype=tf.float32)
        tf.debugging.assert_near(result, expected)

    def test_Di_different_inputs(self):
        KL = tf.constant([[2.0, 4.0], [6.0, 8.0]], dtype=tf.float32)
        rhoi = tf.constant([0.5, 0.5], dtype=tf.float32)
        delomega = tf.constant(0.2, dtype=tf.float32)
        result = Di(KL, rhoi, delomega)
        expected = tf.constant([0.6, 1.4], dtype=tf.float32)
        tf.debugging.assert_near(result, expected)

class TestSpectralNN(unittest.TestCase):

    def test_SpectralNN(self):
        model = SpectralNN(num_output_nodes=3, width=[32, 32])
        inputs = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32)
        output = model(inputs)
        self.assertEqual(output.shape, (1, 3))
        self.assertTrue(tf.reduce_all(output >= 0))  # Check if all outputs are non-negative due to softplus activation

    def test_SpectralNN_initialization(self):
        model = SpectralNN(num_output_nodes=3, width=[64, 64])
        self.assertEqual(len(model.hidden_layers), 2)
        self.assertEqual(model.hidden_layers[0].units, 64)
        self.assertEqual(model.hidden_layers[1].units, 64)
class TestFitRunner(unittest.TestCase):

    def setUp(self):
        self.paramsDefaultDict = {
            #NetworkParams
            "lambda_s": [1e-5],
            "lambda_l2": [1e-8],
            "epochs": [100],
            "learning_rate": [1e-4],
            "width": [32,32,32],
            "errorWeighting": True,
            "networkStructure": "SpectralNN",
            #Correlator/Rho params
            "omega_min": 0,
            "omega_max": 10,
            "omega_points": 3,
            "Nt": 0,
            "extractedQuantity": "RhoOverOmega",
            "FiniteT_kernel": True,
            "multiFit": False,
            "correlatorFile": "test_correlator.txt",
            "xCol": 0,
            "meanCol": 1,
            "errorCol": 2,
            "correlatorCols": "3:",
            "errormethod": "jackknife",
            #General Params
            "saveParams": False,
            "saveLossHistory": False,
            "verbose": False,
            "outputFile": None,
            "outputDir": ''
        }
        self.parameterHandler = ParameterHandler(self.paramsDefaultDict)
        self.data=np.array([
                        [0.0, 1.0, 0.1, 0.5],
                        [0.1, 1.1, 0.1, 0.6],
                        [0.2, 1.2, 0.1, 0.7]
                    ])
        self.correlatorFilename = "test_correlator.txt"
        self.create_mock_correlator(filename=self.correlatorFilename,data=self.data)

        self.fitRunner = FitRunner(self.parameterHandler)

    def create_mock_correlator(self, filename="test_correlator.txt", data=None):
        np.savetxt(filename, data)

    def tearDown(self):
        os.remove(self.correlatorFilename)
        return super().tearDown()


    def test_extractColumns(self):
        x, mean, error, correlators = self.fitRunner.extractColumns(
            "test_correlator.txt", 0, 1, 2, [3]
        )
        np.testing.assert_array_equal(x, self.data[:, 0])
        np.testing.assert_array_equal(mean, self.data[:, 1])
        np.testing.assert_array_equal(error, self.data[:, 2])
        np.testing.assert_array_equal(correlators, self.data[:, 3,None])



    def test_run_fits(self):

        results, loss_histories = self.fitRunner.run_fits()
        self.assertIsNotNone(results)
        self.assertIsNotNone(loss_histories)


    def test_calculate_mean_error(self):
        mean = np.array([1.0, 2.0, 3.0])
        samples = np.array([
            [1.1, 2.1, 3.1],
            [0.9, 1.9, 2.9]
        ])
        error = self.fitRunner.calculate_mean_error(mean, samples, "jackknife")
        expected_error = np.array([0.1, 0.1, 0.1])
        np.testing.assert_almost_equal(error, expected_error, decimal=5)

    def test_save_results(self):
        mean = np.array([1.0, 2.0, 3.0])
        error = np.array([0.1, 0.1, 0.1])
        samples = np.array([
            [1.1, 2.1, 3.1],
            [0.9, 1.9, 2.9]
        ])
        loss_history = np.array([
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4]
        ])
        self.fitRunner.save_results(mean, error, samples, loss_history)
        self.assertTrue(os.path.exists(os.path.join(self.fitRunner.outputDir, self.fitRunner.outputFile)))
        os.remove(os.path.join(self.fitRunner.outputDir, self.fitRunner.outputFile))

    def test_save_results_no_samples_no_error(self):
        mean = np.array([1.0, 2.0, 3.0])
        error = None
        samples = None
        loss_history = np.array([
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4]
        ])
        self.fitRunner.save_results(mean, error, samples, loss_history)
        self.assertTrue(os.path.exists(os.path.join(self.fitRunner.outputDir, self.fitRunner.outputFile)))
        os.remove(os.path.join(self.fitRunner.outputDir, self.fitRunner.outputFile))

    def test_save_loss_history(self):
        loss_history = np.array([
            [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]],
            [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]
        ])
        self.fitRunner.save_loss_history(loss_history, "test_loss_history.dat")
        self.assertTrue(os.path.exists("test_loss_history.dat"))
        os.remove("test_loss_history.dat")

class TestParameterHandler(unittest.TestCase):

    def setUp(self):
        self.paramsDefaultDict = {
            #NetworkParams
            "lambda_s": [1e-5],
            "lambda_l2": [1e-8],
            "epochs": [100],
            "learning_rate": [1e-4],
            "width": [32,32,32],
            "errorWeighting": True,
            "networkStructure": "SpectralNN",
            #Correlator/Rho params
            "omega_min": 0,
            "omega_max": 10,
            "omega_points": 500,
            "Nt": 0,
            "extractedQuantity": "RhoOverOmega",
            "FiniteT_kernel": True,
            "multiFit": False,
            "correlatorFile": "test_correlator.txt",
            "xCol": 0,
            "meanCol": 1,
            "errorCol": 2,
            "correlatorCols": "3:",
            "errormethod": "jackknife",
            #General Params
            "saveParams": False,
            "saveLossHistory": False,
            "verbose": False,
            "outputFile": "",
            "outputDir": ''
        }
        self.parameterHandler = ParameterHandler(self.paramsDefaultDict)
        self.data=np.array([
                        [0.0, 1.0, 0.1, 0.5],
                        [0.1, 1.1, 0.1, 0.6],
                        [0.2, 1.2, 0.1, 0.7]
                    ])
        self.correlatorFilename = "test_correlator.txt"
        np.savetxt(self.correlatorFilename, self.data)

    def tearDown(self):
        os.remove(self.correlatorFilename)
        return super().tearDown()

    def test_load_from_json(self):
        config = {
            "lambda_s": [1e-4],
            "epochs": [200],
            "correlatorFile": "new_correlator.txt"
        }
        with open("test_config.json", "w") as f:
            json.dump(config, f)

        self.parameterHandler.load_from_json("test_config.json")
        params = self.parameterHandler.get_params()
        self.assertEqual(params["lambda_s"], [1e-4])
        self.assertEqual(params["epochs"], [200])
        self.assertEqual(params["correlatorFile"], "new_correlator.txt")
        os.remove("test_config.json")

    def test_override_with_args(self):
        class Args:
            lambda_s = [1e-3]
            learning_rate = [1e-3]
            outputFile = "output.txt"

        args = Args()
        self.parameterHandler.override_with_args(args)
        params = self.parameterHandler.get_params()
        self.assertEqual(params["lambda_s"], [1e-3])
        self.assertEqual(params["learning_rate"], [1e-3])
        self.assertEqual(params["outputFile"], "output.txt")

    def test_check_parameters(self):
        self.parameterHandler.check_parameters()
        self.parameterHandler.params["lambda_s"] = None
        with self.assertRaises(ValueError):
            self.parameterHandler.check_parameters()

    def test_load_params(self):
        config = {
            "lambda_s": [1e-4],
            "epochs": [200],
            "correlatorFile": "new_correlator.txt"
        }
        with open("test_config.json", "w") as f:
            json.dump(config, f)

        class Args:
            learning_rate = [1e-3]
            outputFile = "output.txt"

        args = Args()
        self.parameterHandler.load_params("test_config.json", args)
        params = self.parameterHandler.get_params()
        self.assertEqual(params["lambda_s"], [1e-4])
        self.assertEqual(params["epochs"], [200])
        self.assertEqual(params["correlatorFile"], "new_correlator.txt")
        self.assertEqual(params["learning_rate"], [1e-3])
        self.assertEqual(params["outputFile"], "output.txt")
        os.remove("test_config.json")

    def test_get_width(self):
        self.assertEqual(self.parameterHandler.get_width(), [32, 32, 32])
        self.parameterHandler.params["width"] = 64
        self.assertEqual(self.parameterHandler.get_width(), [64])

    def test_getNetworkParams(self):
        net_params = self.parameterHandler.getNetworkParams()
        self.assertEqual(net_params.lambda_s, [1e-5])
        self.assertEqual(net_params.lambda_l2, [1e-8])
        self.assertEqual(net_params.epochs, [100])
        self.assertEqual(net_params.learning_rate, [1e-4])
        self.assertEqual(net_params.width, [32, 32, 32])
        self.assertEqual(net_params.errorWeighting, True)
        self.assertEqual(net_params.networkStructure, "SpectralNN")

    def test_get_extractedQuantity(self):
        self.assertEqual(self.parameterHandler.get_extractedQuantity(), "RhoOverOmega")

    def test_get_correlator_file(self):
        self.assertEqual(self.parameterHandler.get_correlator_file(), os.path.abspath("test_correlator.txt"))

    def test_get_verbose(self):
        self.assertFalse(self.parameterHandler.get_verbose())

    def test_get_correlator_cols(self):
        self.assertEqual(self.parameterHandler.get_correlator_cols(), list(range(3, 4)))
        self.parameterHandler.params["correlatorCols"] = "3:5"
        self.assertEqual(self.parameterHandler.get_correlator_cols(), list(range(3, 6)))
        self.parameterHandler.params["correlatorCols"] = [3, 4, 5]
        self.assertEqual(self.parameterHandler.get_correlator_cols(), [3, 4, 5])
        self.parameterHandler.params["correlatorCols"] = 3
        self.assertEqual(self.parameterHandler.get_correlator_cols(), [3])
        self.parameterHandler.params["correlatorCols"] = ""
        self.assertEqual(self.parameterHandler.get_correlator_cols(), [])

    def test_check_parameters_missing_required(self):
        self.parameterHandler.params["lambda_s"] = None
        with self.assertRaises(ValueError):
            self.parameterHandler.check_parameters()

    def test_load_params_with_json_and_args(self):
        config = {
            "lambda_s": [1e-4],
            "epochs": [200],
            "correlatorFile": "new_correlator.txt"
        }
        with open("test_config.json", "w") as f:
            json.dump(config, f)

        class Args:
            learning_rate = [1e-3]
            outputFile = "output.txt"

        args = Args()
        self.parameterHandler.load_params("test_config.json", args)
        params = self.parameterHandler.get_params()
        self.assertEqual(params["lambda_s"], [1e-4])
        self.assertEqual(params["epochs"], [200])
        self.assertEqual(params["correlatorFile"], "new_correlator.txt")
        self.assertEqual(params["learning_rate"], [1e-3])
        self.assertEqual(params["outputFile"], "output.txt")
        os.remove("test_config.json")

class TestNetworkTrainer(unittest.TestCase):

    def setUp(self):
        self.x = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32)
        self.y_true = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
        self.std = tf.constant([0.1, 0.1, 0.1], dtype=tf.float32)
        self.kernel = tf.constant([[1.0, 0.5, 0.2], [0.5, 1.0, 0.5], [0.2, 0.5, 1.0]], dtype=tf.float32)
        self.delomega = tf.constant(0.1, dtype=tf.float32)
        model = SpectralNN(num_output_nodes=3, width=[32, 32])
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        lambda_s_func = lambda epoch: 1e-5
        lambda_l2_func = lambda epoch: 1e-8
        loss_calculator = LossCalculator(model=model, y_true=self.y_true, std=self.std, x=self.x, kernel=self.kernel, delomega=self.delomega, lambda_s_func=lambda_s_func, lambda_l2_func=lambda_l2_func)
        self.trainer = networkTrainer(model, optimizer, loss_calculator)

    def test_train_step(self):
        epoch = 0
        total_loss_value, individual_losses = self.trainer.train_step(epoch)
        self.assertIsNotNone(total_loss_value)
        self.assertIsNotNone(individual_losses)

    def test_train(self):
        num_epochs = 5
        losses, individual_losses_history = self.trainer.train(num_epochs, verbose=False)
        self.assertEqual(len(losses), num_epochs)
        self.assertEqual(len(individual_losses_history), num_epochs)
class TestLossCalculator(unittest.TestCase):

    def setUp(self):
        self.model = SpectralNN(num_output_nodes=3, width=[32, 32])
        self.y_true = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
        self.std = tf.constant([0.1, 0.1, 0.1], dtype=tf.float32)
        self.loss_calculator = LossCalculator(model=self.model, y_true=self.y_true, std=self.std)

    def test_get_lambda_s(self):
        epoch = 0
        lambda_s = self.loss_calculator.get_lambda_s(epoch)
        self.assertEqual(lambda_s, 0.0)

    def test_get_lambda_l2(self):
        epoch = 0
        lambda_l2 = self.loss_calculator.get_lambda_l2(epoch)
        self.assertEqual(lambda_l2, 0.0)

    def test_l2_regularization(self):
        weights = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
        result = self.loss_calculator.l2_regularization(weights=weights)
        expected = 7.0  # sum of squares
        self.assertAlmostEqual(result.numpy(), expected, places=5)

    def test_smoothness_loss(self):
        rho = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32)
        result = self.loss_calculator.smoothness_loss(rho)
        expected = 2.0  # example expected value
        self.assertAlmostEqual(result.numpy(), expected, places=5)

    def test_custom_loss(self):
        y_pred = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
        result = self.loss_calculator.custom_loss(y_pred)
        expected = 0.0  # no difference
        self.assertAlmostEqual(result.numpy(), expected, places=5)


if __name__ == '__main__':
    unittest.main()
