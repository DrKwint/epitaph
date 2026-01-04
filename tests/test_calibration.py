import jax
import jax.numpy as jnp
import numpy as np
import unittest
from epitaph.calibration import compute_gaussian_metrics, compute_coverage

class TestCalibration(unittest.TestCase):
    def test_gaussian_metrics_perfect(self):
        # If mean is exactly target and variance is 1, NLL has a fixed value
        N, D = 100, 2
        S = 100
        y_true = jnp.zeros((N, D))
        
        # Samples from N(0, 1)
        key = jax.random.PRNGKey(0)
        y_preds = jax.random.normal(key, (S, N, D))
        
        metrics = compute_gaussian_metrics(y_true, y_preds)
        
        # RMSE should be low (it's the RMSE of the mean prediction, which is ~0)
        self.assertLess(metrics["rmse"], 0.1)
        
        # Average variance should be ~1.0
        self.assertAlmostEqual(metrics["avg_var"], 1.0, places=1)
        
        # NLL for N(0, 1) at x=0 is 0.5 * log(2*pi) + 0 ~ 0.9189
        self.assertAlmostEqual(metrics["nll"], 0.9189, places=1)

    def test_coverage_gaussian(self):
        N, D = 1000, 1
        S = 500
        y_true = jnp.zeros((N, D))
        
        key = jax.random.PRNGKey(42)
        y_preds = jax.random.normal(key, (S, N, D))
        
        # 95% coverage check
        results = compute_coverage(y_true, y_preds, quantiles=[0.95])
        
        # Since samples are N(0,1) and target is 0, mean is ~0, std is ~1
        # Target 0 is always inside the 95% interval [-1.96, 1.96]? 
        # Wait, PICP is usually: what % of true values fall in their respective intervals.
        # Here true value is 0, and interval is ~[-1.96, 1.96]. So 100% coverage.
        self.assertEqual(results["coverage_95"], 1.0)
        
        # Let's try a more realistic case: targets are also sampled from N(0, 1)
        y_true_rand = jax.random.normal(jax.random.PRNGKey(7), (N, D))
        results_rand = compute_coverage(y_true_rand, y_preds, quantiles=[0.95])
        
        # Should be close to 0.95
        self.assertGreater(results_rand["coverage_95"], 0.90)
        self.assertLess(results_rand["coverage_95"], 0.99)

if __name__ == "__main__":
    unittest.main()
