import unittest
import time
from src.evocomp.core.health import HealthChecker

class TestHealthChecker(unittest.TestCase):
    def setUp(self):
        self.checker = HealthChecker(interval=0.1)

    def tearDown(self):
        self.checker.stop()

    def test_lifecycle(self):
        self.checker.start()
        time.sleep(0.3) # Wait for thread to loop at least once
        metrics = self.checker.check_health()
        self.assertTrue('cpu' in metrics, "CPU metric missing")
        self.assertTrue('memory' in metrics, "Memory metric missing")
        self.assertTrue('disk' in metrics, "Disk metric missing")
        self.assertTrue('gpu' in metrics, "GPU metric missing")
        self.checker.stop()
        self.assertFalse(self.checker.thread.is_alive())

    def test_health_threshold(self):
        self.checker.start()
        time.sleep(0.3)
        # Assuming current environment is healthy enough (metrics < 1.0)
        self.assertTrue(self.checker.is_healthy(threshold=1.0))
        # If we set threshold to -0.1, it should be unhealthy (metrics > -0.1)
        self.assertFalse(self.checker.is_healthy(threshold=-0.1))

if __name__ == '__main__':
    unittest.main()
