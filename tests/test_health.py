import unittest
import time
import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from evocomp.core.health import HealthChecker, HealthStatus

class TestHealthChecker(unittest.TestCase):
    def setUp(self):
        self.checker = HealthChecker(node_id=1, cluster_size=1)
        # Stop the background thread to prevent noise/errors during tests if possible,
        # but the thread is daemon, so it should be fine.

    def test_initialization(self):
        self.assertIn("system", self.checker.components)
        self.assertIn("redis", self.checker.components)
        self.assertEqual(self.checker.node_id, 1)

    def test_system_metrics(self):
        # Force update
        self.checker._update_component_health("system")
        system_health = self.checker.get_component_health("system")
        self.assertIsNotNone(system_health)
        self.assertTrue(len(system_health.metrics) > 0)
        self.assertIn("cpu", system_health.metrics)
        self.assertIn("memory", system_health.metrics)
        # Status should be HEALTHY if resources are not exhausted
        # This might be flaky if the test runner is under heavy load, but we can check it's not UNKNOWN
        self.assertNotEqual(system_health.status, HealthStatus.UNKNOWN)

    def test_dependency_propagation(self):
        # Redis depends on system.
        # Ensure system is updated
        self.checker._update_component_health("system")
        system_status = self.checker.get_component_health("system").status

        # Update redis
        self.checker._update_component_health("redis")
        redis_health = self.checker.get_component_health("redis")

        if system_status == HealthStatus.HEALTHY:
            self.assertEqual(redis_health.status, HealthStatus.HEALTHY)
        else:
            # If system is not healthy, redis should reflect that or be UNHEALTHY
            self.assertNotEqual(redis_health.status, HealthStatus.HEALTHY)

    def test_custom_check(self):
        self.checker.register_component("custom_comp")
        self.checker.register_custom_check("custom_comp", lambda: True)
        self.checker._update_component_health("custom_comp")
        comp = self.checker.get_component_health("custom_comp")
        self.assertEqual(comp.status, HealthStatus.HEALTHY)

        self.checker.register_custom_check("custom_comp", lambda: False)
        self.checker._update_component_health("custom_comp")
        comp = self.checker.get_component_health("custom_comp")
        self.assertEqual(comp.status, HealthStatus.UNHEALTHY)

    def test_deep_dependency(self):
        # evolution_engine depends on redis, which depends on system
        self.checker._update_component_health("system")
        self.checker._update_component_health("redis")
        self.checker._update_component_health("evolution_engine")

        evo_health = self.checker.get_component_health("evolution_engine")
        redis_health = self.checker.get_component_health("redis")

        if redis_health.status == HealthStatus.HEALTHY:
            self.assertEqual(evo_health.status, HealthStatus.HEALTHY)
        else:
            self.assertNotEqual(evo_health.status, HealthStatus.HEALTHY)
