import unittest

from backend.platform_api.core.errors import code_for_status, error_body
from backend.platform_api.core.metrics import Metrics


class MetricsText(unittest.TestCase):
    def test_counter_and_histogram_lines(self):
        m = Metrics()
        m.observe_request("POST", "/classify/network", 200, 0.02)
        m.observe_request("POST", "/classify/network", 200, 0.3)
        m.observe_request("POST", "/classify/network", 422, 0.001)
        m.count_classifications("network", "rule", 5)
        text = m.render()
        self.assertIn('app_http_requests_total{method="POST",path="/classify/network",status="200"} 2', text)
        self.assertIn('app_http_requests_total{method="POST",path="/classify/network",status="422"} 1', text)
        self.assertIn('app_http_request_duration_seconds_count{method="POST",path="/classify/network"} 3', text)
        self.assertIn('le="+Inf"} 3', text)
        self.assertIn('le="0.025"} 2', text)       # 0.001 and 0.02 are <= 0.025
        self.assertIn('app_classifications_total{branch="network",source="rule"} 5', text)
        self.assertIn("# TYPE app_http_requests_total counter", text)
        self.assertIn("# TYPE app_http_request_duration_seconds histogram", text)
        self.assertTrue(text.endswith("\n"))

    def test_buckets_are_cumulative(self):
        m = Metrics()
        m.observe_request("GET", "/x", 200, 0.004)
        counts = [int(line.rsplit(" ", 1)[1]) for line in m.render().splitlines() if "_bucket" in line]
        self.assertEqual(counts, sorted(counts))
        self.assertEqual(counts[-1], 1)

    def test_label_values_are_escaped(self):
        m = Metrics()
        m.observe_request("GET", 'a"b', 200, 0.1)
        self.assertIn('path="a\\"b"', m.render())

    def test_empty_registry_renders(self):
        self.assertIn("# HELP", Metrics().render())


class ErrorFormat(unittest.TestCase):
    def test_body_shape(self):
        self.assertEqual(error_body("X", "msg", {"a": 1}), {"error": {"code": "X", "message": "msg", "details": {"a": 1}}})
        self.assertEqual(error_body("X", "msg")["error"]["details"], {})

    def test_status_mapping(self):
        self.assertEqual(code_for_status(415), "UNSUPPORTED_FILE_TYPE")
        self.assertEqual(code_for_status(503), "SERVICE_UNAVAILABLE")
        self.assertEqual(code_for_status(500), "INTERNAL_ERROR")
        self.assertEqual(code_for_status(418), "ERROR")


if __name__ == "__main__":
    unittest.main()
