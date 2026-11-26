"""HTTP-level tests for every endpoint. Skipped automatically where FastAPI / httpx are missing.

IMPORTANT: these were written but NOT run in the environment where the backend was built
(FastAPI could not be installed there). Run them on your machine:

    pytest backend/platform_api/tests -v

Any failure here is a real finding about the HTTP wiring; the logic underneath is covered by
the other test files, which did run.
"""
import json
import os
import unittest
from unittest.mock import patch
from pathlib import Path

from backend.platform_api.tests.fakes import FakeMlflowClient, make_run
from backend.platform_api.tests.support import AppTestCase, csv_bytes, flow, has

SAMPLES = Path(__file__).resolve().parents[2] / "text" / "samples"


@unittest.skipUnless(has("fastapi") and has("httpx"), "fastapi and httpx are required")
class ApiTestCase(AppTestCase):
    def setUp(self):
        super().setUp()
        from fastapi.testclient import TestClient
        from backend.platform_api.core.metrics import metrics
        from backend.platform_api.core.registry_service import RegistryService
        from backend.platform_api.main import create_app
        from backend.platform_api.routers import registry
        import backend.text.model as text_model

        text_model._cached = None
        self.fake = FakeMlflowClient()
        self.fake.add_version("text-classifier", 1, {"f1_macro": 0.9})
        self.fake.add_version("text-classifier", 2, {"f1_macro": 0.95})
        self.fake.aliases["text-classifier"] = {"champion": "1"}
        registry.set_service(RegistryService(client_factory=lambda: self.fake,
                                             reloaders={"text": lambda: {"source": "registry", "version": "2"}},
                                             serving={"text": lambda: {"source": "registry", "version": "1"}}))
        self.metrics = metrics
        self.client = TestClient(create_app(), raise_server_exceptions=False)

    def tearDown(self):
        from backend.platform_api.routers import registry
        from backend.platform_api.core.registry_service import RegistryService
        registry.set_service(RegistryService())
        super().tearDown()

    def assertErr(self, response, status, code):
        self.assertEqual(response.status_code, status, response.text)
        body = response.json()
        self.assertEqual(set(body), {"error"})
        self.assertEqual(body["error"]["code"], code, body)
        self.assertTrue(body["error"]["message"])


class Ops(ApiTestCase):
    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual((r.status_code, r.json()), (200, {"status": "ok"}))

    def test_ready_reports_each_branch(self):
        async def fake_service_status(url, path):
            if path == "/classify/video/health":
                return 200, {
                    "status": "ok",
                    "model": {"loaded": False, "version": "video-v1"},
                }
            if path == "/ready":
                return 200, {
                    "status": "ready",
                    "model": {"loaded": True, "version": "audio-v1"},
                }
            return None, None

        with patch(
            "backend.platform_api.routers.health._get_service_status",
            side_effect=fake_service_status,
        ):
            r = self.client.get("/ready")

        body = r.json()
        self.assertEqual(set(body["checks"]), {
            "network", "text", "video", "audio"
        })
        self.assertTrue(body["checks"]["video"]["ok"])
        self.assertFalse(body["checks"]["video"]["model_loaded"])
        self.assertTrue(body["checks"]["audio"]["ok"])
        self.assertTrue(body["checks"]["audio"]["model_loaded"])

    def test_ready_fails_when_audio_model_is_not_loaded(self):
        async def fake_service_status(url, path):
            if path == "/classify/video/health":
                return 200, {"status": "ok", "model": {"loaded": False}}
            return 503, {
                "status": "not_ready",
                "model": {"loaded": False, "version": None},
            }

        with patch(
            "backend.platform_api.routers.health._get_service_status",
            side_effect=fake_service_status,
        ):
            r = self.client.get("/ready")

        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()["status"], "not_ready")
        self.assertFalse(r.json()["checks"]["audio"]["ok"])
        self.assertTrue(r.json()["checks"]["video"]["ok"])

    def test_metrics_page_counts_requests(self):
        self.client.get("/health")
        self.client.post("/classify/network", files={"file": ("a.csv", b"", "text/csv")})
        text = self.client.get("/metrics").text
        self.assertIn('app_http_requests_total{method="GET",path="/health",status="200"}', text)
        self.assertIn('path="/classify/network",status="400"', text)
        self.assertIn("app_http_request_duration_seconds_bucket", text)

    def test_cors_header_for_frontend_origin(self):
        r = self.client.get("/health", headers={"Origin": "http://localhost:5173"})
        self.assertEqual(r.headers.get("access-control-allow-origin"), "http://localhost:5173")

    def test_unknown_route_uses_error_format(self):
        self.assertErr(self.client.get("/nope"), 404, "NOT_FOUND")

    def test_structured_text_health_error_preserves_details(self):
        from fastapi import APIRouter, HTTPException
        from backend.platform_api.main import create_app
        router = APIRouter(prefix="/test")
        @router.get("/structured-error")
        def structured_error():
            raise HTTPException(status_code=503, detail={"status": "unavailable", "checks": {"model": "missing"}})
        app = create_app()
        app.include_router(router)
        from fastapi.testclient import TestClient
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/test/structured-error")
        self.assertEqual(response.status_code, 503)
        body = response.json()["error"]
        self.assertEqual(body["code"], "SERVICE_UNAVAILABLE")
        self.assertEqual(body["details"]["status"], "unavailable")
        self.assertEqual(body["details"]["checks"]["model"], "missing")


class NetworkEndpoint(ApiTestCase):
    def post(self, name, data, ctype="text/csv"):
        return self.client.post("/classify/network", files={"file": (name, data, ctype)})

    def test_csv_ok(self):
        r = self.post("f.csv", csv_bytes([flow(dst=443), flow(src=40000, dst=5555)]))
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual([x["label"] for x in body["results"]], ["HTTPS", "Unknown"])
        self.assertEqual(body["results"][1]["source"], "default")
        self.assertEqual(body["summary"]["n_flows"], 2)

    def test_counts_classifications_metric(self):
        self.post("f.csv", csv_bytes([flow(dst=443)]))
        self.assertIn('app_classifications_total{branch="network",source="rule"}', self.client.get("/metrics").text)

    def test_empty(self):
        self.assertErr(self.post("a.csv", b""), 400, "EMPTY_FILE")

    def test_wrong_type(self):
        self.assertErr(self.post("a.docx", b"PK\x03\x04", "application/zip"), 415, "UNSUPPORTED_FILE_TYPE")

    def test_oversized(self):
        os.environ["APP_MAX_UPLOAD_BYTES"] = "50"
        self.assertErr(self.post("a.csv", b"x" * 500), 413, "FILE_TOO_LARGE")

    def test_missing_columns(self):
        self.assertErr(self.post("a.csv", csv_bytes([flow()], ["src_port", "dst_port"])), 422, "MISSING_COLUMNS")

    def test_no_file_field(self):
        self.assertErr(self.client.post("/classify/network"), 422, "INVALID_REQUEST")


class TextEndpointThroughTheApp(ApiTestCase):
    def test_txt_document(self):
        path = next((SAMPLES / "invoice").glob("*"), None) or next(SAMPLES.glob("*/*"))
        r = self.client.post("/classify/text", files={"file": (path.name, path.read_bytes(), "application/octet-stream")})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIn("category", r.json())

    def test_text_errors_use_the_shared_error_format(self):
        r = self.client.post("/classify/text", files={"file": ("a.txt", b"", "text/plain")})
        self.assertErr(r, 400, "BAD_REQUEST")
        r = self.client.post("/classify/text", files={"file": ("a.exe", b"MZ\x00\x01", "application/octet-stream")})
        self.assertErr(r, 415, "UNSUPPORTED_FILE_TYPE")

    def test_text_oversized(self):
        os.environ["TEXT_MAX_BYTES"] = "10"
        r = self.client.post("/classify/text", files={"file": ("a.txt", b"x" * 100, "text/plain")})
        self.assertErr(r, 413, "FILE_TOO_LARGE")

    def test_text_model_missing_is_503(self):
        os.environ["TEXT_MODEL_DIR"] = self._tmp.name + "/does-not-exist"
        os.environ["TEXT_MODEL_SOURCE"] = "local"
        r = self.client.post("/classify/text", files={"file": ("a.txt", b"hello " * 50, "text/plain")})
        os.environ.pop("TEXT_MODEL_DIR")
        self.assertErr(r, 503, "SERVICE_UNAVAILABLE")


class RegistryEndpoints(ApiTestCase):
    def test_list(self):
        body = self.client.get("/registry/models").json()
        self.assertTrue(body["available"])
        self.assertEqual({m["name"] for m in body["models"]}, {"text-classifier", "network-classifier"})

    def test_list_without_mlflow_is_200_with_banner(self):
        from backend.platform_api.routers import registry
        from backend.platform_api.core.registry_service import RegistryService
        registry.set_service(RegistryService())
        r = self.client.get("/registry/models")
        self.assertEqual(r.status_code, 200)
        self.assertFalse(r.json()["available"])

    def test_compare(self):
        r = self.client.get("/registry/models/text-classifier/compare", params={"versions": "1,2"})
        self.assertEqual(r.json()["metrics"]["f1_macro"], {"1": 0.9, "2": 0.95})
        self.assertErr(self.client.get("/registry/models/text-classifier/compare", params={"versions": "1"}),
                       400, "NEED_TWO_VERSIONS")
        self.assertErr(self.client.get("/registry/models/text-classifier/compare"), 422, "INVALID_REQUEST")

    def test_promote_needs_confirmation(self):
        r = self.client.post("/registry/models/text-classifier/promote", json={"version": "2"})
        self.assertErr(r, 400, "CONFIRMATION_REQUIRED")
        self.assertEqual(self.fake.aliases["text-classifier"]["champion"], "1")

    def test_promote_then_activity_then_rollback(self):
        r = self.client.post("/registry/models/text-classifier/promote",
                             json={"version": "2", "confirm": True, "actor": "ullas"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self.fake.aliases["text-classifier"]["champion"], "2")
        log = self.client.get("/registry/activity", params={"model": "text-classifier"}).json()["entries"]
        self.assertEqual((log[0]["action"], log[0]["actor"]), ("promote", "ullas"))
        r = self.client.post("/registry/models/text-classifier/rollback", json={"confirm": True, "actor": "sam"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self.fake.aliases["text-classifier"]["champion"], "1")

    def test_promote_errors(self):
        url = "/registry/models/{}/promote"
        self.assertErr(self.client.post(url.format("text-classifier"), json={"version": "9", "confirm": True}),
                       404, "VERSION_NOT_FOUND")
        self.assertErr(self.client.post(url.format("network-classifier"), json={"version": "1", "confirm": True}),
                       409, "NOT_WIRED_TO_REGISTRY")
        self.assertErr(self.client.post(url.format("text-classifier"), json={"confirm": True}), 422, "INVALID_REQUEST")

    def test_reload(self):
        self.assertTrue(self.client.post("/registry/models/text-classifier/reload").json()["reloaded"])
        self.assertFalse(self.client.post("/registry/models/network-classifier/reload").json()["reloaded"])

    def test_network_runs(self):
        self.fake.add_experiment("network-known-unknown", [make_run("r1", "RF", {"recall_Unknown": 0.0})])
        body = self.client.get("/registry/network/runs").json()
        self.assertEqual(body["runs"][0]["metrics"]["recall_Unknown"], 0.0)

    def test_activity_empty(self):
        self.assertEqual(self.client.get("/registry/activity").json(), {"entries": []})


if __name__ == "__main__":
    unittest.main()
