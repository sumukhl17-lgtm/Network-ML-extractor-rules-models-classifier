import os
import unittest

from backend.platform_api.core import activity
from backend.platform_api.core.errors import AppError
from backend.platform_api.core.registry_service import RegistryService
from backend.platform_api.tests.fakes import FakeMlflowClient, make_run
from backend.platform_api.tests.support import AppTestCase

TEXT = "text-classifier"
NET = "network-classifier"


class RegistryTestCase(AppTestCase):
    def setUp(self):
        super().setUp()
        self.client = FakeMlflowClient()
        self.client.add_version(TEXT, 1, {"f1_macro": 0.90, "accuracy": 0.91})
        self.client.add_version(TEXT, 2, {"f1_macro": 0.95, "accuracy": 0.96, "cv_f1_macro_mean": 0.94})
        self.client.add_version(TEXT, 3, {"f1_macro": 0.93})
        self.client.aliases[TEXT] = {"champion": "2"}
        self.reloads = []
        self.live = {"source": "registry", "version": "2"}

        def reload_text():
            self.reloads.append(1)
            return dict(self.live)

        self.svc = RegistryService(client_factory=lambda: self.client,
                                   reloaders={"text": reload_text},
                                   serving={"text": lambda: dict(self.live)})

    def code(self, fn, *a, **k):
        with self.assertRaises(AppError) as cm:
            fn(*a, **k)
        return cm.exception.status, cm.exception.code


class Listing(RegistryTestCase):
    def test_lists_versions_newest_first_with_champion_and_metrics(self):
        out = self.svc.list_models()
        self.assertTrue(out["available"])
        text = next(m for m in out["models"] if m["name"] == TEXT)
        self.assertEqual([v["version"] for v in text["versions"]], ["3", "2", "1"])
        self.assertEqual(text["champion"], "2")
        self.assertEqual([v["is_champion"] for v in text["versions"]], [False, True, False])
        self.assertEqual(text["versions"][1]["metrics"]["f1_macro"], 0.95)
        self.assertTrue(text["can_promote"])
        self.assertTrue(text["serving"]["wired_to_registry"])

    def test_network_model_listed_but_not_promotable_and_flagged_unwired(self):
        out = self.svc.list_models()
        net = next(m for m in out["models"] if m["name"] == NET)
        self.assertFalse(net["registered"])
        self.assertFalse(net["can_promote"])
        self.assertFalse(net["serving"]["wired_to_registry"])
        self.assertEqual(net["serving"]["source"], "repo-file")

    def test_unknown_other_registered_models_are_included(self):
        self.client.add_version("audio-model", 1, {})
        names = [m["name"] for m in self.svc.list_models()["models"]]
        self.assertIn("audio-model", names)

    def test_registry_not_configured_is_a_banner_not_an_error(self):
        out = RegistryService().list_models()          # real factory, MLFLOW_TRACKING_URI unset
        self.assertFalse(out["available"])
        self.assertIn("MLFLOW_TRACKING_URI", out["reason"])
        self.assertEqual(out["models"], [])

    def test_unreachable_server_is_a_banner(self):
        def boom():
            raise ConnectionError("connection refused")
        out = RegistryService(client_factory=boom).list_models()
        self.assertFalse(out["available"])
        self.assertIn("connection refused", out["reason"])

    def test_serving_failure_is_reported_not_raised(self):
        def broken():
            raise FileNotFoundError("no model")
        svc = RegistryService(client_factory=lambda: self.client, serving={"text": broken})
        text = next(m for m in svc.list_models()["models"] if m["name"] == TEXT)
        self.assertEqual(text["serving"]["source"], "error")

    def test_text_serving_exposes_real_data_validation_status(self):
        live = {"source": "local", "version": "local-demo", "data_source": "synthetic-demo",
                "real_data_validated": False, "evaluation_note": "DEMO MODEL", "unknown_threshold": 0.5,
                "metrics": {"f1_macro": 1.0}}
        svc = RegistryService(client_factory=lambda: self.client, serving={"text": lambda: live})
        text = next(m for m in svc.list_models()["models"] if m["name"] == TEXT)
        self.assertFalse(text["serving"]["real_data_validated"])
        self.assertEqual(text["serving"]["data_source"], "synthetic-demo")
        self.assertEqual(text["serving"]["evaluation_note"], "DEMO MODEL")
        self.assertEqual(text["serving"]["metrics"]["f1_macro"], 1.0)


class Compare(RegistryTestCase):
    def test_side_by_side(self):
        out = self.svc.compare(TEXT, ["1", "2"])
        self.assertEqual(out["metrics"]["f1_macro"], {"1": 0.90, "2": 0.95})
        self.assertIsNone(out["metrics"]["cv_f1_macro_mean"]["1"])      # metric only logged for v2
        self.assertEqual(out["champion"], "2")

    def test_needs_two_versions(self):
        self.assertEqual(self.code(self.svc.compare, TEXT, ["1"]), (400, "NEED_TWO_VERSIONS"))

    def test_unknown_version_and_model(self):
        self.assertEqual(self.code(self.svc.compare, TEXT, ["1", "9"]), (404, "VERSION_NOT_FOUND"))
        self.assertEqual(self.code(self.svc.compare, "nope", ["1", "2"]), (404, "MODEL_NOT_FOUND"))

    def test_bad_name_rejected(self):
        self.assertEqual(self.code(self.svc.compare, "a'b", ["1", "2"]), (400, "INVALID_MODEL_NAME"))


class Promote(RegistryTestCase):
    def test_requires_explicit_confirmation(self):
        self.assertEqual(self.code(self.svc.promote, TEXT, "3", "sam", False), (400, "CONFIRMATION_REQUIRED"))
        self.assertEqual(self.client.alias_calls, [])           # nothing moved

    def test_promote_moves_alias_logs_it_and_reloads(self):
        out = self.svc.promote(TEXT, "3", "ullas", True)
        self.assertEqual(self.client.aliases[TEXT]["champion"], "3")
        self.assertEqual(out["activity"]["from_version"], "2")
        self.assertEqual(out["activity"]["to_version"], "3")
        self.assertEqual(out["activity"]["actor"], "ullas")
        self.assertTrue(out["reload"]["reloaded"])
        self.assertEqual(len(self.reloads), 1)
        log = activity.read(model=TEXT)
        self.assertEqual((log[0]["action"], log[0]["actor"]), ("promote", "ullas"))

    def test_reload_can_be_skipped(self):
        out = self.svc.promote(TEXT, "3", "x", True, reload=False)
        self.assertIsNone(out["reload"])
        self.assertEqual(self.reloads, [])

    def test_reload_failure_does_not_undo_the_promotion(self):
        def bad():
            raise RuntimeError("cannot load")
        svc = RegistryService(client_factory=lambda: self.client, reloaders={"text": bad})
        out = svc.promote(TEXT, "3", "x", True)
        self.assertEqual(self.client.aliases[TEXT]["champion"], "3")
        self.assertFalse(out["reload"]["reloaded"])
        self.assertIn("cannot load", out["reload"]["error"])

    def test_errors(self):
        self.assertEqual(self.code(self.svc.promote, TEXT, "2", "x", True), (409, "ALREADY_CHAMPION"))
        self.assertEqual(self.code(self.svc.promote, TEXT, "99", "x", True), (404, "VERSION_NOT_FOUND"))
        self.assertEqual(self.code(self.svc.promote, "ghost", "1", "x", True), (404, "MODEL_NOT_FOUND"))
        self.assertEqual(self.code(self.svc.promote, NET, "1", "x", True), (409, "NOT_WIRED_TO_REGISTRY"))

    def test_first_promotion_when_there_is_no_champion(self):
        self.client.aliases[TEXT] = {}
        out = self.svc.promote(TEXT, "1", "x", True)
        self.assertIsNone(out["activity"]["from_version"])


class Rollback(RegistryTestCase):
    def test_rollback_returns_to_previous_champion(self):
        self.svc.promote(TEXT, "3", "a", True)
        out = self.svc.rollback(TEXT, "b", True)
        self.assertEqual(self.client.aliases[TEXT]["champion"], "2")
        self.assertEqual(out["activity"]["action"], "rollback")
        self.assertEqual([e["action"] for e in activity.read(model=TEXT)], ["rollback", "promote"])

    def test_requires_confirmation(self):
        self.svc.promote(TEXT, "3", "a", True)
        self.assertEqual(self.code(self.svc.rollback, TEXT, "b", False), (400, "CONFIRMATION_REQUIRED"))
        self.assertEqual(self.client.aliases[TEXT]["champion"], "3")

    def test_no_history_means_choose_a_version(self):
        self.assertEqual(self.code(self.svc.rollback, TEXT, "b", True), (409, "NO_PREVIOUS_VERSION"))

    def test_explicit_target(self):
        out = self.svc.rollback(TEXT, "b", True, to_version="1")
        self.assertEqual(self.client.aliases[TEXT]["champion"], "1")
        self.assertEqual(out["activity"]["from_version"], "2")

    def test_target_errors(self):
        self.assertEqual(self.code(self.svc.rollback, TEXT, "b", True, "42"), (404, "VERSION_NOT_FOUND"))
        self.assertEqual(self.code(self.svc.rollback, TEXT, "b", True, "2"), (409, "ALREADY_CHAMPION"))

    def test_no_champion(self):
        self.client.aliases[TEXT] = {}
        self.assertEqual(self.code(self.svc.rollback, TEXT, "b", True), (409, "NO_CHAMPION"))

    def test_network_not_rollbackable(self):
        self.assertEqual(self.code(self.svc.rollback, NET, "b", True), (409, "NOT_WIRED_TO_REGISTRY"))


class ReloadAndRuns(RegistryTestCase):
    def test_reload_text(self):
        out = self.svc.reload(TEXT)
        self.assertTrue(out["reloaded"])
        self.assertEqual(out["serving"]["version"], "2")

    def test_reload_network_explains_why_not(self):
        out = self.svc.reload(NET)
        self.assertFalse(out["reloaded"])
        self.assertIn("rf.joblib", out["reason"])

    def test_reload_failure_is_503(self):
        def bad():
            raise RuntimeError("x")
        svc = RegistryService(client_factory=lambda: self.client, reloaders={"text": bad})
        self.assertEqual(self.code(svc.reload, TEXT), (503, "RELOAD_FAILED"))

    def test_network_runs_listed_with_metrics(self):
        self.client.add_experiment("network-known-unknown", [
            make_run("r1", "Random Forest-x", {"recall_Unknown": 0.0, "f1_Unknown": 0.0, "roc_auc": 0.5},
                     {"features": "[\"packet_count\"]", "other": "ignored"}),
        ])
        out = self.svc.network_runs()
        self.assertTrue(out["found"])
        self.assertEqual(out["runs"][0]["metrics"]["recall_Unknown"], 0.0)
        self.assertEqual(list(out["runs"][0]["params"]), ["features"])

    def test_network_runs_when_experiment_missing(self):
        out = self.svc.network_runs()
        self.assertEqual((out["found"], out["runs"]), (False, []))


class ActivityLog(AppTestCase):
    def test_newest_first_filter_limit_and_corrupt_line(self):
        activity.record("promote", "m1", "a", None, "1")
        activity.record("promote", "m2", "b", None, "1")
        activity.record("rollback", "m1", "c", "1", "0")
        with open(os.path.join(os.environ["APP_DATA_DIR"], "activity.jsonl"), "a") as f:
            f.write("{not json\n")
        self.assertEqual([e["actor"] for e in activity.read()], ["c", "b", "a"])
        self.assertEqual([e["actor"] for e in activity.read(model="m1")], ["c", "a"])
        self.assertEqual(len(activity.read(limit=1)), 1)

    def test_empty(self):
        self.assertEqual(activity.read(), [])

    def test_blank_actor_becomes_anonymous(self):
        self.assertEqual(activity.record("promote", "m", "", None, "1")["actor"], "anonymous")


if __name__ == "__main__":
    unittest.main()
