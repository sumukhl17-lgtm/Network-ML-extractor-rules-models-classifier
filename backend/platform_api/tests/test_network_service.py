"""Network upload handling. Uses the REAL rules.py, classify_network.py and rf.joblib."""
import os
import unittest
from unittest import mock

from backend.platform_api.core import network_adapter as adapter
from backend.platform_api.core import network_service as svc
from backend.platform_api.core.errors import AppError
from backend.platform_api.tests.support import AppTestCase, csv_bytes, flow, has

PCAP_HEADER = b"\xd4\xc3\xb2\xa1" + b"\x00" * 20


class CsvClassification(AppTestCase):
    def test_port_rule_wins(self):
        out = svc.classify_upload("flows.csv", csv_bytes([flow(dst=443), flow(dst=53), flow(src=22, dst=60000), flow(dst=80)]))
        labels = [r["label"] for r in out["results"]]
        self.assertEqual(labels, ["HTTPS", "DNS", "SSH", "HTTP"])
        self.assertTrue(all(r["source"] == "rule" and not r["advisory"] for r in out["results"]))
        self.assertEqual(out["input_kind"], "csv")
        self.assertEqual(out["summary"]["n_flows"], 4)
        self.assertEqual(out["summary"]["n_unknown"], 0)

    def test_unmatched_flow_is_unknown_with_advisory_ml(self):
        out = svc.classify_upload("f.csv", csv_bytes([flow(src=40000, dst=5555)]))
        r = out["results"][0]
        self.assertEqual((r["label"], r["source"]), ("Unknown", "default"))
        self.assertTrue(r["is_unknown"] and r["advisory"])
        self.assertIn(r["ml_label"], ("Known", "Unknown"))
        self.assertTrue(0.0 <= r["ml_confidence"] <= 1.0)
        self.assertIsNone(r["confidence"])
        self.assertEqual(out["mode"], "rules_with_advisory_ml")
        self.assertEqual(out["summary"]["n_advisory"], 1)

    def test_ml_overrides_mode(self):
        import classify_network
        with mock.patch.object(classify_network, "ML_OVERRIDES_UNMATCHED", True):
            out = svc.classify_upload("f.csv", csv_bytes([flow(src=40000, dst=5555)]))
        r = out["results"][0]
        self.assertEqual(r["source"], "ml")
        self.assertIn(r["label"], ("Known", "Unknown"))
        self.assertTrue(0.0 <= r["confidence"] <= 1.0 and r["advisory"])
        self.assertEqual(out["mode"], "ml_overrides_unmatched")

    def test_float_ports_from_csv_still_match_rules(self):
        out = svc.classify_upload("f.csv", csv_bytes([flow(dst="443.0")]))
        self.assertEqual(out["results"][0]["label"], "HTTPS")

    def test_extra_columns_are_ignored_and_application_name_not_returned(self):
        header = ["src_port", "dst_port", "packet_count", "total_bytes", "avg_packet_size", "duration",
                  "packet_rate", "byte_rate", "application_name", "syn"]
        data = csv_bytes([dict(flow(), application_name="TLS.Google", syn=1)], header)
        out = svc.classify_upload("f.csv", data)
        self.assertNotIn("application_name", out["results"][0])

    def test_response_matches_schema_fields(self):
        out = svc.classify_upload("f.csv", csv_bytes([flow()]))
        for key in ("index", "kind", "src_port", "dst_port", "label", "is_unknown", "source", "confidence",
                    "advisory", "ml_label", "ml_confidence"):
            self.assertIn(key, out["results"][0])


class CsvErrors(AppTestCase):
    def assertError(self, status, code, fn, *args):
        with self.assertRaises(AppError) as cm:
            fn(*args)
        self.assertEqual((cm.exception.status, cm.exception.code), (status, code))
        return cm.exception

    def test_empty_file(self):
        self.assertError(400, "EMPTY_FILE", svc.classify_upload, "a.csv", b"")

    def test_oversized_file(self):
        os.environ["APP_MAX_UPLOAD_BYTES"] = "100"
        self.assertError(413, "FILE_TOO_LARGE", svc.classify_upload, "a.csv", b"x" * 101)

    def test_too_many_flows(self):
        os.environ["APP_MAX_FLOWS"] = "2"
        self.assertError(413, "TOO_MANY_FLOWS", svc.classify_upload, "a.csv", csv_bytes([flow(), flow(), flow()]))

    def test_missing_columns_lists_them(self):
        err = self.assertError(422, "MISSING_COLUMNS", svc.classify_upload, "a.csv",
                               csv_bytes([flow()], ["src_port", "dst_port", "packet_count"]))
        self.assertIn("byte_rate", err.details["missing"])

    def test_non_numeric_value_names_row_and_column(self):
        err = self.assertError(422, "INVALID_FLOW_VALUE", svc.classify_upload, "a.csv",
                               csv_bytes([flow(), flow(duration="fast")]))
        self.assertEqual((err.details["row"], err.details["column"]), (3, "duration"))

    def test_blank_value_is_invalid(self):
        self.assertError(422, "INVALID_FLOW_VALUE", svc.classify_upload, "a.csv", csv_bytes([flow(byte_rate="")]))

    def test_nan_is_invalid(self):
        self.assertError(422, "INVALID_FLOW_VALUE", svc.classify_upload, "a.csv", csv_bytes([flow(byte_rate="nan")]))

    def test_header_only(self):
        self.assertError(422, "NO_FLOWS", svc.classify_upload, "a.csv", csv_bytes([]))

    def test_wrong_type(self):
        self.assertError(415, "UNSUPPORTED_FILE_TYPE", svc.classify_upload, "notes.docx", b"PK\x03\x04 hello")

    def test_ipfix_is_refused_with_explanation(self):
        err = self.assertError(415, "UNSUPPORTED_FILE_TYPE", svc.classify_upload, "export.ipfix", b"\x00\x0a\x00\x10")
        self.assertIn("IPFIX", err.message)

    def test_binary_csv(self):
        self.assertError(422, "INVALID_CSV", svc.classify_upload, "a.csv", b"a,b\x00\x01\x02,c")

    def test_non_utf8_csv(self):
        self.assertError(422, "INVALID_CSV", svc.classify_upload, "a.csv", b"src_port,dst_port\n\xff\xfe,1")


class PcapPath(AppTestCase):
    def test_pcap_extension_with_junk_content(self):
        with self.assertRaises(AppError) as cm:
            svc.classify_upload("capture.pcap", b"this is not a capture at all")
        self.assertEqual(cm.exception.code, "INVALID_PCAP")

    def test_pcap_magic_bytes_detected_even_without_extension(self):
        self.assertEqual(svc.detect_kind("capture", PCAP_HEADER), "pcap")
        self.assertEqual(svc.detect_kind("x.pcapng", b"\x0a\x0d\x0d\x0a" + b"\0" * 8), "pcap")

    def test_missing_nfstream_gives_clear_503(self):
        if has("nfstream"):
            self.skipTest("NFStream is installed here; this checks the not-installed message")
        with self.assertRaises(AppError) as cm:
            svc.classify_upload("c.pcap", PCAP_HEADER)
        self.assertEqual((cm.exception.status, cm.exception.code), (503, "EXTRACTOR_UNAVAILABLE"))

    @unittest.skipUnless(has("pandas"), "pandas needed")
    def test_dataframe_from_extractor_becomes_results(self):
        # Same columns as extract_flows.extract() returns. NFStream itself is replaced by this fake.
        import pandas as pd
        frame = pd.DataFrame([
            dict(src_port=51000, dst_port=443, protocol=6, packet_count=30, total_bytes=9000, avg_packet_size=300.0,
                 duration=1.5, packet_rate=20.0, byte_rate=6000.0, syn=1, fin=1, rst=0, psh=4, ack=29,
                 application_name="TLS.Google", application_is_guessed=0),
            dict(src_port=40000, dst_port=9999, protocol=17, packet_count=5, total_bytes=700, avg_packet_size=140.0,
                 duration=0.2, packet_rate=25.0, byte_rate=3500.0, syn=0, fin=0, rst=0, psh=0, ack=0,
                 application_name="Unknown", application_is_guessed=0),
        ])
        with mock.patch.object(adapter, "extract_pcap", return_value=frame):
            out = svc.classify_upload("c.pcap", PCAP_HEADER)
        self.assertEqual(out["input_kind"], "pcap")
        self.assertEqual([r["label"] for r in out["results"]], ["HTTPS", "Unknown"])
        self.assertEqual(out["results"][1]["source"], "default")
        self.assertNotIn("application_name", out["results"][0])

    @unittest.skipUnless(has("pandas"), "pandas needed")
    def test_empty_capture(self):
        import pandas as pd
        with mock.patch.object(adapter, "extract_pcap", return_value=pd.DataFrame()):
            with self.assertRaises(AppError) as cm:
                svc.classify_upload("c.pcap", PCAP_HEADER)
        self.assertEqual(cm.exception.code, "NO_FLOWS")

    def test_extractor_crash_is_a_422_not_a_500(self):
        with mock.patch.object(adapter, "extract_pcap", side_effect=RuntimeError("bad capture")):
            with self.assertRaises(AppError) as cm:
                svc.classify_upload("c.pcap", PCAP_HEADER)
        self.assertEqual((cm.exception.status, cm.exception.code), (422, "PCAP_EXTRACTION_FAILED"))


class AdapterErrors(AppTestCase):
    def test_model_features_come_from_the_model(self):
        self.assertEqual(adapter.model_features(),
                         ["packet_count", "total_bytes", "avg_packet_size", "duration", "packet_rate", "byte_rate"])

    def test_unloadable_network_model_is_503(self):
        adapter.reset_for_tests()
        with mock.patch("importlib.import_module", side_effect=ImportError("no module")):
            with self.assertRaises(AppError) as cm:
                adapter.model_features()
        self.assertEqual((cm.exception.status, cm.exception.code), (503, "NETWORK_MODEL_UNAVAILABLE"))


if __name__ == "__main__":
    unittest.main()
