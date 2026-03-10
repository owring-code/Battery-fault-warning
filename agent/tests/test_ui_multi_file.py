import os
import tempfile
import unittest

import ui


class MultiFileSessionTests(unittest.TestCase):
    def test_ensure_session_record_adds_uploaded_files_field(self):
        sessions = {"s1": {"title": "demo", "history": []}}
        record = ui.ensure_session_record(sessions, "s1")
        self.assertIn("uploaded_files", record)
        self.assertEqual(record["uploaded_files"], [])

    def test_register_uploaded_file_appends_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "alpha.csv")
            with open(path, "w", encoding="utf-8") as f:
                f.write("a,b\n1,2\n")

            sessions = {}
            sessions, session_id, file_meta = ui.register_uploaded_file(sessions, "", path)
            self.assertTrue(session_id)
            self.assertEqual(file_meta["name"], "alpha.csv")
            self.assertEqual(len(sessions[session_id]["uploaded_files"]), 1)

    def test_resolve_file_reference_by_name(self):
        files = [
            {"id": "1", "name": "alpha.csv", "path": r"D:\\data\\alpha.csv", "uploaded_at": 1},
            {"id": "2", "name": "beta.csv", "path": r"D:\\data\\beta.csv", "uploaded_at": 2},
        ]
        result = ui.resolve_file_for_message("请分析 beta.csv 的趋势", files)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["file"]["name"], "beta.csv")

    def test_resolve_file_reference_by_ordinal(self):
        files = [
            {"id": "1", "name": "alpha.csv", "path": r"D:\\data\\alpha.csv", "uploaded_at": 1},
            {"id": "2", "name": "beta.csv", "path": r"D:\\data\\beta.csv", "uploaded_at": 2},
        ]
        result = ui.resolve_file_for_message("请对第2个文件做综合诊断", files)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["file"]["name"], "beta.csv")

    def test_resolve_file_reference_requires_explicit_choice_when_multiple_files_exist(self):
        files = [
            {"id": "1", "name": "alpha.csv", "path": r"D:\\data\\alpha.csv", "uploaded_at": 1},
            {"id": "2", "name": "beta.csv", "path": r"D:\\data\\beta.csv", "uploaded_at": 2},
        ]
        result = ui.resolve_file_for_message("帮我分析一下", files)
        self.assertEqual(result["status"], "ambiguous")
        self.assertIn("第1个文件", result["message"])
        self.assertIn("第2个文件", result["message"])


if __name__ == "__main__":
    unittest.main()
