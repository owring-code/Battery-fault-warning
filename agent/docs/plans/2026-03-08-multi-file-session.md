# Multi-File Session Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add session-level multi-file upload support with explicit file selection and ambiguity handling.

**Architecture:** Extend the existing `chat_sessions.json` schema so each session stores `history` plus `uploaded_files`. Implement pure helper functions in `ui.py` for normalization, file registration, file deletion, and user-message resolution. Wire the sidebar and upload flow to the new session file registry while preserving current chat behavior.

**Tech Stack:** Python 3.10, Gradio 6.x, standard library `unittest`, existing `ui.py` app structure.

---

### Task 1: Add regression tests for session file helpers

**Files:**
- Create: `tests/test_ui_multi_file.py`
- Modify: `ui.py`

**Step 1: Write the failing test**

```python
def test_register_uploaded_file_creates_session_and_file_entry(self):
    sessions, session_id, file_meta = ui.register_uploaded_file({}, "", r"D:\agent\tmp_upload\a.csv")
    self.assertEqual(file_meta["name"], "a.csv")
```

**Step 2: Run test to verify it fails**

Run: `d:\agent\.venv\Scripts\python.exe -m unittest tests.test_ui_multi_file -v`
Expected: FAIL because helpers do not exist yet.

**Step 3: Write minimal implementation**

Add helper functions in `ui.py`:
- `ensure_session_record`
- `build_uploaded_file_meta`
- `register_uploaded_file`
- `remove_uploaded_file`
- `resolve_file_reference`

**Step 4: Run test to verify it passes**

Run: `d:\agent\.venv\Scripts\python.exe -m unittest tests.test_ui_multi_file -v`
Expected: PASS

**Step 5: Commit**

Git repo is unavailable in this workspace, so no commit step can be executed here.

### Task 2: Wire session storage and data-routing to file resolution

**Files:**
- Modify: `ui.py`

**Step 1: Write the failing test**

```python
def test_resolve_file_reference_requires_explicit_choice_when_multiple_files_exist(self):
    result = ui.resolve_file_for_message("帮我分析一下", files)
    self.assertEqual(result["status"], "ambiguous")
```

**Step 2: Run test to verify it fails**

Run: `d:\agent\.venv\Scripts\python.exe -m unittest tests.test_ui_multi_file -v`
Expected: FAIL because ambiguity handling is not implemented.

**Step 3: Write minimal implementation**

Update data-path extraction and routing so data-dependent branches call the new resolver before analysis.

**Step 4: Run test to verify it passes**

Run: `d:\agent\.venv\Scripts\python.exe -m unittest tests.test_ui_multi_file -v`
Expected: PASS

**Step 5: Commit**

Git repo is unavailable in this workspace, so no commit step can be executed here.

### Task 3: Update Gradio sidebar and upload behavior

**Files:**
- Modify: `ui.py`

**Step 1: Write the failing test**

No separate UI automation test; verify through helper coverage and manual run.

**Step 2: Run targeted verification before implementation**

Run: `d:\agent\.venv\Scripts\python.exe -m py_compile ui.py`
Expected: PASS before wiring edits.

**Step 3: Write minimal implementation**

Add:
- sidebar file list
- per-file delete action
- upload handler that registers files to the current session
- session load that refreshes file list

**Step 4: Run verification**

Run:
- `d:\agent\.venv\Scripts\python.exe -m unittest tests.test_ui_multi_file -v`
- `d:\agent\.venv\Scripts\python.exe -m py_compile ui.py`

Expected: PASS

**Step 5: Commit**

Git repo is unavailable in this workspace, so no commit step can be executed here.
