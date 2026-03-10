# Battery Fault Warning Merge Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Clone the `Battery-fault-warning` repository, merge the current local agent project into it under `agent/`, and update the root README to explain the combined repository layout.

**Architecture:** Keep the upstream repository intact and introduce the local project as a self-contained top-level `agent/` module. Treat the work mainly as repository restructuring and documentation, with a small verification step focused on file presence and the existing agent test file.

**Tech Stack:** Git, PowerShell, Python, Gradio, LangChain, unittest

---

### Task 1: Prepare merge documentation

**Files:**
- Create: `docs/plans/2026-03-10-battery-fault-warning-merge-design.md`
- Create: `docs/plans/2026-03-10-battery-fault-warning-merge.md`

**Step 1: Write the planning documents**

Document the approved side-by-side merge approach and the implementation sequence.

**Step 2: Verify the files exist**

Run: `Get-ChildItem docs/plans`
Expected: both new 2026-03-10 plan files appear in the listing

### Task 2: Clone the upstream repository into the workspace

**Files:**
- Create: `Battery-fault-warning/`

**Step 1: Clone the repository**

Run: `git clone https://github.com/owring-code/Battery-fault-warning.git`
Expected: a new `Battery-fault-warning` directory is created under `D:\agent`

**Step 2: Inspect the cloned layout**

Run: `rg --files D:\agent\Battery-fault-warning`
Expected: upstream files such as `README.md` and code files are visible

### Task 3: Write the failing merge test

**Files:**
- Modify: `Battery-fault-warning/tests/` if present, otherwise create a repository-level smoke test or use a file-structure assertion command

**Step 1: Add a failing structure test if the target repo has a Python test layout**

If practical, create a minimal test that asserts the merged repository exposes `agent/run_agent.py` and `agent/ui.py`.

**Step 2: Run the test to verify it fails before copying**

Run: project-appropriate test command
Expected: failure because `agent/` does not exist yet

**Step 3: If no test harness exists**

Fallback to a shell-level red/green check:
- RED command: `Test-Path D:\agent\Battery-fault-warning\agent\run_agent.py`
- Expected: `False`

### Task 4: Copy the local agent project into `agent/`

**Files:**
- Create: `Battery-fault-warning/agent/run_agent.py`
- Create: `Battery-fault-warning/agent/ui.py`
- Create: `Battery-fault-warning/agent/build_rag_vectorstore.py`
- Create: `Battery-fault-warning/agent/tests/test_ui_multi_file.py`
- Create: `Battery-fault-warning/agent/docs/...`
- Create: `Battery-fault-warning/agent/manuals/...`
- Create: `Battery-fault-warning/agent/faiss_industrial_index/...`
- Create: `Battery-fault-warning/agent/chat_sessions.json`

**Step 1: Copy the project into the new subdirectory**

Run a recursive copy that excludes transient caches if possible.

**Step 2: Re-run the structure check**

Run: `Test-Path D:\agent\Battery-fault-warning\agent\run_agent.py`
Expected: `True`

**Step 3: Remove transient cache files if any were copied**

Delete `__pycache__` folders and other obvious generated artifacts if they appear under `agent/`.

### Task 5: Update the root README

**Files:**
- Modify: `Battery-fault-warning/README.md`

**Step 1: Inspect the existing README**

Summarize the current sections and preserve the original project identity.

**Step 2: Rewrite the README for the merged repository**

Include:
- project overview
- original module summary
- new `agent/` module summary
- directory tree
- quick-start notes

**Step 3: Review the rendered Markdown mentally for clarity**

Check for broken structure, duplicated sections, and inconsistent terminology.

### Task 6: Verify the merge

**Files:**
- Test: `Battery-fault-warning/agent/tests/test_ui_multi_file.py`

**Step 1: Run the agent test file if dependencies are available**

Run: `python -m unittest agent.tests.test_ui_multi_file`
Expected: pass, or fail only due to missing third-party dependencies not installed in this environment

**Step 2: Run a repository file listing**

Run: `rg --files D:\agent\Battery-fault-warning`
Expected: upstream files plus the new `agent/` subtree

**Step 3: Check git status inside the cloned repository**

Run: `git -C D:\agent\Battery-fault-warning status --short`
Expected: new `agent/` files and `README.md` modifications are visible
