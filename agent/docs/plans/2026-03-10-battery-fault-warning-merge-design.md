# Battery Fault Warning Merge Design

**Date:** 2026-03-10

**Goal:** Merge the local `D:\agent` project into the `Battery-fault-warning` GitHub repository without disturbing the repository's original battery warning code layout.

## Context

The target repository currently presents a lightweight battery fault warning project with its own source directory and README. The local `D:\agent` workspace is a separate Python-based industrial diagnosis agent with a Gradio UI, tests, vector-store assets, and project docs. The requested outcome is to preserve the target repository's original structure while adding the agent as a distinct module inside the same repository.

## Approved Direction

Use a side-by-side repository layout:

1. Keep the upstream repository's current top-level files and folders unchanged.
2. Add a new top-level `agent/` directory containing the local agent project.
3. Update the root README so the repository clearly explains the two parts:
   - the original battery fault warning code
   - the newly merged agent module

## Alternatives Considered

### 1. Replace the original repository structure with the local agent project

Rejected because it would erase the identity of the original repository and make the merge look like a takeover rather than an extension.

### 2. Flatten both projects into one shared top-level source tree

Rejected because it would increase the chance of file collisions and make future maintenance harder.

### 3. Add the local project as an isolated `agent/` subdirectory

Chosen because it is the safest merge strategy, preserves provenance, and keeps documentation straightforward.

## Repository Shape

After the merge, the repository should look conceptually like this:

- existing upstream files and folders remain in place
- `agent/`
- `agent/run_agent.py`
- `agent/ui.py`
- `agent/build_rag_vectorstore.py`
- `agent/tests/`
- `agent/docs/`
- `agent/manuals/`
- `agent/faiss_industrial_index/`

## README Strategy

The root README should:

1. Describe the repository as a combined workspace.
2. Briefly explain the original battery fault warning code.
3. Introduce the new `agent/` module and its capabilities.
4. Show a clear directory overview.
5. Provide separate quick-start guidance for the original code and the new agent.

## Risks And Mitigations

- Large binary/vector assets may make the merged repository heavier.
  - Mitigation: copy only the currently required assets and keep the README explicit about their role.
- Runtime dependencies may differ between the upstream project and the agent.
  - Mitigation: document the `agent/` environment separately in the README.
- The local workspace is not currently a git repository.
  - Mitigation: clone the target repository into the writable workspace and perform the merge there.
