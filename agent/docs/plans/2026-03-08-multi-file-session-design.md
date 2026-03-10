# Multi-File Session Design

**Date:** 2026-03-08

**Goal:** Let a single chat session register multiple uploaded files and require the user to explicitly identify which file to use for analysis or diagnosis.

## Context

The current UI stores only chat history per session. Uploading a file inserts a generated message containing the uploaded file path, but there is no session-level file registry. This means repeated uploads are possible, yet file reuse across multiple turns depends on the model inferring the intended file from natural language history, which is unreliable once more than one file exists in the same session.

## Requirements

1. A session can hold multiple uploaded files.
2. There is no implicit active file.
3. The user must explicitly identify the file by path, file name, or ordinal reference such as "第2个文件".
4. If multiple files exist and the user does not clearly identify one, the system must ask for clarification instead of guessing.
5. The sidebar should show the current session's uploaded files and allow deletion of individual file records.
6. Existing history/session behavior should remain intact.

## Recommended Approach

Store an `uploaded_files` list alongside each session record in `chat_sessions.json`. Each entry keeps:

- `id`
- `name`
- `path`
- `uploaded_at`

Add helper functions in `ui.py` to normalize old session data, add/remove uploaded files, and resolve a referenced file from a user message. Resolution priority:

1. Explicit path in message
2. Exact file name in message
3. Ordinal reference such as "第N个文件"

If resolution fails and multiple files are available, return a clarification message listing the available files.

## Alternatives Considered

### 1. Use the most recently uploaded file automatically

Rejected because it is fast but unsafe in multi-file sessions.

### 2. Maintain a single "current file" selected in the sidebar

Rejected because it still introduces hidden state and does not match the user's preference for explicit selection in each request.

### 3. Require explicit selection in every data-dependent request

Chosen because it is predictable and prevents accidental analysis on the wrong file.

## Data Flow

1. User uploads a file.
2. Upload handler ensures a session exists, appends the file metadata to that session, and updates a sidebar file list.
3. The upload handler writes a neutral prompt into the input box showing how to reference the uploaded files.
4. When the user sends a message:
   - intent routing still decides between data analysis, diagnosis, and comprehensive diagnosis
   - any data-dependent branch first resolves the target file from the session file pool
   - if no explicit file is resolved and the session contains multiple files, the system asks the user to specify the file

## Error Handling

- Unsupported path references: return a clear message that the file was not found.
- Multiple files with ambiguous user wording: return a clarification prompt with file index and file name.
- Legacy session records without `uploaded_files`: normalize them to an empty list on load.

## Testing Strategy

Use `unittest` to cover pure helper logic:

1. session normalization adds `uploaded_files`
2. uploaded file metadata is appended correctly
3. file resolution works for file name and ordinal references
4. ambiguous multi-file cases return clarification instead of guessing

## Constraints

- No new runtime dependencies
- Keep `ui.py` as the integration point for now
- Preserve compatibility with existing session history records
