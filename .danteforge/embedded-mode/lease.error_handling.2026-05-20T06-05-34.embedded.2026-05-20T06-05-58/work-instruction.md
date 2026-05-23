# Work Instruction — `lease.error_handling.2026-05-20T06-05-34.embedded.2026-05-20T06-05-58`

**Packet:** Close gap on Error Handling & Recovery
**Worktree:** `C:\Projects\DanteDojo\.danteforge-worktrees\lease.error_handling.2026-05-20T06-05-34.embedded.2026-05-20T06-05-58`
**Host AI:** unknown
**Created:** 2026-05-20T06:05:58.958Z

## Objective

Move "Error Handling & Recovery" from current score 5 toward target 9.

## Scope

**Owned paths (you may edit these):**

**Read-only paths:**
_(none)_

**Forbidden paths (do NOT touch):**
- `.danteforge/compete/matrix.json`
- `.danteforge/compete/matrix-*.json`
- `.danteforge/compete/COMPETE_REPORT.md`
- `.danteforge/scores/**`
- `.danteforge/score-proposals/**`

## Acceptance Criteria

- Implementation passes typecheck and tests
- Score moves from 5 toward 7
- Score for error_handling moves from 5 to at least 9 (gap: 4.0)

## Instructions

You are the host AI executing lease `lease.error_handling.2026-05-20T06-05-34.embedded.2026-05-20T06-05-58` for work packet `work.error_handling.2026-05-20T06-05-34`.

Steps:
1. Read the full work-instruction.json next to this file for machine-readable scope.
2. Make the changes inline using your own Edit/Write tools, **inside the worktree at `C:\Projects\DanteDojo\.danteforge-worktrees\lease.error_handling.2026-05-20T06-05-34.embedded.2026-05-20T06-05-58`** — NOT in the main repo checkout. Every 
   ownedPath is relative to that worktree root. The worktree is on branch `matrix/error_handling/embedded-06-05-58`; edits there don't touch your main branch.
3. STAY within `ownedPaths`. Never touch `forbiddenPaths`.
4. Run any required local checks (typecheck/tests) **from the worktree directory** before reporting completion.
5. When done, run: `danteforge matrix-kernel embedded-complete lease.error_handling.2026-05-20T06-05-34.embedded.2026-05-20T06-05-58`
   That command captures your diff via `git diff --name-only HEAD` inside the worktree and feeds it into verify-court.

Do NOT spawn another Claude Code / Codex subprocess. The kernel chose embedded mode precisely to avoid that.
