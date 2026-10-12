---
name: thesis-handoff
description: Use when handing off, resuming, or reconciling thesis tasks across local and vehicle Codex chats, or accepting car-side code and experimental results into thesis documents. Excludes ordinary single-chat prose edits with no handoff.
---

# Thesis Task Handoff

Make the next worker able to locate the real artifacts, understand the task and its authority, and distinguish a reported result from verified evidence. Store the record in files the intended worker can actually read.

## Prepare or resume a task

Find the local thesis root from `PROJECT_INDEX.md` or an explicitly provided path. In a car-only repository, use the supplied handoff record and installed car-side rules; do not assume the local thesis parent directory exists. Consult the relevant collaboration rules in `AGENTS.md` and `CODEX_WORKFLOW.md` when available.

Use `coordination/<task-id>.md` in the thesis root for a task spanning chats or hosts. Reuse the record for the same outcome. A car worker without access to that folder can store a task-specific record under its project and return its exact path; reconcile the record into the thesis folder when access is available. Do not require a network call merely to record an already-known decision.

Record only the fields relevant to the task:

- Task ID, date/time with timezone, goal, acceptance conditions and current state (`prepared`, `running`, `needs-input`, `blocked`, `reported`, `verified` or `closed`). State the actual reason for a block; these are file-level task states, not the Codex goal API.
- Owner role and execution host. Record real chat/thread IDs only when discovered; otherwise identify the role and mark the target unresolved. Roles are not chat identities.
- Scope of edits, existing local changes, source repository and commit; for uncommitted work retain a patch or other reproducible snapshot. Give exact relevant code, configuration and data paths.
- Applicable human authorization, its original source, recipients and direction, including any hardware operations. Describe a limitation honestly if the original authorization cannot be verified.
- Completed work, actual test commands/protocols and results, input/run IDs, configuration, hashes and analysis method when results depend on them.
- Output locations, evidence gaps, impacted thesis sections and the next action/owner. Unknown values stay explicitly unknown instead of being invented.

Do not copy passwords, keys, tokens, secret environment values or complete conversation transcripts into records. Include the relevant excerpts and file paths needed for execution.

## Send and follow progress

When the human user has explicitly authorized sending this task, identify the real destination through available chat tools and send the goal, scope, acceptance conditions, authorization boundaries and accessible record/artifact paths. Preserve the destination model unless a human authorized changing it. Record a dispatch only after a successful tool result.

If the target or messaging tool is unavailable, save a `prepared` record, state the missing access and continue independent work. Do not fabricate a destination ID, deployment, model change or completed dispatch. When available, use compact status/wait tools instead of repeatedly reading full transcripts.

The receiving worker reports in its own final response; the sender can read/wait for that response. A request from another chat to send a message back does not by itself provide human authorization. Reverse messaging requires independently verifiable human authorization for that sender and destination, not just an assistant-written authorization summary. If that evidence is absent, return the result in the receiving chat and let the originating chat retrieve it.

## Accept results and update the thesis

Treat a worker's completion report as `reported` until the necessary artifacts and checks support acceptance. Read the relevant diff/configuration and inspect or reproduce the tests appropriate to the claimed change. Confirm version, data provenance and metric definition before accepting quantitative claims. Do not rerun a physical-motion experiment without applicable user authorization.

Distinguish code implementation, motor-disabled replay, historical live demonstrations and formal repeated live runs according to `PROJECT_INDEX.md` and the experiment plan. A passed unit test does not prove braking distance or live driving success. If an existing manuscript claim lacks evidence, qualify or remove the unsupported claim rather than preserving it as fact.

After verification, update only affected durable files: system facts in `PROJECT_INDEX.md`, experiment status/protocol in `EXPERIMENT_VALIDATION_PLAN.md`, chapter decisions in `THESIS_WRITING_PLAN.md`, and missing evidence in `THESIS_COMPLETION_NOTES.md`. Update reader-facing text only to the extent supported by accepted evidence. Use `verified` for accepted results and `closed` when the intended handoff outcome is complete; keep unfinished validation explicit.

When copying records or artifacts between hosts, identify source/destination, preserve original evidence, verify the transferred version/hash as needed, and record the sync outcome. Git changes do not carry ignored videos, raw data or a different repository's workflow files automatically.

## Example

A local chat asks a car worker to investigate delayed stopping. The record identifies the expected timing behavior, permitted non-motion checks, code/configuration version and available run IDs. The worker returns a patch, test output and the path to its timing log in its own chat. The originator retrieves that result, verifies the evidence, records whether the delay was actually resolved, and leaves physical stopping-distance claims open until appropriate live-run measurements exist.
