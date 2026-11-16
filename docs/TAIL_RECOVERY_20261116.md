# CE6190 ICS: audited tail recovery, 16 November 2026

The original recovery run has 1,365 successful formal shards, five failed formal shards, nine uncertain submissions, and fourteen formal shards waiting for one LVIS smoke check. This change adds a separate recovery entry point; it does not edit any frozen worker or controller source.

## What it resumes

- Nine submissions whose exact current job names have no accounting or execution evidence: eight INSID3 formal shards and the LVIS no-aggregation smoke.
- One interrupted INSID3 formal shard, preserving its 155 successful episodes and resuming only its 95 missing episodes.
- The fourteen existing no-aggregation formal shards become eligible through the existing manager after the missing smoke passes.

The two incorrectly associated job IDs, 201529 and 201530, belong to successful fold-9 tasks. Their correct tasks and result records are preserved. New submissions use new attempt numbers and immutable intents; old intents and claims are retained.

## Four scientific failures stay visible

The four CG-ICS errors are `Reference mask is empty at the CG-ICS working resolution`. They occur during input preparation before SAM/MLLM inference. These are deterministic failures under the frozen evaluation pipeline. Their four FAILED shard records and error episode JSON remain unchanged, alongside 1,489 successful records in those shards. This recovery does not resize, replace, discard, or fabricate those cases.

The existing collector can run once all dependencies are terminal, including FAILED. If every operational gap succeeds, the expected recovery queue outcome is **COMPLETE_WITH_FAILURES**, with 1,451 PASS and four FAILED tasks (formal: 1,388 PASS and four FAILED). A collector PASS is not a declaration that all model episodes succeeded.

## Guarded operation

`scripts/resume_ics_tail.py audit --run-dir RUN` performs read-only checks and prints the concrete selection. `resume` performs the same checks, invokes the existing verified manager pause, repeats the checks while holding the manager and queue locks, applies an atomic recovery transaction, and invokes the existing activation command. The manager retains its 60-task submission window and existing partition/node policies.

The tool pins this exact plan, worker/controller snapshot identities, repair policy, and scheduling policy. It checks fresh full-user accounting and an empty queue, successful task markers, the two real job-ID owners, sealed intents, absent execution evidence for uncertain tasks, and selected episode/prediction hashes. The frozen runner also recomputes the complete manifest/config/code/data binding and refuses to resume into an existing output directory if it differs. Unexpected evidence refuses recovery. It does not blindly clear UNCERTAIN or overwrite successful rows.

Only the ten selected tasks' old job metadata, rolling markers, and batch scripts move into `RUN/tail-resume-20261116/archive`. Result directories, predictions, immutable intents, and claims stay in place. All old logs are retained; any logs inside a selected job metadata directory move with that directory into the archive, while Slurm logs remain at their original paths. Before/after state and evidence are stored in a sealed transaction. Complete preparation publishes atomically; interrupted application rolls forward. Repeating an applied operation never resets tasks that have since been submitted or completed.

Run with the existing INSID3 Python environment, after manually committing/pushing these added repository files and pulling on Orion. No new environment or resource download is needed. Local tests use synthetic data and scheduler fixtures; they are not a substitute for the server's model results.
