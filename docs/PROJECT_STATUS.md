# Repository packaging status — 17 November 2026

- Scientific model implementations and experiment configurations remain frozen.
  The cleanup changes documentation, archival locations, and compact analysis
  deliverables. Existing run snapshots remain separate from this checkout.
- B1 and B2 passed output-level acceptance at the packaging snapshot; their
  compact summaries are included. Subsequent B4 acceptance is handled separately.
  This packaging snapshot does not certify final experiment completion, final
  report metrics, or the status of any controller or job.
- The non-report updater preserves the entire local `report/` directory. It
  neither reads nor writes that directory and carries no report payload. Final
  report source and PDF must be synchronized separately after report review.
- The course source ZIP and final report PDF are separate submission files.
  The repository update bundle itself is not the course source submission.
- Changed non-report paths retain before/after SHA256 checks. A local conflict
  stops the update before repository writes. Replaced or removed files are
  backed up outside the repository; repeating a completed update is a no-op.
- Review the diff and commit/push manually. This packaging performs no Git
  commit/push, Orion deployment, or scheduler action.

Earlier operational and research records remain under `docs/history/` for
provenance. They are historical records, not current run instructions. This
repository coordination note is excluded from the course source ZIP.
