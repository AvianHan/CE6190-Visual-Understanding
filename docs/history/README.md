# Historical records

These files preserve earlier research and operational decisions. Their dates and
status statements describe their own snapshots; they are not current instructions
to start, cancel, repair, or resubmit a run.

- `README_20261104.md`: the previous repository landing page.
- `PROJECT_LOG_before_cleanup.md`: the project log previously tracked at the root.
- `operations/`: dated deployment, recovery, and scheduling notes.
- `research/`: earlier audits, validation records, and source research.

Scientific and controller source paths remain unchanged. The root
`UPDATE_MANIFEST.json` is retained only as historical release metadata because old
update CLIs read that path. It does not describe this reorganized tree: do not use
those old CLIs to install the cleanup. Use the separate cleanup overlay installer.
The course source ZIP excludes this archive and the historical update CLIs.
