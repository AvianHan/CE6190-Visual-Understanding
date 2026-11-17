# Batch-2 pending-job acceleration

This update addresses two observed constraints: the shared 60-job submission
window is full, and the original eligible GPU nodes are nearly fully allocated.
The 17 November 10:29 SGT audit identified a rebooted L40S node with eight
unallocated GPUs and BFS9B74 in testqueue with six unallocated H200 GPUs.
These are historical allocation observations, not a promise of immediate access.

## Operation

The accelerator is a separate background companion for the existing
`ics-batch2-20261116` run. The original manager continues to own submissions,
task states, model execution and collection. The companion never calls sbatch,
scancel or requeue, and never resets or retries a task.

It first selects up to four accepted, unstarted formal tasks: one INSID3 and
one CG-ICS task for each candidate node. Core tasks are preferred because their
observed shards are short. These pilot tasks were already required by the
experiment plan. They keep their job IDs, frozen commands, inputs and outputs;
no completed experiment is repeated and no 61st queue slot is requested.

Candidates:

| Route | Exact node | Family-specific evidence required |
|---|---|---|
| l40sp | A484652X3B10601 | Original model completion + claim + task marker + matching Slurm success |
| testqueue | BFS9B74 | The same evidence, with exact actual node and partition |

Known-configuration pilots use a one-hour INSID3 limit or a two-hour CG-ICS
limit, each with substantial margin over the recorded runtime; existing shorter
limits are not extended. Their success authorizes only
that node and resource family, bound to the observed boot identity. A failure
blocks that route/family and remains visible in the original run; it is never
silently retried or changed into a success. Passing INSID3 does not authorize
CG-ICS. Other testqueue nodes remain excluded. The unrelated 2FS9B74 CG-ICS
exclusion is preserved.

For accepted pending work, the companion verifies the sealed intent, worker
command, job ID/name, owner, account and QoS; writes an audit journal; places
its own temporary hold; rechecks that the job is still pending; updates only
partition/required-node/excluded-node/walltime; verifies unchanged CPU, memory,
GPU and task requests; and releases its hold. Existing user/admin holds are
not touched. A pending-to-running race causes a skip and release of only the
companion's newly applied hold. Failed changes are rolled back while held.
An interrupted transaction is recovered before further modifications, including
a released pilot whose state update was interrupted. Transient scheduler query
failures defer routing and retry on the next poll without permanently disabling
a proven node. Incomplete route evidence during initial preparation stops before
creating a policy, so the same activation can be retried cleanly.

Known INSID3 pending tasks can use a 30-minute limit and known CG-ICS tasks a
two-hour limit. The observed maxima were approximately 5.3 and 33.6 minutes,
respectively. Unknown configurations retain their limit. Running jobs are
never shortened. A model failure after migration blocks further expansion
on the corresponding candidate node/family. Hardware and source identities
are recorded separately from scientific configuration.

## Deploy

Apply the checksum-verified ZIP on the Mac existing repository. Review and
commit/push manually. The installer adds only the six listed files, checks
all destination paths before writing, and creates a backup receipt. It does
not modify unlisted files or Git state.

On Orion:

```bash
(
    set -e
    cd /scratch/prj0000000262_1/hansy/CE6190-Visual-Understanding
    git pull --ff-only
    export ICS_RUNTIME=/scratch/prj0000000262_1/hansy/ce6190-ics-runtime
    export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
    export PATH=/cm/shared/apps/slurm/current/bin:$PATH
    "$ICS_RUNTIME/envs/insid3/bin/python" -I -B -u scripts/accelerate_batch2.py activate \
        --run-dir "$ICS_RUNTIME/runs/ics-batch2-20261116"
)
```

`activate` freezes a separate operational source under the existing run and
starts a detached, lock-protected companion. Repeating it is idempotent.
It does not restart the original manager. The original manager must be active.
You can exit SSH after observing the accelerator's live heartbeat; initial
STARTED alone does not prove a pilot has started or passed.

Status:

```bash
ICS_RUN=/scratch/prj0000000262_1/hansy/ce6190-ics-runtime/runs/ics-batch2-20261116
ICS_PY=/scratch/prj0000000262_1/hansy/ce6190-ics-runtime/envs/insid3/bin/python
"$ICS_PY" -I -B -u "$ICS_RUN/acceleration-20261117/source/scripts/accelerate_batch2.py" \
    status --run-dir "$ICS_RUN"
```

Files owned by this companion are under `acceleration-20261117/`: immutable
policy/source, state, transaction journals, `controller.log`, `manager.json`
and the independent lock. The main rolling states and original frozen source
are preserved. `stop` requests a graceful companion exit and hold recovery;
it leaves the original manager and existing jobs running. Already released
routing changes remain valid.

Local mocked Slurm/process tests are recorded in
`research/validation_2026-11-17_acceleration.json`. They do not substitute for
the first real node-specific pilot. No Orion command was run by the assistant.
