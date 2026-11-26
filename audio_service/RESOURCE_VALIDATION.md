# Audio worker resource validation

The Compose worker memory limit (`AUDIO_WORKER_MEMORY_LIMIT`, default `1g`)
and CPU quota (`AUDIO_WORKER_CPU_QUOTA`, default `1.0`) are provisional settings.
They have not been validated against production workloads.

The 512 MiB limit was insufficient for the official YAMNet runtime with a
120-second input: an ARM64 Linux container reached its cgroup memory maximum and
the inference process was OOM-killed. A 5-second inference with the maximum
20 MiB upload held in both worker processes peaked near 494 MiB, leaving too
little headroom. The 1 GiB limit is a measured safety adjustment, not a claim
that production peak memory has been characterized. Keep CPU and concurrency
limits unchanged and remeasure with representative maximum-size inputs.

On a Linux host with Docker Compose v2:

1. Set a non-empty `AUDIO_SERVICE_TOKEN`.
2. Configure `AUDIO_SERVICE_TOKEN` in your local `.env` file, then start the complete platform. Audio API and worker are included in the default Compose stack:

   ```sh
   docker compose up --build -d
   ```

3. Inspect the worker's effective container configuration:

   ```sh
   docker inspect "$(docker compose ps -q audio-worker)" --format 'Memory={{.HostConfig.Memory}} NanoCpus={{.HostConfig.NanoCpus}}'
   ```

4. Inspect the cgroup limits from inside the worker (the first branch is cgroup
   v2; the fallback reports the common cgroup v1 files):

   ```sh
   docker compose exec audio-worker sh -c 'if [ -r /sys/fs/cgroup/memory.max ]; then printf "memory.max="; cat /sys/fs/cgroup/memory.max; printf "cpu.max="; cat /sys/fs/cgroup/cpu.max; else printf "memory.limit_in_bytes="; cat /sys/fs/cgroup/memory/memory.limit_in_bytes; printf "cpu.cfs_quota_us="; cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us; fi'
   ```

The memory value should be finite and the CPU quota should reflect the configured
limit. Record peak memory and decode time using the supported maximum-size test
inputs before raising or lowering either limit.

The child process also applies `RLIMIT_CPU` on Linux using
`AUDIO_CHILD_CPU_LIMIT_SECONDS`. `RLIMIT_AS` is an optional Linux defense-in-depth
setting (`AUDIO_CHILD_ADDRESS_SPACE_LIMIT_BYTES`) and is disabled by default
because it limits virtual address space and needs runtime-specific tuning. These
limits are not substitutes for the container cgroup; the worker's cgroup memory
limit is the hard container boundary. The child does not apply these limits on
macOS. Running the worker in Docker Desktop exercises Linux cgroups inside its
VM, but the VM's overall resource allocation remains a separate constraint.

The Audio API and worker are part of the default Compose stack; no profile flag
is required. The worker is attached to the default network and the private
`audio-internal` network, and has no published host port.

The supported Compose deployment is one replica of `audio-api` and one replica
of `audio-worker`, each running one Uvicorn worker. `DecodeCapacity` is an
in-memory per-process limit, not a distributed/global limit. Scaling either
service or increasing its Uvicorn worker count multiplies the effective
concurrency; multi-replica operation is not supported by this configuration.
