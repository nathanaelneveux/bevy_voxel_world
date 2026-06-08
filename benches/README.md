# Streaming Benchmark Goals

These benches exist to keep normal chunk lifecycle performance honest:
spawning, LOD refresh, remesh admission, mesh publishing, map flush,
retirement, and despawn. User terrain generation and custom meshing can be
expensive, so BVW's own overhead should stay small enough that users keep most
of the 16.67 ms frame budget at 60 FPS.

The main target is:

- no frames over `FRAME_TIME` in steady streaming scenarios;
- spawn admission keeps up with the camera when the scenario is meant to
  saturate `max_spawn_per_frame`;
- `chunks_deque`/candidate discovery finds enough useful work without growing
  into a large backlog;
- retire/despawn work does not create visible spikes when the camera moves fast
  or jumps;
- static and slow-moving cameras do not churn chunks that are already visible;
- LOD changes are bounded by priority/rate limiting so they do not starve
  nearby chunk generation.

## Running Reports

Criterion measures whole benchmark functions. For lifecycle tuning, the CSV
report is usually more useful:

```sh
BVW_BENCH_REPORT=1 BVW_BENCH_REPORT_ONLY=1 cargo bench --bench streaming -- --nocapture
```

Filter to one scenario or sweep while iterating:

```sh
BVW_BENCH_REPORT=1 BVW_BENCH_REPORT_ONLY=1 BVW_BENCH_REPORT_FILTER=ray_sweep_d250_32 cargo bench --bench streaming -- --nocapture
```

## Reading Metrics

Frame budget:

- `avg_update_us`, `p95_update_us`, `max_update_us`, `frames_over_budget`
  describe the total Bevy app update for each measured frame.
- For 60 FPS, `frames_over_budget` should be zero and `p95_update_us` should
  stay comfortably below `16_667`.

Spawn throughput:

- `spawned`, `max_spawned`, and `spawn_cap_hit_frames` show whether chunk
  creation is saturating the requested cap.
- `spawn_candidates`, `spawn_unique_candidates`, `spawn_existing_chunks`,
  `spawn_distance_culled`, and `spawn_frustum_culled` separate useful spawn
  work from discovery/filtering overhead.
- In ray sweeps, the best ray count is the lowest value that mostly saturates
  `max_spawn_per_frame` without excessive candidate queue or ray-step budget
  hits.

LOD and remesh:

- `lod_chunks_scanned` versus `lod_changed` estimates how much of LOD refresh
  is empty scanning.
- `lod_high_priority` and `lod_low_priority` should show nearby work staying
  ahead of far LOD churn.
- `remesh_started`, `remesh_cap_hit_frames`, and `remesh_active_threads_max`
  show whether task admission or active-thread limits are the bottleneck.

Retire/despawn:

- `retire_chunks_scanned` versus `retire_marked` estimates empty retire scans.
- `retire_frustum_checks`, `retire_frustum_culled`, and
  `retire_distance_culled` show whether out-of-view culling or distance culling
  is doing most of the work.
- `despawn_retired_scanned`, `despawned_diag`, and
  `despawn_cap_hit_diag_frames` show whether despawn caps are smoothing spikes
  or falling behind.

Publishing and buffers:

- `chunk_threads_polled` versus `chunk_threads_completed` measures async task
  polling overhead in `spawn_meshes`.
- `mesh_cache_hits`, `mesh_cache_misses`, and `mesh_cache_stores` show whether
  repeated voxel payloads avoid duplicate mesh assets.
- `chunk_map_inserts_flushed`, `chunk_map_updates_flushed`,
  `chunk_map_removes_flushed`, and `chunk_map_bounds_rebuilt` explain
  `flush_chunk_map_buffers` spikes.

## Scenario Intent

- `base_fast_camera` is the broad lifecycle smoke test for a fast camera at a
  moderate draw distance.
- `distance_128_*` and `ray_sweep_d128_*` target long draw distance with relaxed
  LOD. They should keep frame time under budget while showing which ray counts
  saturate delta-load spawning.
- `distance_250_*` and `ray_sweep_d250_*` are deliberately punishing. They are
  for finding the edge of far-terrain discovery and candidate queue behavior,
  not for expecting every configuration to hit 60 FPS.
- `*_single_voxel_*` minimizes user generation and meshing cost so BVW overhead
  is easier to see.
- `*_expensive_*` adds synthetic generation cost to check whether BVW rate
  limiting still leaves the app responsive when user work is heavier.
- `static_camera_long_view_churn_regression` and
  `slow_in_chunk_camera_long_view` protect against spawn/despawn/LOD churn when
  the camera is not crossing many chunk boundaries.
- `despawn_pressure_jump` and `despawn_cap_*` focus on teardown spikes and
  whether despawn caps smooth them without letting retired chunks accumulate.
- `lod_churn_thresholds` and `lod_interval_*` focus on LOD scan cost, hysteresis,
  and low-priority remesh behavior.
- `lock_contention_generation_and_writes` and `chunk_threads_*` exercise
  modified voxel writes, chunk-map locks, generation, and active-thread limits.
- `mesh_cache_*` isolates mesh cache value and lifecycle cost for repeated
  versus unique chunk payloads.
- `multicam_same_view_*` should stay close to the single-camera result because
  identical camera views are deduplicated before lifecycle systems run.
- `multicam_split_view_*` models independent cameras in different parts of the
  same voxel world. These should be cheaper than running separate voxel worlds,
  but they are expected to scale with the number of unique views because
  visibility, distance, and spawn-ray work must consider each view.
- `generation/*` isolates `Chunk::generate` overhead for empty, full, and sparse
  chunks at each LOD data shape.
