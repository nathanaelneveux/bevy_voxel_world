use std::{
    env,
    hint::black_box,
    time::{Duration, Instant},
};

use bevy::{
    asset::AssetPlugin,
    camera::{CameraPlugin, CameraProjection, RenderTargetInfo, Viewport},
    mesh::MeshPlugin,
    prelude::*,
    time::TimeUpdateStrategy,
    transform::TransformPlugin,
};
use bevy_voxel_world::{
    benchmark::{generate_chunk_for_bench, GenerationPattern},
    custom_meshing::CHUNK_SIZE_I,
    prelude::*,
};
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};

const FRAME_TIME: Duration = Duration::from_nanos(16_666_667);
const BENCH_VIEWPORT_SIZE: UVec2 = UVec2::new(1920, 1080);

#[derive(Clone, Copy)]
struct BenchScenario {
    name: &'static str,
    frames: u32,
    warmup_frames: u32,
    world: WorldShape,
    camera_path: CameraPath,
    writes: WriteLoad,
    spawn_strategy: ChunkSpawnStrategy,
    despawn_strategy: ChunkDespawnStrategy,
    spawning_distance: u32,
    min_despawn_distance: u32,
    spawning_rays: usize,
    max_spawn_per_frame: usize,
    max_active_chunk_threads: usize,
    max_chunk_despawns_per_frame: usize,
    retire_chunks_interval: Duration,
    chunk_lod_update_interval: Duration,
    lod_profile: LodProfile,
    generation_work: u32,
    attach_chunks_to_root: bool,
}

#[derive(Clone, Copy, Default)]
enum WorldShape {
    Empty,
    #[default]
    AsteroidField,
    DenseOccluding,
    FlatTerrain,
    SingleVoxelPerChunk,
}

#[derive(Clone, Copy, Default)]
enum CameraPath {
    #[default]
    FastLinear,
    HighDistanceCruise,
    Static,
    LodOscillation,
    DespawnJump,
}

#[derive(Clone, Copy, Default)]
enum LodProfile {
    #[default]
    Off,
    Tight,
    Relaxed128,
}

#[derive(Clone, Copy, Default)]
enum WriteLoad {
    #[default]
    None,
    SameValue {
        writes_per_frame: i32,
    },
    MovingEdits {
        writes_per_frame: i32,
    },
}

#[derive(Resource, Clone)]
struct BenchWorld {
    scenario: BenchScenario,
}

#[derive(Resource, Default)]
struct BenchControl {
    frame: u32,
}

#[derive(Resource, Default)]
struct BenchStats {
    frames: u64,
    spawned: u64,
    despawned: u64,
    remeshed: u64,
    lod_changed: u64,
    voxel_updated: u64,
    voxel_writes_issued: u64,
    max_spawned_per_frame: u64,
    max_despawned_per_frame: u64,
    max_remeshed_per_frame: u64,
    max_lod_changed_per_frame: u64,
    max_voxel_updated_per_frame: u64,
    update_times: Vec<Duration>,
    total_update_time: Duration,
    max_update_time: Duration,
    frames_over_budget: u64,
    spawn_cap_hit_frames: u64,
    despawn_cap_hit_frames: u64,
    total_active_chunks: u64,
    initial_active_chunks: u64,
    final_active_chunks: u64,
    min_active_chunks: u64,
    max_active_chunks: u64,
    max_retiring_chunks: u64,
    total_spawn_chunks_us: u64,
    total_spawn_collect_candidates_us: u64,
    total_spawn_process_queue_us: u64,
    total_update_lods_us: u64,
    total_retire_chunks_us: u64,
    total_despawn_chunks_us: u64,
    total_remesh_dirty_chunks_us: u64,
    total_spawn_meshes_us: u64,
    total_flush_voxel_writes_us: u64,
    total_flush_chunk_map_buffers_us: u64,
    total_flush_mesh_cache_buffers_us: u64,
    max_spawn_chunks_us: u64,
    max_spawn_collect_candidates_us: u64,
    max_spawn_process_queue_us: u64,
    max_update_lods_us: u64,
    max_retire_chunks_us: u64,
    max_despawn_chunks_us: u64,
    max_remesh_dirty_chunks_us: u64,
    max_spawn_meshes_us: u64,
    max_flush_voxel_writes_us: u64,
    max_flush_chunk_map_buffers_us: u64,
    max_flush_mesh_cache_buffers_us: u64,
    spawn_ray_steps: u64,
    spawn_candidates: u64,
    spawn_unique_candidates: u64,
    spawn_distance_culled: u64,
    spawn_frustum_checks: u64,
    spawn_frustum_culled: u64,
    spawn_existing_chunks: u64,
    spawn_admitted: u64,
    spawn_cap_hit_diag_frames: u64,
    spawn_low_priority_promoted: u64,
    lod_chunks_scanned: u64,
    lod_high_priority: u64,
    lod_low_priority: u64,
    lod_threads_canceled: u64,
    retire_chunks_scanned: u64,
    retire_marked: u64,
    retire_frustum_checks: u64,
    retire_frustum_culled: u64,
    retire_distance_culled: u64,
    despawn_retired_scanned: u64,
    despawned_diag: u64,
    despawn_cap_hit_diag_frames: u64,
    remesh_pending_high_max: u64,
    remesh_pending_low_max: u64,
    remesh_active_threads_max: u64,
    remesh_started: u64,
    remesh_cap_hit_frames: u64,
    chunk_threads_polled: u64,
    chunk_threads_completed: u64,
    chunk_map_updates_queued: u64,
    chunk_map_inserts_flushed: u64,
    chunk_map_updates_flushed: u64,
    chunk_map_removes_flushed: u64,
}

impl Default for BenchWorld {
    fn default() -> Self {
        Self {
            scenario: base_fast_camera(),
        }
    }
}

impl VoxelWorldConfig for BenchWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn spawning_distance(&self) -> u32 {
        self.scenario.spawning_distance
    }

    fn min_despawn_distance(&self) -> u32 {
        self.scenario.min_despawn_distance
    }

    fn spawning_rays(&self) -> usize {
        self.scenario.spawning_rays
    }

    fn max_spawn_per_frame(&self) -> usize {
        self.scenario.max_spawn_per_frame
    }

    fn max_active_chunk_threads(&self) -> usize {
        self.scenario.max_active_chunk_threads
    }

    fn max_chunk_despawns_per_frame(&self) -> usize {
        self.scenario.max_chunk_despawns_per_frame
    }

    fn retire_chunks_interval(&self) -> Duration {
        self.scenario.retire_chunks_interval
    }

    fn chunk_lod_update_interval(&self) -> Duration {
        self.scenario.chunk_lod_update_interval
    }

    fn diagnostics_enabled(&self) -> bool {
        true
    }

    fn attach_chunks_to_root(&self) -> bool {
        self.scenario.attach_chunks_to_root
    }

    fn chunk_spawn_strategy(&self) -> ChunkSpawnStrategy {
        self.scenario.spawn_strategy
    }

    fn chunk_despawn_strategy(&self) -> ChunkDespawnStrategy {
        self.scenario.despawn_strategy
    }

    fn chunk_lod(
        &self,
        chunk_position: IVec3,
        previous_lod: Option<LodLevel>,
        camera_position: Vec3,
    ) -> LodLevel {
        let chunk_center = chunk_position.as_vec3() * CHUNK_SIZE_I as f32
            + Vec3::splat(CHUNK_SIZE_I as f32 * 0.5);
        let distance = chunk_center.distance(camera_position);
        let target = match self.scenario.lod_profile {
            LodProfile::Off => return 0,
            LodProfile::Tight => {
                if distance < 96.0 {
                    0
                } else if distance < 192.0 {
                    1
                } else {
                    2
                }
            }
            LodProfile::Relaxed128 => {
                if distance < 256.0 {
                    0
                } else if distance < 1_024.0 {
                    1
                } else {
                    2
                }
            }
        };

        if let Some(previous_lod) = previous_lod {
            let previous_lod = previous_lod.min(2);
            if (previous_lod as i8 - target as i8).unsigned_abs() == 1 {
                return target;
            }
        }

        target
    }

    fn chunk_data_shape(&self, lod_level: LodLevel) -> UVec3 {
        match lod_level {
            0 => padded_chunk_shape_uniform(32),
            1 => padded_chunk_shape_uniform(16),
            _ => padded_chunk_shape_uniform(8),
        }
    }

    fn chunk_meshing_shape(&self, lod_level: LodLevel) -> UVec3 {
        self.chunk_data_shape(lod_level)
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<Self::MaterialIndex> {
        let shape = self.scenario.world;
        let generation_work = self.scenario.generation_work;

        Box::new(move |chunk_pos, _, _| {
            Box::new(move |world_pos, _| {
                if generation_work > 0 {
                    burn_cpu(chunk_pos, world_pos, generation_work);
                }

                match shape {
                    WorldShape::Empty => WorldVoxel::Air,
                    WorldShape::DenseOccluding => WorldVoxel::Solid(1),
                    WorldShape::FlatTerrain => {
                        if world_pos.y <= 0 {
                            WorldVoxel::Solid(1)
                        } else {
                            WorldVoxel::Air
                        }
                    }
                    WorldShape::AsteroidField => asteroid_voxel(chunk_pos, world_pos),
                    WorldShape::SingleVoxelPerChunk => {
                        single_voxel_chunk(chunk_pos, world_pos)
                    }
                }
            })
        })
    }
}

fn streaming_benches(c: &mut Criterion) {
    if env::var_os("BVW_BENCH_REPORT").is_some() {
        print_reports("streaming", scenarios());
        print_reports("knob_sweeps", knob_scenarios());
        if env::var_os("BVW_BENCH_REPORT_ONLY").is_some() {
            return;
        }
    }

    let mut group = c.benchmark_group("streaming");
    group.sample_size(10);

    for scenario in scenarios() {
        group.bench_function(scenario.name, |b| {
            b.iter_batched(
                || build_app(scenario),
                |mut app| {
                    run_frames(&mut app, scenario.frames);
                    let stats = app.world().resource::<BenchStats>();
                    black_box(stats.summary_tuple());
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();

    let mut knob_group = c.benchmark_group("knob_sweeps");
    knob_group.sample_size(10);

    for scenario in knob_scenarios() {
        knob_group.bench_function(scenario.name, |b| {
            b.iter_batched(
                || build_app(scenario),
                |mut app| {
                    run_frames(&mut app, scenario.frames);
                    let stats = app.world().resource::<BenchStats>();
                    black_box(stats.summary_tuple());
                },
                BatchSize::SmallInput,
            );
        });
    }

    knob_group.finish();

    let mut generation_group = c.benchmark_group("generation");
    generation_group.sample_size(10);

    for (name, shape, pattern) in generation_scenarios() {
        generation_group.bench_function(name, |b| {
            b.iter(|| {
                let result = generate_chunk_for_bench(shape, pattern);
                black_box((
                    result.is_empty,
                    result.is_full,
                    result.voxels_len,
                    result.voxels_hash,
                ));
            });
        });
    }

    generation_group.finish();
}

fn build_app(scenario: BenchScenario) -> App {
    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        TransformPlugin,
        AssetPlugin::default(),
        MeshPlugin,
        CameraPlugin,
        VoxelWorldPlugin::headless_with_config(BenchWorld { scenario }),
    ));
    app.insert_resource(TimeUpdateStrategy::ManualDuration(FRAME_TIME));
    app.insert_resource(BenchControl::default());
    app.insert_resource(BenchStats::default());
    app.add_systems(Startup, spawn_camera);
    app.add_systems(First, (drive_camera, issue_voxel_writes));
    app.add_systems(Last, collect_stats);

    run_frames(&mut app, scenario.warmup_frames);
    app.world_mut().resource_mut::<BenchStats>().reset();
    let initial_active_chunks = count_active_chunks(&mut app);
    let mut stats = app.world_mut().resource_mut::<BenchStats>();
    stats.initial_active_chunks = initial_active_chunks;
    stats.min_active_chunks = initial_active_chunks;
    stats.final_active_chunks = initial_active_chunks;
    app
}

fn count_active_chunks(app: &mut App) -> u64 {
    app.world_mut()
        .query_filtered::<(), (With<Chunk<BenchWorld>>, Without<NeedsDespawn>)>()
        .iter(app.world())
        .count() as u64
}

fn run_frames(app: &mut App, frames: u32) {
    for _ in 0..frames {
        let start = Instant::now();
        app.update();
        let update_time = start.elapsed();
        let world = app.world_mut();
        world
            .resource_mut::<BenchStats>()
            .record_update_time(update_time);
        world.resource_mut::<BenchControl>().frame += 1;
    }
}

fn spawn_camera(mut commands: Commands) {
    let projection = PerspectiveProjection {
        aspect_ratio: BENCH_VIEWPORT_SIZE.x as f32 / BENCH_VIEWPORT_SIZE.y as f32,
        far: 999_999.0,
        ..default()
    };
    let mut camera = Camera {
        viewport: Some(Viewport {
            physical_size: BENCH_VIEWPORT_SIZE,
            ..default()
        }),
        ..default()
    };
    camera.computed.target_info = Some(RenderTargetInfo {
        physical_size: BENCH_VIEWPORT_SIZE,
        scale_factor: 1.0,
    });
    camera.computed.clip_from_view = projection.get_clip_from_view();

    commands.spawn((
        Camera3d::default(),
        camera,
        Projection::Perspective(projection),
        Transform::from_xyz(0.0, 32.0, -64.0)
            .looking_at(Vec3::new(0.0, 16.0, 256.0), Vec3::Y),
        VoxelWorldCamera::<BenchWorld>::default(),
    ));
}

fn drive_camera(
    control: Res<BenchControl>,
    world: Res<BenchWorld>,
    mut camera: Query<&mut Transform, With<VoxelWorldCamera<BenchWorld>>>,
) {
    let Ok(mut transform) = camera.single_mut() else {
        return;
    };

    let frame = control.frame as f32;
    let position = match world.scenario.camera_path {
        CameraPath::FastLinear => Vec3::new(frame * 18.0, 48.0, -96.0 + frame * 30.0),
        CameraPath::HighDistanceCruise => {
            Vec3::new(frame * 24.0, 96.0, -256.0 + frame * 48.0)
        }
        CameraPath::Static => Vec3::new(0.0, 72.0, -96.0),
        CameraPath::LodOscillation => {
            let z = 128.0 + (frame * 0.42).sin() * 144.0;
            Vec3::new(0.0, 64.0, z)
        }
        CameraPath::DespawnJump => {
            let jump = ((control.frame / 12) % 2) as f32;
            Vec3::new(jump * 2_400.0, 64.0, -128.0 + frame * 4.0)
        }
    };

    *transform = Transform::from_translation(position)
        .looking_at(position + Vec3::new(0.0, -0.1, 512.0), Vec3::Y);
}

fn issue_voxel_writes(
    control: Res<BenchControl>,
    world: Res<BenchWorld>,
    mut stats: ResMut<BenchStats>,
    mut voxel_world: VoxelWorld<BenchWorld>,
) {
    match world.scenario.writes {
        WriteLoad::None => {}
        WriteLoad::SameValue { writes_per_frame } => {
            for i in 0..writes_per_frame {
                let x = i % 32;
                let z = i / 32;
                voxel_world.set_voxel(IVec3::new(x, 1, z), WorldVoxel::Solid(7));
            }
            stats.voxel_writes_issued += writes_per_frame as u64;
        }
        WriteLoad::MovingEdits { writes_per_frame } => {
            let frame = control.frame as i32;
            for i in 0..writes_per_frame {
                let x = frame * 3 + i;
                let z = frame * 2 + (i % 17);
                voxel_world.set_voxel(IVec3::new(x, 1, z), WorldVoxel::Solid(3));
            }
            stats.voxel_writes_issued += writes_per_frame as u64;
        }
    }
}

fn collect_stats(
    world: Res<BenchWorld>,
    diagnostics: Res<VoxelWorldDiagnostics<BenchWorld>>,
    mut stats: ResMut<BenchStats>,
    active_chunks: Query<(), (With<Chunk<BenchWorld>>, Without<NeedsDespawn>)>,
    retiring_chunks: Query<(), (With<Chunk<BenchWorld>>, With<NeedsDespawn>)>,
    mut spawned: MessageReader<ChunkWillSpawn<BenchWorld>>,
    mut despawned: MessageReader<ChunkWillDespawn<BenchWorld>>,
    mut remeshed: MessageReader<ChunkWillRemesh<BenchWorld>>,
    mut lod_changed: MessageReader<ChunkWillChangeLod<BenchWorld>>,
    mut voxel_updated: MessageReader<ChunkWillUpdate<BenchWorld>>,
) {
    let spawned = spawned.read().count() as u64;
    let despawned = despawned.read().count() as u64;
    let remeshed = remeshed.read().count() as u64;
    let lod_changed = lod_changed.read().count() as u64;
    let voxel_updated = voxel_updated.read().count() as u64;
    let active_chunks = active_chunks.iter().count() as u64;
    let retiring_chunks = retiring_chunks.iter().count() as u64;

    stats.frames += 1;
    if stats.frames == 1 {
        stats.min_active_chunks = active_chunks;
    }
    stats.spawned += spawned;
    stats.despawned += despawned;
    stats.remeshed += remeshed;
    stats.lod_changed += lod_changed;
    stats.voxel_updated += voxel_updated;
    stats.total_active_chunks += active_chunks;
    stats.final_active_chunks = active_chunks;
    stats.min_active_chunks = stats.min_active_chunks.min(active_chunks);
    stats.max_active_chunks = stats.max_active_chunks.max(active_chunks);
    stats.max_retiring_chunks = stats.max_retiring_chunks.max(retiring_chunks);
    stats.max_spawned_per_frame = stats.max_spawned_per_frame.max(spawned);
    stats.max_despawned_per_frame = stats.max_despawned_per_frame.max(despawned);
    stats.max_remeshed_per_frame = stats.max_remeshed_per_frame.max(remeshed);
    stats.max_lod_changed_per_frame = stats.max_lod_changed_per_frame.max(lod_changed);
    stats.max_voxel_updated_per_frame =
        stats.max_voxel_updated_per_frame.max(voxel_updated);

    if world.scenario.max_spawn_per_frame != usize::MAX
        && spawned >= world.scenario.max_spawn_per_frame as u64
    {
        stats.spawn_cap_hit_frames += 1;
    }

    if world.scenario.max_chunk_despawns_per_frame != usize::MAX
        && despawned >= world.scenario.max_chunk_despawns_per_frame as u64
    {
        stats.despawn_cap_hit_frames += 1;
    }

    let diagnostics = diagnostics.frame;
    stats.total_spawn_chunks_us += diagnostics.spawn_chunks_us;
    stats.total_spawn_collect_candidates_us += diagnostics.spawn_collect_candidates_us;
    stats.total_spawn_process_queue_us += diagnostics.spawn_process_queue_us;
    stats.total_update_lods_us += diagnostics.update_lods_us;
    stats.total_retire_chunks_us += diagnostics.retire_chunks_us;
    stats.total_despawn_chunks_us += diagnostics.despawn_chunks_us;
    stats.total_remesh_dirty_chunks_us += diagnostics.remesh_dirty_chunks_us;
    stats.total_spawn_meshes_us += diagnostics.spawn_meshes_us;
    stats.total_flush_voxel_writes_us += diagnostics.flush_voxel_writes_us;
    stats.total_flush_chunk_map_buffers_us += diagnostics.flush_chunk_map_buffers_us;
    stats.total_flush_mesh_cache_buffers_us += diagnostics.flush_mesh_cache_buffers_us;
    stats.max_spawn_chunks_us =
        stats.max_spawn_chunks_us.max(diagnostics.spawn_chunks_us);
    stats.max_spawn_collect_candidates_us = stats
        .max_spawn_collect_candidates_us
        .max(diagnostics.spawn_collect_candidates_us);
    stats.max_spawn_process_queue_us = stats
        .max_spawn_process_queue_us
        .max(diagnostics.spawn_process_queue_us);
    stats.max_update_lods_us = stats.max_update_lods_us.max(diagnostics.update_lods_us);
    stats.max_retire_chunks_us =
        stats.max_retire_chunks_us.max(diagnostics.retire_chunks_us);
    stats.max_despawn_chunks_us = stats
        .max_despawn_chunks_us
        .max(diagnostics.despawn_chunks_us);
    stats.max_remesh_dirty_chunks_us = stats
        .max_remesh_dirty_chunks_us
        .max(diagnostics.remesh_dirty_chunks_us);
    stats.max_spawn_meshes_us =
        stats.max_spawn_meshes_us.max(diagnostics.spawn_meshes_us);
    stats.max_flush_voxel_writes_us = stats
        .max_flush_voxel_writes_us
        .max(diagnostics.flush_voxel_writes_us);
    stats.max_flush_chunk_map_buffers_us = stats
        .max_flush_chunk_map_buffers_us
        .max(diagnostics.flush_chunk_map_buffers_us);
    stats.max_flush_mesh_cache_buffers_us = stats
        .max_flush_mesh_cache_buffers_us
        .max(diagnostics.flush_mesh_cache_buffers_us);
    stats.spawn_ray_steps += diagnostics.spawn_ray_steps;
    stats.spawn_candidates += diagnostics.spawn_candidates;
    stats.spawn_unique_candidates += diagnostics.spawn_unique_candidates;
    stats.spawn_distance_culled += diagnostics.spawn_distance_culled;
    stats.spawn_frustum_checks += diagnostics.spawn_frustum_checks;
    stats.spawn_frustum_culled += diagnostics.spawn_frustum_culled;
    stats.spawn_existing_chunks += diagnostics.spawn_existing_chunks;
    stats.spawn_admitted += diagnostics.spawn_admitted;
    stats.spawn_low_priority_promoted += diagnostics.spawn_low_priority_promoted;
    stats.lod_chunks_scanned += diagnostics.lod_chunks_scanned;
    stats.lod_high_priority += diagnostics.lod_high_priority;
    stats.lod_low_priority += diagnostics.lod_low_priority;
    stats.lod_threads_canceled += diagnostics.lod_threads_canceled;
    stats.retire_chunks_scanned += diagnostics.retire_chunks_scanned;
    stats.retire_marked += diagnostics.retire_marked;
    stats.retire_frustum_checks += diagnostics.retire_frustum_checks;
    stats.retire_frustum_culled += diagnostics.retire_frustum_culled;
    stats.retire_distance_culled += diagnostics.retire_distance_culled;
    stats.despawn_retired_scanned += diagnostics.despawn_retired_scanned;
    stats.despawned_diag += diagnostics.despawned;
    stats.remesh_pending_high_max = stats
        .remesh_pending_high_max
        .max(diagnostics.remesh_pending_high);
    stats.remesh_pending_low_max = stats
        .remesh_pending_low_max
        .max(diagnostics.remesh_pending_low);
    stats.remesh_active_threads_max = stats
        .remesh_active_threads_max
        .max(diagnostics.remesh_active_threads);
    stats.remesh_started += diagnostics.remesh_started;
    stats.chunk_threads_polled += diagnostics.chunk_threads_polled;
    stats.chunk_threads_completed += diagnostics.chunk_threads_completed;
    stats.chunk_map_updates_queued += diagnostics.chunk_map_updates_queued;
    stats.chunk_map_inserts_flushed += diagnostics.chunk_map_inserts_flushed;
    stats.chunk_map_updates_flushed += diagnostics.chunk_map_updates_flushed;
    stats.chunk_map_removes_flushed += diagnostics.chunk_map_removes_flushed;

    if diagnostics.spawn_cap_hit {
        stats.spawn_cap_hit_diag_frames += 1;
    }
    if diagnostics.despawn_cap_hit {
        stats.despawn_cap_hit_diag_frames += 1;
    }
    if diagnostics.remesh_cap_hit {
        stats.remesh_cap_hit_frames += 1;
    }
}

impl BenchStats {
    fn reset(&mut self) {
        *self = Self::default();
    }

    fn record_update_time(&mut self, update_time: Duration) {
        self.update_times.push(update_time);
        self.total_update_time += update_time;
        self.max_update_time = self.max_update_time.max(update_time);
        if update_time > FRAME_TIME {
            self.frames_over_budget += 1;
        }
    }

    fn avg_update_micros(&self) -> u128 {
        if self.update_times.is_empty() {
            return 0;
        }

        self.total_update_time.as_micros() / self.update_times.len() as u128
    }

    fn p95_update_micros(&self) -> u128 {
        if self.update_times.is_empty() {
            return 0;
        }

        let mut update_times = self.update_times.clone();
        update_times.sort_unstable();
        let index = (update_times.len() * 95 / 100).min(update_times.len() - 1);
        update_times[index].as_micros()
    }

    fn avg_active_chunks(&self) -> u64 {
        if self.frames == 0 {
            return 0;
        }

        self.total_active_chunks / self.frames
    }

    fn delta_active_chunks(&self) -> i64 {
        self.final_active_chunks as i64 - self.initial_active_chunks as i64
    }

    fn avg_per_frame(&self, value: u64) -> u64 {
        if self.frames == 0 {
            return 0;
        }

        value / self.frames
    }

    fn summary_tuple(&self) -> (u64, u64, u64, u64, u64, u64, u64) {
        (
            self.frames,
            self.spawned,
            self.despawned,
            self.remeshed,
            self.lod_changed,
            self.voxel_updated,
            self.voxel_writes_issued,
        )
    }
}

fn asteroid_voxel(chunk_pos: IVec3, world_pos: IVec3) -> WorldVoxel<u8> {
    let chunk_hash = hash_ivec3(chunk_pos);
    if chunk_hash % 5 > 1 {
        return WorldVoxel::Air;
    }

    let origin = chunk_pos * CHUNK_SIZE_I;
    let local = world_pos - origin;
    let center = IVec3::new(
        8 + ((chunk_hash >> 3) & 15) as i32,
        8 + ((chunk_hash >> 8) & 15) as i32,
        8 + ((chunk_hash >> 13) & 15) as i32,
    );
    let radius = 7 + (chunk_hash & 7) as i32;

    if local.distance_squared(center) <= radius * radius {
        WorldVoxel::Solid(1)
    } else {
        WorldVoxel::Air
    }
}

fn single_voxel_chunk(chunk_pos: IVec3, world_pos: IVec3) -> WorldVoxel<u8> {
    let origin = chunk_pos * CHUNK_SIZE_I;
    let local = world_pos - origin;

    if local == IVec3::splat(CHUNK_SIZE_I / 2) {
        WorldVoxel::Solid(1)
    } else {
        WorldVoxel::Air
    }
}

fn burn_cpu(chunk_pos: IVec3, world_pos: IVec3, iterations: u32) {
    let mut x = hash_ivec3(chunk_pos) ^ hash_ivec3(world_pos);
    for _ in 0..iterations {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
    }
    black_box(x);
}

fn hash_ivec3(pos: IVec3) -> u32 {
    let mut x = pos.x as u32;
    x = x.wrapping_mul(0x8da6_b343);
    x ^= (pos.y as u32).wrapping_mul(0xd816_3841);
    x ^= (pos.z as u32).wrapping_mul(0xcb1a_b31f);
    x ^ (x >> 16)
}

fn print_reports(group: &str, scenarios: Vec<BenchScenario>) {
    let filter = env::var("BVW_BENCH_REPORT_FILTER").ok();
    eprintln!("\n{group}");
    eprintln!(
        "{}",
        [
            "scenario",
            "frames",
            "avg_update_us",
            "p95_update_us",
            "max_update_us",
            "frames_over_budget",
            "spawn_cap_hit_frames",
            "despawn_cap_hit_frames",
            "initial_active_chunks",
            "avg_active_chunks",
            "final_active_chunks",
            "delta_active_chunks",
            "min_active_chunks",
            "max_active_chunks",
            "max_retiring_chunks",
            "spawned",
            "despawned",
            "remeshed",
            "lod_changed",
            "voxel_updated",
            "writes_issued",
            "max_spawned",
            "max_despawned",
            "max_remeshed",
            "max_lod_changed",
            "max_voxel_updated",
            "avg_spawn_system_us",
            "max_spawn_system_us",
            "avg_spawn_collect_candidates_us",
            "max_spawn_collect_candidates_us",
            "avg_spawn_process_queue_us",
            "max_spawn_process_queue_us",
            "avg_lod_system_us",
            "max_lod_system_us",
            "avg_retire_system_us",
            "max_retire_system_us",
            "avg_despawn_system_us",
            "max_despawn_system_us",
            "avg_remesh_admit_system_us",
            "max_remesh_admit_system_us",
            "avg_spawn_meshes_system_us",
            "max_spawn_meshes_system_us",
            "avg_flush_voxel_writes_us",
            "max_flush_voxel_writes_us",
            "avg_flush_chunk_map_buffers_us",
            "max_flush_chunk_map_buffers_us",
            "avg_flush_mesh_cache_buffers_us",
            "max_flush_mesh_cache_buffers_us",
            "spawn_ray_steps",
            "spawn_candidates",
            "spawn_unique_candidates",
            "spawn_distance_culled",
            "spawn_frustum_checks",
            "spawn_frustum_culled",
            "spawn_existing_chunks",
            "spawn_admitted",
            "spawn_cap_hit_diag_frames",
            "spawn_low_priority_promoted",
            "lod_chunks_scanned",
            "lod_high_priority",
            "lod_low_priority",
            "lod_threads_canceled",
            "retire_chunks_scanned",
            "retire_marked",
            "retire_frustum_checks",
            "retire_frustum_culled",
            "retire_distance_culled",
            "despawn_retired_scanned",
            "despawned_diag",
            "despawn_cap_hit_diag_frames",
            "remesh_pending_high_max",
            "remesh_pending_low_max",
            "remesh_active_threads_max",
            "remesh_started",
            "remesh_cap_hit_frames",
            "chunk_threads_polled",
            "chunk_threads_completed",
            "chunk_map_updates_queued",
            "chunk_map_inserts_flushed",
            "chunk_map_updates_flushed",
            "chunk_map_removes_flushed",
        ]
        .join(",")
    );

    for scenario in scenarios {
        if filter
            .as_deref()
            .is_some_and(|filter| !scenario.name.contains(filter))
        {
            continue;
        }

        let mut app = build_app(scenario);
        run_frames(&mut app, scenario.frames);
        let stats = app.world().resource::<BenchStats>();
        let row = vec![
            scenario.name.to_string(),
            stats.frames.to_string(),
            stats.avg_update_micros().to_string(),
            stats.p95_update_micros().to_string(),
            stats.max_update_time.as_micros().to_string(),
            stats.frames_over_budget.to_string(),
            stats.spawn_cap_hit_frames.to_string(),
            stats.despawn_cap_hit_frames.to_string(),
            stats.initial_active_chunks.to_string(),
            stats.avg_active_chunks().to_string(),
            stats.final_active_chunks.to_string(),
            stats.delta_active_chunks().to_string(),
            stats.min_active_chunks.to_string(),
            stats.max_active_chunks.to_string(),
            stats.max_retiring_chunks.to_string(),
            stats.spawned.to_string(),
            stats.despawned.to_string(),
            stats.remeshed.to_string(),
            stats.lod_changed.to_string(),
            stats.voxel_updated.to_string(),
            stats.voxel_writes_issued.to_string(),
            stats.max_spawned_per_frame.to_string(),
            stats.max_despawned_per_frame.to_string(),
            stats.max_remeshed_per_frame.to_string(),
            stats.max_lod_changed_per_frame.to_string(),
            stats.max_voxel_updated_per_frame.to_string(),
            stats.avg_per_frame(stats.total_spawn_chunks_us).to_string(),
            stats.max_spawn_chunks_us.to_string(),
            stats
                .avg_per_frame(stats.total_spawn_collect_candidates_us)
                .to_string(),
            stats.max_spawn_collect_candidates_us.to_string(),
            stats
                .avg_per_frame(stats.total_spawn_process_queue_us)
                .to_string(),
            stats.max_spawn_process_queue_us.to_string(),
            stats.avg_per_frame(stats.total_update_lods_us).to_string(),
            stats.max_update_lods_us.to_string(),
            stats
                .avg_per_frame(stats.total_retire_chunks_us)
                .to_string(),
            stats.max_retire_chunks_us.to_string(),
            stats
                .avg_per_frame(stats.total_despawn_chunks_us)
                .to_string(),
            stats.max_despawn_chunks_us.to_string(),
            stats
                .avg_per_frame(stats.total_remesh_dirty_chunks_us)
                .to_string(),
            stats.max_remesh_dirty_chunks_us.to_string(),
            stats.avg_per_frame(stats.total_spawn_meshes_us).to_string(),
            stats.max_spawn_meshes_us.to_string(),
            stats
                .avg_per_frame(stats.total_flush_voxel_writes_us)
                .to_string(),
            stats.max_flush_voxel_writes_us.to_string(),
            stats
                .avg_per_frame(stats.total_flush_chunk_map_buffers_us)
                .to_string(),
            stats.max_flush_chunk_map_buffers_us.to_string(),
            stats
                .avg_per_frame(stats.total_flush_mesh_cache_buffers_us)
                .to_string(),
            stats.max_flush_mesh_cache_buffers_us.to_string(),
            stats.spawn_ray_steps.to_string(),
            stats.spawn_candidates.to_string(),
            stats.spawn_unique_candidates.to_string(),
            stats.spawn_distance_culled.to_string(),
            stats.spawn_frustum_checks.to_string(),
            stats.spawn_frustum_culled.to_string(),
            stats.spawn_existing_chunks.to_string(),
            stats.spawn_admitted.to_string(),
            stats.spawn_cap_hit_diag_frames.to_string(),
            stats.spawn_low_priority_promoted.to_string(),
            stats.lod_chunks_scanned.to_string(),
            stats.lod_high_priority.to_string(),
            stats.lod_low_priority.to_string(),
            stats.lod_threads_canceled.to_string(),
            stats.retire_chunks_scanned.to_string(),
            stats.retire_marked.to_string(),
            stats.retire_frustum_checks.to_string(),
            stats.retire_frustum_culled.to_string(),
            stats.retire_distance_culled.to_string(),
            stats.despawn_retired_scanned.to_string(),
            stats.despawned_diag.to_string(),
            stats.despawn_cap_hit_diag_frames.to_string(),
            stats.remesh_pending_high_max.to_string(),
            stats.remesh_pending_low_max.to_string(),
            stats.remesh_active_threads_max.to_string(),
            stats.remesh_started.to_string(),
            stats.remesh_cap_hit_frames.to_string(),
            stats.chunk_threads_polled.to_string(),
            stats.chunk_threads_completed.to_string(),
            stats.chunk_map_updates_queued.to_string(),
            stats.chunk_map_inserts_flushed.to_string(),
            stats.chunk_map_updates_flushed.to_string(),
            stats.chunk_map_removes_flushed.to_string(),
        ];
        eprintln!("{}", row.join(","));
    }
}

fn base_fast_camera() -> BenchScenario {
    BenchScenario {
        name: "fast_camera_asteroid_field",
        frames: 96,
        warmup_frames: 12,
        world: WorldShape::AsteroidField,
        camera_path: CameraPath::FastLinear,
        writes: WriteLoad::None,
        spawn_strategy: ChunkSpawnStrategy::CloseAndInView,
        despawn_strategy: ChunkDespawnStrategy::FarAwayOrOutOfView,
        spawning_distance: 48,
        min_despawn_distance: 2,
        spawning_rays: 128,
        max_spawn_per_frame: 512,
        max_active_chunk_threads: 64,
        max_chunk_despawns_per_frame: usize::MAX,
        retire_chunks_interval: Duration::ZERO,
        chunk_lod_update_interval: Duration::ZERO,
        lod_profile: LodProfile::Off,
        generation_work: 0,
        attach_chunks_to_root: false,
    }
}

fn distance_128_relaxed_lod() -> BenchScenario {
    BenchScenario {
        name: "distance_128_relaxed_lod_view_despawn",
        frames: 180,
        warmup_frames: 24,
        world: WorldShape::AsteroidField,
        camera_path: CameraPath::HighDistanceCruise,
        writes: WriteLoad::None,
        spawn_strategy: ChunkSpawnStrategy::CloseAndInView,
        despawn_strategy: ChunkDespawnStrategy::FarAwayOrOutOfView,
        spawning_distance: 128,
        min_despawn_distance: 4,
        spawning_rays: 384,
        max_spawn_per_frame: 2_048,
        max_active_chunk_threads: 128,
        max_chunk_despawns_per_frame: usize::MAX,
        retire_chunks_interval: Duration::ZERO,
        chunk_lod_update_interval: Duration::ZERO,
        lod_profile: LodProfile::Relaxed128,
        generation_work: 0,
        attach_chunks_to_root: false,
    }
}

fn distance_128_single_voxel() -> BenchScenario {
    BenchScenario {
        name: "distance_128_single_voxel_delta_load",
        frames: 45,
        warmup_frames: 15,
        world: WorldShape::SingleVoxelPerChunk,
        camera_path: CameraPath::HighDistanceCruise,
        writes: WriteLoad::None,
        spawn_strategy: ChunkSpawnStrategy::CloseAndInView,
        despawn_strategy: ChunkDespawnStrategy::FarAwayOrOutOfView,
        spawning_distance: 128,
        min_despawn_distance: 4,
        spawning_rays: 1_024,
        max_spawn_per_frame: 8_192,
        max_active_chunk_threads: 8_192,
        max_chunk_despawns_per_frame: usize::MAX,
        retire_chunks_interval: Duration::ZERO,
        chunk_lod_update_interval: Duration::ZERO,
        lod_profile: LodProfile::Relaxed128,
        generation_work: 0,
        attach_chunks_to_root: false,
    }
}

fn scenarios() -> Vec<BenchScenario> {
    vec![
        base_fast_camera(),
        distance_128_relaxed_lod(),
        BenchScenario {
            name: "distance_128_relaxed_lod_far_away",
            despawn_strategy: ChunkDespawnStrategy::FarAway,
            ..distance_128_relaxed_lod()
        },
        BenchScenario {
            name: "distance_128_expensive_relaxed_lod_view_despawn",
            generation_work: 256,
            ..distance_128_relaxed_lod()
        },
        BenchScenario {
            name: "distance_128_expensive_relaxed_lod_far_away",
            despawn_strategy: ChunkDespawnStrategy::FarAway,
            generation_work: 256,
            ..distance_128_relaxed_lod()
        },
        BenchScenario {
            name: "distance_128_single_voxel_initial_load",
            frames: 30,
            warmup_frames: 0,
            camera_path: CameraPath::Static,
            spawning_rays: 2_048,
            max_spawn_per_frame: 32_768,
            max_active_chunk_threads: 32_768,
            lod_profile: LodProfile::Off,
            ..distance_128_single_voxel()
        },
        distance_128_single_voxel(),
        BenchScenario {
            name: "long_draw_distance_static_camera",
            frames: 96,
            warmup_frames: 0,
            world: WorldShape::AsteroidField,
            camera_path: CameraPath::Static,
            writes: WriteLoad::None,
            spawn_strategy: ChunkSpawnStrategy::CloseAndInView,
            despawn_strategy: ChunkDespawnStrategy::FarAwayOrOutOfView,
            spawning_distance: 96,
            min_despawn_distance: 3,
            spawning_rays: 256,
            max_spawn_per_frame: 2_048,
            max_active_chunk_threads: 128,
            max_chunk_despawns_per_frame: usize::MAX,
            retire_chunks_interval: Duration::ZERO,
            chunk_lod_update_interval: Duration::ZERO,
            lod_profile: LodProfile::Off,
            generation_work: 0,
            attach_chunks_to_root: false,
        },
        BenchScenario {
            name: "dense_occluding_terrain",
            frames: 96,
            warmup_frames: 8,
            world: WorldShape::DenseOccluding,
            camera_path: CameraPath::FastLinear,
            writes: WriteLoad::None,
            spawn_strategy: ChunkSpawnStrategy::CloseAndInView,
            despawn_strategy: ChunkDespawnStrategy::FarAwayOrOutOfView,
            spawning_distance: 64,
            min_despawn_distance: 2,
            spawning_rays: 256,
            max_spawn_per_frame: 512,
            max_active_chunk_threads: 64,
            max_chunk_despawns_per_frame: usize::MAX,
            retire_chunks_interval: Duration::ZERO,
            chunk_lod_update_interval: Duration::ZERO,
            lod_profile: LodProfile::Off,
            generation_work: 0,
            attach_chunks_to_root: false,
        },
        BenchScenario {
            name: "frequent_same_value_writes",
            frames: 96,
            warmup_frames: 16,
            world: WorldShape::Empty,
            camera_path: CameraPath::Static,
            writes: WriteLoad::SameValue {
                writes_per_frame: 512,
            },
            spawn_strategy: ChunkSpawnStrategy::CloseAndInView,
            despawn_strategy: ChunkDespawnStrategy::FarAwayOrOutOfView,
            spawning_distance: 32,
            min_despawn_distance: 2,
            spawning_rays: 96,
            max_spawn_per_frame: 256,
            max_active_chunk_threads: 64,
            max_chunk_despawns_per_frame: usize::MAX,
            retire_chunks_interval: Duration::ZERO,
            chunk_lod_update_interval: Duration::ZERO,
            lod_profile: LodProfile::Off,
            generation_work: 0,
            attach_chunks_to_root: false,
        },
        BenchScenario {
            name: "lod_churn_thresholds",
            frames: 128,
            warmup_frames: 16,
            world: WorldShape::FlatTerrain,
            camera_path: CameraPath::LodOscillation,
            writes: WriteLoad::None,
            spawn_strategy: ChunkSpawnStrategy::CloseAndInView,
            despawn_strategy: ChunkDespawnStrategy::FarAwayOrOutOfView,
            spawning_distance: 56,
            min_despawn_distance: 2,
            spawning_rays: 128,
            max_spawn_per_frame: 512,
            max_active_chunk_threads: 64,
            max_chunk_despawns_per_frame: usize::MAX,
            retire_chunks_interval: Duration::ZERO,
            chunk_lod_update_interval: Duration::ZERO,
            lod_profile: LodProfile::Tight,
            generation_work: 0,
            attach_chunks_to_root: false,
        },
        BenchScenario {
            name: "despawn_pressure_jump",
            frames: 96,
            warmup_frames: 24,
            world: WorldShape::AsteroidField,
            camera_path: CameraPath::DespawnJump,
            writes: WriteLoad::None,
            spawn_strategy: ChunkSpawnStrategy::CloseAndInView,
            despawn_strategy: ChunkDespawnStrategy::FarAwayOrOutOfView,
            spawning_distance: 56,
            min_despawn_distance: 2,
            spawning_rays: 128,
            max_spawn_per_frame: 512,
            max_active_chunk_threads: 64,
            max_chunk_despawns_per_frame: 256,
            retire_chunks_interval: Duration::ZERO,
            chunk_lod_update_interval: Duration::ZERO,
            lod_profile: LodProfile::Off,
            generation_work: 0,
            attach_chunks_to_root: false,
        },
        BenchScenario {
            name: "lock_contention_generation_and_writes",
            frames: 96,
            warmup_frames: 16,
            world: WorldShape::AsteroidField,
            camera_path: CameraPath::FastLinear,
            writes: WriteLoad::MovingEdits {
                writes_per_frame: 256,
            },
            spawn_strategy: ChunkSpawnStrategy::CloseAndInView,
            despawn_strategy: ChunkDespawnStrategy::FarAwayOrOutOfView,
            spawning_distance: 48,
            min_despawn_distance: 2,
            spawning_rays: 128,
            max_spawn_per_frame: 512,
            max_active_chunk_threads: 128,
            max_chunk_despawns_per_frame: usize::MAX,
            retire_chunks_interval: Duration::ZERO,
            chunk_lod_update_interval: Duration::ZERO,
            lod_profile: LodProfile::Off,
            generation_work: 16,
            attach_chunks_to_root: false,
        },
    ]
}

fn knob_scenarios() -> Vec<BenchScenario> {
    let fast = base_fast_camera();
    let despawn = BenchScenario {
        name: "despawn_pressure_jump",
        frames: 96,
        warmup_frames: 24,
        world: WorldShape::AsteroidField,
        camera_path: CameraPath::DespawnJump,
        writes: WriteLoad::None,
        spawn_strategy: ChunkSpawnStrategy::CloseAndInView,
        despawn_strategy: ChunkDespawnStrategy::FarAwayOrOutOfView,
        spawning_distance: 56,
        min_despawn_distance: 2,
        spawning_rays: 128,
        max_spawn_per_frame: 512,
        max_active_chunk_threads: 64,
        max_chunk_despawns_per_frame: usize::MAX,
        retire_chunks_interval: Duration::ZERO,
        chunk_lod_update_interval: Duration::ZERO,
        lod_profile: LodProfile::Off,
        generation_work: 0,
        attach_chunks_to_root: false,
    };
    let lod = BenchScenario {
        name: "lod_churn_thresholds",
        frames: 128,
        warmup_frames: 16,
        world: WorldShape::FlatTerrain,
        camera_path: CameraPath::LodOscillation,
        writes: WriteLoad::None,
        spawn_strategy: ChunkSpawnStrategy::CloseAndInView,
        despawn_strategy: ChunkDespawnStrategy::FarAwayOrOutOfView,
        spawning_distance: 56,
        min_despawn_distance: 2,
        spawning_rays: 128,
        max_spawn_per_frame: 512,
        max_active_chunk_threads: 64,
        max_chunk_despawns_per_frame: usize::MAX,
        retire_chunks_interval: Duration::ZERO,
        chunk_lod_update_interval: Duration::ZERO,
        lod_profile: LodProfile::Tight,
        generation_work: 0,
        attach_chunks_to_root: false,
    };
    let contention = BenchScenario {
        name: "lock_contention_generation_and_writes",
        frames: 96,
        warmup_frames: 16,
        world: WorldShape::AsteroidField,
        camera_path: CameraPath::FastLinear,
        writes: WriteLoad::MovingEdits {
            writes_per_frame: 256,
        },
        spawn_strategy: ChunkSpawnStrategy::CloseAndInView,
        despawn_strategy: ChunkDespawnStrategy::FarAwayOrOutOfView,
        spawning_distance: 48,
        min_despawn_distance: 2,
        spawning_rays: 128,
        max_spawn_per_frame: 512,
        max_active_chunk_threads: 128,
        max_chunk_despawns_per_frame: usize::MAX,
        retire_chunks_interval: Duration::ZERO,
        chunk_lod_update_interval: Duration::ZERO,
        lod_profile: LodProfile::Off,
        generation_work: 16,
        attach_chunks_to_root: false,
    };

    vec![
        BenchScenario {
            name: "spawn_cap_64",
            max_spawn_per_frame: 64,
            ..fast
        },
        BenchScenario {
            name: "spawn_cap_128",
            max_spawn_per_frame: 128,
            ..fast
        },
        BenchScenario {
            name: "spawn_cap_512",
            max_spawn_per_frame: 512,
            ..fast
        },
        BenchScenario {
            name: "chunk_threads_16",
            max_active_chunk_threads: 16,
            ..contention
        },
        BenchScenario {
            name: "chunk_threads_64",
            max_active_chunk_threads: 64,
            ..contention
        },
        BenchScenario {
            name: "chunk_threads_128",
            max_active_chunk_threads: 128,
            ..contention
        },
        BenchScenario {
            name: "despawn_cap_64",
            max_chunk_despawns_per_frame: 64,
            ..despawn
        },
        BenchScenario {
            name: "despawn_cap_256",
            max_chunk_despawns_per_frame: 256,
            ..despawn
        },
        BenchScenario {
            name: "despawn_cap_unlimited",
            max_chunk_despawns_per_frame: usize::MAX,
            ..despawn
        },
        BenchScenario {
            name: "lod_interval_0ms",
            chunk_lod_update_interval: Duration::ZERO,
            ..lod
        },
        BenchScenario {
            name: "lod_interval_53ms",
            chunk_lod_update_interval: Duration::from_millis(53),
            ..lod
        },
        BenchScenario {
            name: "retire_interval_0ms",
            retire_chunks_interval: Duration::ZERO,
            ..despawn
        },
        BenchScenario {
            name: "retire_interval_47ms",
            retire_chunks_interval: Duration::from_millis(47),
            ..despawn
        },
    ]
}

fn generation_scenarios() -> Vec<(&'static str, UVec3, GenerationPattern)> {
    vec![
        (
            "lod0_empty",
            padded_chunk_shape_uniform(32),
            GenerationPattern::Empty,
        ),
        (
            "lod0_full",
            padded_chunk_shape_uniform(32),
            GenerationPattern::Full,
        ),
        (
            "lod0_single_voxel",
            padded_chunk_shape_uniform(32),
            GenerationPattern::SingleVoxel,
        ),
        (
            "lod1_empty",
            padded_chunk_shape_uniform(16),
            GenerationPattern::Empty,
        ),
        (
            "lod1_full",
            padded_chunk_shape_uniform(16),
            GenerationPattern::Full,
        ),
        (
            "lod1_single_voxel",
            padded_chunk_shape_uniform(16),
            GenerationPattern::SingleVoxel,
        ),
        (
            "lod2_empty",
            padded_chunk_shape_uniform(8),
            GenerationPattern::Empty,
        ),
        (
            "lod2_full",
            padded_chunk_shape_uniform(8),
            GenerationPattern::Full,
        ),
        (
            "lod2_single_voxel",
            padded_chunk_shape_uniform(8),
            GenerationPattern::SingleVoxel,
        ),
    ]
}

criterion_group!(streaming, streaming_benches);
criterion_main!(streaming);
