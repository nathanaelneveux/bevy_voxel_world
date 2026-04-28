use std::{env, hint::black_box, time::Duration};

use bevy::{
    asset::AssetPlugin, camera::CameraPlugin, mesh::MeshPlugin, prelude::*,
    time::TimeUpdateStrategy, transform::TransformPlugin,
};
use bevy_voxel_world::{custom_meshing::CHUNK_SIZE_I, prelude::*};
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};

const FRAME_TIME: Duration = Duration::from_millis(16);

#[derive(Clone, Copy)]
struct BenchScenario {
    name: &'static str,
    frames: u32,
    warmup_frames: u32,
    world: WorldShape,
    camera_path: CameraPath,
    writes: WriteLoad,
    spawning_distance: u32,
    min_despawn_distance: u32,
    spawning_rays: usize,
    max_spawn_per_frame: usize,
    max_active_chunk_threads: usize,
    max_chunk_despawns_per_frame: usize,
    retire_chunks_interval: Duration,
    chunk_lod_update_interval: Duration,
    enable_lod: bool,
    expensive_generation: bool,
}

#[derive(Clone, Copy, Default)]
enum WorldShape {
    Empty,
    #[default]
    AsteroidField,
    DenseOccluding,
    FlatTerrain,
}

#[derive(Clone, Copy, Default)]
enum CameraPath {
    #[default]
    FastLinear,
    Static,
    LodOscillation,
    DespawnJump,
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

    fn chunk_spawn_strategy(&self) -> ChunkSpawnStrategy {
        ChunkSpawnStrategy::CloseAndInView
    }

    fn chunk_despawn_strategy(&self) -> ChunkDespawnStrategy {
        ChunkDespawnStrategy::FarAwayOrOutOfView
    }

    fn chunk_lod(
        &self,
        chunk_position: IVec3,
        previous_lod: Option<LodLevel>,
        camera_position: Vec3,
    ) -> LodLevel {
        if !self.scenario.enable_lod {
            return 0;
        }

        let chunk_center = chunk_position.as_vec3() * CHUNK_SIZE_I as f32
            + Vec3::splat(CHUNK_SIZE_I as f32 * 0.5);
        let distance = chunk_center.distance(camera_position);
        let target = if distance < 96.0 {
            0
        } else if distance < 192.0 {
            1
        } else {
            2
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
        let expensive_generation = self.scenario.expensive_generation;

        Box::new(move |chunk_pos, _, _| {
            Box::new(move |world_pos, _| {
                if expensive_generation {
                    burn_cpu(chunk_pos, world_pos);
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
    app
}

fn run_frames(app: &mut App, frames: u32) {
    for frame in 0..frames {
        app.world_mut().resource_mut::<BenchControl>().frame = frame;
        app.update();
    }
}

fn spawn_camera(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            far: 999_999.0,
            ..default()
        }),
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
    mut stats: ResMut<BenchStats>,
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

    stats.frames += 1;
    stats.spawned += spawned;
    stats.despawned += despawned;
    stats.remeshed += remeshed;
    stats.lod_changed += lod_changed;
    stats.voxel_updated += voxel_updated;
    stats.max_spawned_per_frame = stats.max_spawned_per_frame.max(spawned);
    stats.max_despawned_per_frame = stats.max_despawned_per_frame.max(despawned);
    stats.max_remeshed_per_frame = stats.max_remeshed_per_frame.max(remeshed);
    stats.max_lod_changed_per_frame = stats.max_lod_changed_per_frame.max(lod_changed);
    stats.max_voxel_updated_per_frame =
        stats.max_voxel_updated_per_frame.max(voxel_updated);
}

impl BenchStats {
    fn reset(&mut self) {
        *self = Self::default();
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

fn burn_cpu(chunk_pos: IVec3, world_pos: IVec3) {
    let mut x = hash_ivec3(chunk_pos) ^ hash_ivec3(world_pos);
    for _ in 0..16 {
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
    eprintln!("\n{group}");
    eprintln!(
        "scenario,frames,spawned,despawned,remeshed,lod_changed,voxel_updated,writes_issued,max_spawned,max_despawned,max_remeshed,max_lod_changed,max_voxel_updated"
    );

    for scenario in scenarios {
        let mut app = build_app(scenario);
        run_frames(&mut app, scenario.frames);
        let stats = app.world().resource::<BenchStats>();
        eprintln!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{}",
            scenario.name,
            stats.frames,
            stats.spawned,
            stats.despawned,
            stats.remeshed,
            stats.lod_changed,
            stats.voxel_updated,
            stats.voxel_writes_issued,
            stats.max_spawned_per_frame,
            stats.max_despawned_per_frame,
            stats.max_remeshed_per_frame,
            stats.max_lod_changed_per_frame,
            stats.max_voxel_updated_per_frame,
        );
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
        spawning_distance: 48,
        min_despawn_distance: 2,
        spawning_rays: 128,
        max_spawn_per_frame: 512,
        max_active_chunk_threads: 64,
        max_chunk_despawns_per_frame: usize::MAX,
        retire_chunks_interval: Duration::ZERO,
        chunk_lod_update_interval: Duration::ZERO,
        enable_lod: false,
        expensive_generation: false,
    }
}

fn scenarios() -> Vec<BenchScenario> {
    vec![
        base_fast_camera(),
        BenchScenario {
            name: "long_draw_distance_static_camera",
            frames: 96,
            warmup_frames: 0,
            world: WorldShape::AsteroidField,
            camera_path: CameraPath::Static,
            writes: WriteLoad::None,
            spawning_distance: 96,
            min_despawn_distance: 3,
            spawning_rays: 256,
            max_spawn_per_frame: 2_048,
            max_active_chunk_threads: 128,
            max_chunk_despawns_per_frame: usize::MAX,
            retire_chunks_interval: Duration::ZERO,
            chunk_lod_update_interval: Duration::ZERO,
            enable_lod: false,
            expensive_generation: false,
        },
        BenchScenario {
            name: "dense_occluding_terrain",
            frames: 96,
            warmup_frames: 8,
            world: WorldShape::DenseOccluding,
            camera_path: CameraPath::FastLinear,
            writes: WriteLoad::None,
            spawning_distance: 64,
            min_despawn_distance: 2,
            spawning_rays: 256,
            max_spawn_per_frame: 512,
            max_active_chunk_threads: 64,
            max_chunk_despawns_per_frame: usize::MAX,
            retire_chunks_interval: Duration::ZERO,
            chunk_lod_update_interval: Duration::ZERO,
            enable_lod: false,
            expensive_generation: false,
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
            spawning_distance: 32,
            min_despawn_distance: 2,
            spawning_rays: 96,
            max_spawn_per_frame: 256,
            max_active_chunk_threads: 64,
            max_chunk_despawns_per_frame: usize::MAX,
            retire_chunks_interval: Duration::ZERO,
            chunk_lod_update_interval: Duration::ZERO,
            enable_lod: false,
            expensive_generation: false,
        },
        BenchScenario {
            name: "lod_churn_thresholds",
            frames: 128,
            warmup_frames: 16,
            world: WorldShape::FlatTerrain,
            camera_path: CameraPath::LodOscillation,
            writes: WriteLoad::None,
            spawning_distance: 56,
            min_despawn_distance: 2,
            spawning_rays: 128,
            max_spawn_per_frame: 512,
            max_active_chunk_threads: 64,
            max_chunk_despawns_per_frame: usize::MAX,
            retire_chunks_interval: Duration::ZERO,
            chunk_lod_update_interval: Duration::ZERO,
            enable_lod: true,
            expensive_generation: false,
        },
        BenchScenario {
            name: "despawn_pressure_jump",
            frames: 96,
            warmup_frames: 24,
            world: WorldShape::AsteroidField,
            camera_path: CameraPath::DespawnJump,
            writes: WriteLoad::None,
            spawning_distance: 56,
            min_despawn_distance: 2,
            spawning_rays: 128,
            max_spawn_per_frame: 512,
            max_active_chunk_threads: 64,
            max_chunk_despawns_per_frame: 256,
            retire_chunks_interval: Duration::ZERO,
            chunk_lod_update_interval: Duration::ZERO,
            enable_lod: false,
            expensive_generation: false,
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
            spawning_distance: 48,
            min_despawn_distance: 2,
            spawning_rays: 128,
            max_spawn_per_frame: 512,
            max_active_chunk_threads: 128,
            max_chunk_despawns_per_frame: usize::MAX,
            retire_chunks_interval: Duration::ZERO,
            chunk_lod_update_interval: Duration::ZERO,
            enable_lod: false,
            expensive_generation: true,
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
        spawning_distance: 56,
        min_despawn_distance: 2,
        spawning_rays: 128,
        max_spawn_per_frame: 512,
        max_active_chunk_threads: 64,
        max_chunk_despawns_per_frame: usize::MAX,
        retire_chunks_interval: Duration::ZERO,
        chunk_lod_update_interval: Duration::ZERO,
        enable_lod: false,
        expensive_generation: false,
    };
    let lod = BenchScenario {
        name: "lod_churn_thresholds",
        frames: 128,
        warmup_frames: 16,
        world: WorldShape::FlatTerrain,
        camera_path: CameraPath::LodOscillation,
        writes: WriteLoad::None,
        spawning_distance: 56,
        min_despawn_distance: 2,
        spawning_rays: 128,
        max_spawn_per_frame: 512,
        max_active_chunk_threads: 64,
        max_chunk_despawns_per_frame: usize::MAX,
        retire_chunks_interval: Duration::ZERO,
        chunk_lod_update_interval: Duration::ZERO,
        enable_lod: true,
        expensive_generation: false,
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
        spawning_distance: 48,
        min_despawn_distance: 2,
        spawning_rays: 128,
        max_spawn_per_frame: 512,
        max_active_chunk_threads: 128,
        max_chunk_despawns_per_frame: usize::MAX,
        retire_chunks_interval: Duration::ZERO,
        chunk_lod_update_interval: Duration::ZERO,
        enable_lod: false,
        expensive_generation: true,
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

criterion_group!(streaming, streaming_benches);
criterion_main!(streaming);
