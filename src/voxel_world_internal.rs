///
/// Voxel World internals
/// This module contains the internal systems and resources used to implement bevy_voxel_world.
///
use bevy::{
    camera::primitives::{Frustum, Sphere},
    ecs::system::SystemParam,
    math::Vec3A,
    platform::collections::{HashMap, HashSet},
    prelude::*,
    tasks::AsyncComputeTaskPool,
};
use futures_lite::future;
use rand::RngExt;
use smallvec::SmallVec;
use std::{
    collections::VecDeque,
    marker::PhantomData,
    sync::{Arc, RwLock, TryLockError},
    time::Duration,
};

use crate::{
    chunk::*,
    chunk_map::*,
    configuration::{ChunkDespawnStrategy, ChunkSpawnStrategy, VoxelWorldConfig},
    mesh_cache::*,
    plugin::VoxelWorldMaterialHandle,
    prelude::default_chunk_meshing_delegate,
    voxel::WorldVoxel,
    voxel_material::LoadingTexture,
    voxel_world::{
        get_affected_chunk_positions, ChunkWillChangeLod, ChunkWillDespawn,
        ChunkWillRemesh, ChunkWillSpawn, ChunkWillUpdate, VoxelWorldCamera,
    },
};

#[derive(SystemParam, Deref)]
pub struct CameraInfo<'w, 's, C: VoxelWorldConfig>(
    Query<
        'w,
        's,
        (
            &'static Camera,
            &'static GlobalTransform,
            &'static Projection,
            &'static Frustum,
        ),
        With<VoxelWorldCamera<C>>,
    >,
);

/// Holds a map of modified voxels that will persist between chunk spawn/despawn
#[derive(Resource, Deref, DerefMut, Clone)]
pub struct ModifiedVoxels<C, I>(
    #[deref] Arc<RwLock<HashMap<IVec3, WorldVoxel<I>>>>,
    PhantomData<C>,
);

impl<C: VoxelWorldConfig> Default for ModifiedVoxels<C, C::MaterialIndex> {
    fn default() -> Self {
        Self(Arc::new(RwLock::new(HashMap::new())), PhantomData)
    }
}

impl<C: VoxelWorldConfig> ModifiedVoxels<C, C::MaterialIndex> {
    pub fn get_voxel(&self, position: &IVec3) -> Option<WorldVoxel<C::MaterialIndex>> {
        let modified_voxels = self.0.read().unwrap();
        modified_voxels.get(position).cloned()
    }
}

/// A temporary buffer for voxel modifications that will get flushed to the `ModifiedVoxels` resource
/// at the end of the frame.
#[derive(Resource, Deref, DerefMut, Default)]
pub struct VoxelWriteBuffer<C, I>(#[deref] Vec<(IVec3, WorldVoxel<I>)>, PhantomData<C>);

#[derive(Component)]
pub(crate) struct NeedsMaterial<C>(PhantomData<C>);

pub(crate) struct Internals<C>(PhantomData<C>);

#[derive(Default)]
pub(crate) struct DynamicIntervalGate {
    interval: Duration,
    timer: Option<Timer>,
}

impl DynamicIntervalGate {
    fn should_run(&mut self, interval: Duration, delta: Duration) -> bool {
        if interval.is_zero() {
            self.timer = None;
            self.interval = interval;
            return true;
        }

        let timer = match self.timer.as_mut() {
            Some(timer) if self.interval == interval => timer,
            _ => {
                self.interval = interval;
                self.timer = Some(Timer::new(interval, TimerMode::Repeating));
                self.timer.as_mut().expect("timer was just initialized")
            }
        };

        timer.tick(delta).just_finished()
    }
}

#[derive(Component)]
pub struct WorldRoot<C>(PhantomData<C>);

#[derive(Default)]
pub(crate) struct SpawnChunkScratch {
    visited: HashSet<IVec3>,
    chunks_deque: VecDeque<SpawnCandidate>,
    protected_offsets: Vec<IVec3>,
    protected_offsets_distance: i32,
}

struct SpawnCandidate {
    position: IVec3,
    known_missing_chunk: bool,
    protected_chunk: bool,
}

impl SpawnChunkScratch {
    fn protected_offsets(&mut self, distance: i32) -> &[IVec3] {
        if self.protected_offsets_distance == distance
            && !self.protected_offsets.is_empty()
        {
            return &self.protected_offsets;
        }

        self.protected_offsets.clear();
        self.protected_offsets_distance = distance;

        let distance_sq = distance * distance;
        for x in -distance..=distance {
            let x_sq = x * x;
            let y_limit = ((distance_sq - x_sq) as f32).sqrt() as i32;

            for y in -y_limit..=y_limit {
                let y_sq = y * y;
                let z_limit = ((distance_sq - x_sq - y_sq) as f32).sqrt() as i32;

                for z in -z_limit..=z_limit {
                    self.protected_offsets.push(IVec3::new(x, y, z));
                }
            }
        }

        &self.protected_offsets
    }
}

impl<C> Internals<C>
where
    C: VoxelWorldConfig,
{
    /// Init the resources used internally by bevy_voxel_world
    pub fn setup(mut commands: Commands, configuration: Res<C>) {
        commands.init_resource::<ChunkMap<C, C::MaterialIndex>>();
        commands.init_resource::<ChunkMapInsertBuffer<C, C::MaterialIndex>>();
        commands.init_resource::<ChunkMapUpdateBuffer<C, C::MaterialIndex>>();
        commands.init_resource::<ChunkMapRemoveBuffer<C>>();
        commands.init_resource::<MeshCache<C>>();
        commands.init_resource::<MeshCacheInsertBuffer<C>>();
        commands.init_resource::<ModifiedVoxels<C, C::MaterialIndex>>();
        commands.init_resource::<VoxelWriteBuffer<C, C::MaterialIndex>>();
        commands.init_resource::<TrackedCameras<C>>();

        // Create the root node and allow to modify it by the configuration.
        let world_root = commands
            .spawn((
                WorldRoot::<C>(PhantomData),
                Visibility::default(),
                Transform::default(),
            ))
            .id();
        configuration.init_root(commands, world_root)
    }

    pub fn refresh_camera_snapshot(
        mut cameras: ResMut<TrackedCameras<C>>,
        camera_info: CameraInfo<C>,
    ) {
        cameras.refresh(&camera_info);
    }

    /// Find and spawn chunks in need of spawning
    #[allow(clippy::too_many_arguments)]
    pub fn spawn_chunks(
        mut commands: Commands,
        mut chunk_map_insert_buffer: ResMut<ChunkMapInsertBuffer<C, C::MaterialIndex>>,
        world_root: Query<Entity, With<WorldRoot<C>>>,
        chunk_map: Res<ChunkMap<C, C::MaterialIndex>>,
        chunk_low_priority: Query<(), With<NeedsRemeshLowPriority>>,
        configuration: Res<C>,
        cameras: Res<TrackedCameras<C>>,
        mut scratch: Local<SpawnChunkScratch>,
    ) {
        // Panic if no root exists as it is already inserted in the setup.
        let world_root = world_root.single().unwrap();
        let attach_to_root = configuration.attach_chunks_to_root();

        if cameras.is_empty() {
            return;
        }

        let spawning_distance = configuration.spawning_distance() as i32;
        let spawning_distance_squared = spawning_distance * spawning_distance;
        let min_despawn_distance = configuration.min_despawn_distance() as i32;
        let spawn_strategy = configuration.chunk_spawn_strategy();
        let max_spawn_per_frame = configuration.max_spawn_per_frame();
        let spawning_rays = configuration.spawning_rays();

        let visibility_margin = if spawn_strategy == ChunkSpawnStrategy::CloseAndInView {
            cameras.viewport_margin_to_ndc(configuration.spawning_ray_margin())
        } else {
            Vec2::ZERO
        };
        let candidate_queue_limit = if max_spawn_per_frame == usize::MAX {
            usize::MAX
        } else {
            // Keep enough slack for duplicate and culled ray hits without letting
            // the candidate queue grow far beyond what this frame can spawn.
            let distance_scale = (spawning_distance as usize).div_ceil(128).max(1);
            let distance_queue_slack =
                max_spawn_per_frame.saturating_mul(distance_scale.saturating_sub(1));
            max_spawn_per_frame.saturating_add(
                (max_spawn_per_frame / 4)
                    .max(spawning_rays)
                    .max(distance_queue_slack),
            )
        };
        let mut spawn_ray_step_budget = if max_spawn_per_frame == usize::MAX {
            usize::MAX
        } else {
            candidate_queue_limit.saturating_mul(2).max(spawning_rays)
        };
        let queue_capacity = if candidate_queue_limit == usize::MAX {
            spawning_rays.saturating_mul(spawning_distance as usize)
        } else {
            candidate_queue_limit
        };
        scratch.visited.clear();
        let visited_capacity = scratch.visited.capacity();
        if visited_capacity < queue_capacity {
            scratch.visited.reserve(queue_capacity - visited_capacity);
        }
        let carryover_limit = if max_spawn_per_frame == usize::MAX {
            queue_capacity
        } else {
            queue_capacity.min(max_spawn_per_frame)
        };
        while scratch.chunks_deque.len() > carryover_limit {
            scratch.chunks_deque.pop_back();
        }
        for candidate in scratch.chunks_deque.iter_mut() {
            candidate.known_missing_chunk = false;
            candidate.protected_chunk = false;
        }
        let chunks_deque_capacity = scratch.chunks_deque.capacity();
        if chunks_deque_capacity < queue_capacity {
            scratch
                .chunks_deque
                .reserve(queue_capacity - chunks_deque_capacity);
        }
        scratch.protected_offsets(min_despawn_distance);
        let SpawnChunkScratch {
            visited,
            chunks_deque,
            protected_offsets,
            ..
        } = &mut *scratch;

        let Some(chunk_map_read_lock) = chunk_map.try_get_read_lock() else {
            return;
        };
        let has_low_priority_chunks = !chunk_low_priority.is_empty();
        let mut promote_low_priority = |chunk_data: &ChunkData<C::MaterialIndex>| {
            if !has_low_priority_chunks {
                return;
            }
            if chunk_low_priority.get(chunk_data.entity).is_ok() {
                if let Ok(mut entity_commands) = commands.get_entity(chunk_data.entity) {
                    entity_commands
                        .remove::<NeedsRemeshLowPriority>()
                        .try_insert(NeedsRemesh);
                }
            }
        };

        // Shoots a ray from the given point, and queue all (non-spawned) chunks intersecting the ray
        let mut queue_chunks_intersecting_ray_from_point =
            |camera: &TrackedCamera,
             point: Vec2,
             queue: &mut VecDeque<SpawnCandidate>,
             ray_step_budget: &mut usize| {
                let Ok(ray) = camera.camera.viewport_to_world(&camera.transform, point)
                else {
                    return;
                };
                let max_t = (spawning_distance * CHUNK_SIZE_I) as f32;
                let mut chunk_pos = world_position_to_chunk_position(ray.origin);
                let step = IVec3::new(
                    axis_step(ray.direction.x),
                    axis_step(ray.direction.y),
                    axis_step(ray.direction.z),
                );
                let mut t_max = Vec3::new(
                    initial_chunk_boundary_t(
                        ray.origin.x,
                        ray.direction.x,
                        chunk_pos.x,
                        step.x,
                    ),
                    initial_chunk_boundary_t(
                        ray.origin.y,
                        ray.direction.y,
                        chunk_pos.y,
                        step.y,
                    ),
                    initial_chunk_boundary_t(
                        ray.origin.z,
                        ray.direction.z,
                        chunk_pos.z,
                        step.z,
                    ),
                );
                let t_delta = Vec3::new(
                    chunk_t_delta(ray.direction.x),
                    chunk_t_delta(ray.direction.y),
                    chunk_t_delta(ray.direction.z),
                );
                let mut t = 0.0;
                while t < max_t
                    && queue.len() < candidate_queue_limit
                    && *ray_step_budget > 0
                {
                    *ray_step_budget -= 1;
                    if let Some(chunk) = ChunkMap::<C, C::MaterialIndex>::get_ref(
                        &chunk_pos,
                        &chunk_map_read_lock,
                    ) {
                        promote_low_priority(chunk);
                        if chunk.is_full {
                            // If we hit a full chunk, we can stop the ray early
                            break;
                        }
                    } else {
                        queue.push_back(SpawnCandidate {
                            position: chunk_pos,
                            known_missing_chunk: true,
                            protected_chunk: false,
                        });
                    }

                    let next_t = t_max.x.min(t_max.y).min(t_max.z);
                    if !next_t.is_finite() {
                        break;
                    }

                    const AXIS_EPSILON: f32 = 0.0001;
                    if t_max.x <= next_t + AXIS_EPSILON {
                        chunk_pos.x += step.x;
                        t_max.x += t_delta.x;
                    }
                    if t_max.y <= next_t + AXIS_EPSILON {
                        chunk_pos.y += step.y;
                        t_max.y += t_delta.y;
                    }
                    if t_max.z <= next_t + AXIS_EPSILON {
                        chunk_pos.z += step.z;
                        t_max.z += t_delta.z;
                    }
                    t = next_t;
                }
            };

        // Each frame we pick some random points on the screen
        let m = configuration.spawning_ray_margin();
        let mut rng = rand::rng();
        for ray_index in 0..spawning_rays {
            if chunks_deque.len() >= candidate_queue_limit || spawn_ray_step_budget == 0 {
                break;
            }
            let camera = cameras
                .camera_for_ray(ray_index)
                .expect("cameras is not empty");
            let viewport_size = camera.viewport_size;
            let random_point_in_viewport = {
                let x = rng.random::<f32>() * (viewport_size.x + m * 2) as f32 - m as f32;
                let y = rng.random::<f32>() * (viewport_size.y + m * 2) as f32 - m as f32;
                Vec2::new(x, y)
            };

            // Then, for each point, we cast a ray, picking up any unspawned chunks along the ray
            queue_chunks_intersecting_ray_from_point(
                camera,
                random_point_in_viewport,
                chunks_deque,
                &mut spawn_ray_step_budget,
            );
        }

        // We also queue the chunks closest to every camera to make sure they will always spawn early.
        cameras.for_each_unique_chunk_position(|camera_chunk_position| {
            for offset in protected_offsets.iter() {
                chunks_deque.push_front(SpawnCandidate {
                    position: camera_chunk_position + *offset,
                    known_missing_chunk: false,
                    protected_chunk: true,
                });
            }
        });

        // Then, when we have a queue of chunks, we can set them up for spawning
        let mut spawned_this_frame = 0;
        while let Some(candidate) = chunks_deque.pop_front() {
            if spawned_this_frame >= max_spawn_per_frame {
                break;
            }
            let chunk_position = candidate.position;
            if !visited.insert(chunk_position) {
                continue;
            }

            if !candidate.protected_chunk
                && !cameras.is_chunk_close(chunk_position, spawning_distance_squared)
            {
                continue;
            }

            if !candidate.protected_chunk
                && spawn_strategy == ChunkSpawnStrategy::CloseAndInView
                && !cameras.is_chunk_visible(chunk_position, visibility_margin)
            {
                continue;
            }

            let has_chunk = !candidate.known_missing_chunk
                && ChunkMap::<C, C::MaterialIndex>::contains_chunk(
                    &chunk_position,
                    &chunk_map_read_lock,
                );

            if !has_chunk {
                let translation = Transform::from_translation(
                    chunk_position.as_vec3() * CHUNK_SIZE_F - 1.0,
                );
                let chunk_entity = commands.spawn((NeedsRemesh, translation)).id();
                if attach_to_root {
                    commands.entity(world_root).add_child(chunk_entity);
                }
                let closest_camera = cameras.closest_camera_to_chunk(chunk_position);
                let lod_level = configuration.chunk_lod(
                    chunk_position,
                    None,
                    closest_camera.position,
                );
                let data_shape = configuration.chunk_data_shape(lod_level);
                let mesh_shape = configuration.chunk_meshing_shape(lod_level);
                let chunk = Chunk::<C>::new(
                    chunk_position,
                    lod_level,
                    chunk_entity,
                    data_shape,
                    mesh_shape,
                );

                let mut chunk_data = ChunkData::with_entity(chunk.entity);
                chunk_data.position = chunk_position;
                chunk_data.lod_level = lod_level;
                chunk_data.data_shape = data_shape;
                chunk_data.mesh_shape = mesh_shape;
                chunk_map_insert_buffer.push((chunk_position, chunk_data));

                commands.entity(chunk_entity).try_insert(chunk);
                spawned_this_frame += 1;
            } else {
                continue;
            }

            if spawn_strategy != ChunkSpawnStrategy::Close {
                continue;
            }

            // If we get here, we queue the neighbors
            'neighbors: for x in -1..=1 {
                for y in -1..=1 {
                    for z in -1..=1 {
                        let queue_pos = chunk_position + IVec3::new(x, y, z);
                        if queue_pos == chunk_position {
                            continue;
                        }
                        if chunks_deque.len() >= candidate_queue_limit {
                            break 'neighbors;
                        }
                        chunks_deque.push_back(SpawnCandidate {
                            position: queue_pos,
                            known_missing_chunk: false,
                            protected_chunk: false,
                        });
                    }
                }
            }
        }
    }

    /// Update chunk LOD assignments and schedule remeshing when a change occurs.
    #[allow(clippy::too_many_arguments)]
    pub fn update_chunk_lods(
        mut commands: Commands,
        mut chunks: Query<(Entity, &mut Chunk<C>), Without<NeedsDespawn>>,
        configuration: Res<C>,
        time: Res<Time>,
        mut interval_gate: Local<DynamicIntervalGate>,
        cameras: Res<TrackedCameras<C>>,
        mut ev_chunk_will_change_lod: MessageWriter<ChunkWillChangeLod<C>>,
    ) {
        if !interval_gate
            .should_run(configuration.chunk_lod_update_interval(), time.delta())
        {
            return;
        }
        if cameras.is_empty() {
            return;
        }

        let min_despawn_distance = configuration.min_despawn_distance() as i32;
        let min_despawn_distance_sq = min_despawn_distance * min_despawn_distance;

        for (entity, mut chunk) in chunks.iter_mut() {
            let closest_camera = cameras.closest_camera_to_chunk(chunk.position);
            let target_lod = configuration.chunk_lod(
                chunk.position,
                Some(chunk.lod_level),
                closest_camera.position,
            );
            if target_lod == chunk.lod_level {
                continue;
            }

            ev_chunk_will_change_lod
                .write(ChunkWillChangeLod::<C>::new(chunk.position, entity));

            let data_shape = configuration.chunk_data_shape(target_lod);
            let mesh_shape = configuration.chunk_meshing_shape(target_lod);

            if chunk.data_shape == data_shape && chunk.mesh_shape == mesh_shape {
                chunk.lod_level = target_lod;
                // Shape did not change, so nothing to regenerate/remesh.
                continue;
            }

            chunk.data_shape = data_shape;
            chunk.mesh_shape = mesh_shape;
            chunk.lod_level = target_lod;

            let mut entity_commands = commands.entity(entity);
            if chunk
                .position
                .distance_squared(closest_camera.chunk_position)
                <= min_despawn_distance_sq
            {
                entity_commands
                    .try_insert(NeedsRemesh)
                    .remove::<NeedsRemeshLowPriority>();
            } else {
                entity_commands
                    .try_insert(NeedsRemeshLowPriority)
                    .remove::<NeedsRemesh>();
            }
            entity_commands.remove::<ChunkThread<C, C::MaterialIndex>>();
        }
    }

    /// Tags chunks that are eligible for despawning
    #[allow(clippy::too_many_arguments)]
    pub fn retire_chunks(
        mut commands: Commands,
        all_chunks: Query<&Chunk<C>, Without<NeedsDespawn>>,
        configuration: Res<C>,
        time: Res<Time>,
        mut interval_gate: Local<DynamicIntervalGate>,
        cameras: Res<TrackedCameras<C>>,
        mut ev_chunk_will_despawn: MessageWriter<ChunkWillDespawn<C>>,
    ) {
        if !interval_gate.should_run(configuration.retire_chunks_interval(), time.delta())
        {
            return;
        }
        if configuration.max_chunk_despawns_per_frame() == 0 {
            return;
        }

        if cameras.is_empty() {
            return;
        }

        let spawning_distance = configuration.spawning_distance() as i32;
        let spawning_distance_squared = spawning_distance * spawning_distance;
        let min_despawn_distance = configuration.min_despawn_distance() as i32;
        let near_distance_squared = min_despawn_distance * min_despawn_distance;
        let strategy = configuration.chunk_despawn_strategy();

        match strategy {
            ChunkDespawnStrategy::FarAway => {
                for chunk in all_chunks.iter() {
                    if cameras
                        .is_chunk_close(chunk.position, spawning_distance_squared + 1)
                    {
                        continue;
                    }

                    commands
                        .entity(chunk.entity)
                        .try_insert(NeedsDespawn)
                        .remove::<NeedsRemesh>()
                        .remove::<NeedsRemeshLowPriority>();
                    ev_chunk_will_despawn
                        .write(ChunkWillDespawn::<C>::new(chunk.position, chunk.entity));
                }
            }
            ChunkDespawnStrategy::FarAwayOrOutOfView => {
                let visibility_margin =
                    cameras.viewport_margin_to_ndc(configuration.spawning_ray_margin());
                for chunk in all_chunks.iter() {
                    if !cameras.is_chunk_close_and_visible(
                        chunk.position,
                        spawning_distance_squared + 1,
                        near_distance_squared,
                        visibility_margin,
                    ) {
                        commands
                            .entity(chunk.entity)
                            .try_insert(NeedsDespawn)
                            .remove::<NeedsRemesh>()
                            .remove::<NeedsRemeshLowPriority>();
                        ev_chunk_will_despawn.write(ChunkWillDespawn::<C>::new(
                            chunk.position,
                            chunk.entity,
                        ));
                    }
                }
            }
        }
    }

    /// Despawns chunks that have been tagged for despawning
    pub fn despawn_retired_chunks(
        mut commands: Commands,
        mut chunk_map_remove_buffer: ResMut<ChunkMapRemoveBuffer<C>>,
        configuration: Res<C>,
        retired_chunks: Query<(Entity, &Chunk<C>), With<NeedsDespawn>>,
    ) {
        let max_despawns = configuration.max_chunk_despawns_per_frame();
        if max_despawns == 0 {
            return;
        }

        for (retired, (entity, chunk)) in retired_chunks.iter().enumerate() {
            if retired >= max_despawns {
                break;
            }

            commands.entity(entity).despawn();
            chunk_map_remove_buffer.push(chunk.position);
        }
    }

    /// Spawn a thread for each chunk that has been marked by NeedsRemesh
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    pub fn remesh_dirty_chunks(
        mut commands: Commands,
        mut ev_chunk_will_remesh: MessageWriter<ChunkWillRemesh<C>>,
        dirty_chunks: Query<
            &Chunk<C>,
            (
                With<NeedsRemesh>,
                Without<NeedsDespawn>,
                Without<ChunkThread<C, C::MaterialIndex>>,
            ),
        >,
        dirty_chunks_low: Query<
            &Chunk<C>,
            (
                With<NeedsRemeshLowPriority>,
                Without<NeedsDespawn>,
                Without<ChunkThread<C, C::MaterialIndex>>,
            ),
        >,
        chunk_threads: Query<(), With<ChunkThread<C, C::MaterialIndex>>>,
        mesh_cache: Res<MeshCache<C>>,
        modified_voxels: Res<ModifiedVoxels<C, C::MaterialIndex>>,
        chunk_map: Res<ChunkMap<C, C::MaterialIndex>>,
        configuration: Res<C>,
    ) {
        let thread_pool = AsyncComputeTaskPool::get();
        let max_threads = configuration.max_active_chunk_threads();
        let active_threads = chunk_threads.iter().len();
        let available_threads = max_threads.saturating_sub(active_threads);

        if max_threads == 0 {
            return;
        }

        let Some(chunk_map_read_lock) = chunk_map.try_get_read_lock() else {
            return;
        };

        let regenerate_strategy = configuration.chunk_regenerate_strategy();
        let voxel_lookup_delegate = configuration.voxel_lookup_delegate();
        let chunk_meshing_delegate = configuration.chunk_meshing_delegate();
        let texture_index_mapper = configuration.texture_index_mapper();
        let mesh_map = mesh_cache.get_mesh_map();

        for chunk in dirty_chunks
            .iter()
            .chain(dirty_chunks_low.iter())
            .take(available_threads)
        {
            let previous_chunk_data = ChunkMap::<C, C::MaterialIndex>::get(
                &chunk.position,
                &chunk_map_read_lock,
            );

            let lod_level = chunk.lod_level;

            let voxel_data_fn = voxel_lookup_delegate(
                chunk.position,
                lod_level,
                previous_chunk_data.clone(),
            );
            let data_shape = chunk.data_shape;
            let mesh_shape = chunk.mesh_shape;
            let chunk_meshing_fn =
                if let Some(chunk_meshing_delegate) = chunk_meshing_delegate.as_ref() {
                    chunk_meshing_delegate(
                        chunk.position,
                        lod_level,
                        data_shape,
                        mesh_shape,
                        previous_chunk_data.clone(),
                    )
                } else {
                    default_chunk_meshing_delegate(
                        chunk.position,
                        lod_level,
                        data_shape,
                        mesh_shape,
                        previous_chunk_data.clone(),
                    )
                };
            let texture_index_mapper = texture_index_mapper.clone();

            let mut chunk_task = ChunkTask::<C, C::MaterialIndex>::new(
                chunk.entity,
                chunk.position,
                lod_level,
                data_shape,
                mesh_shape,
                modified_voxels.clone(),
            );

            let mesh_map = mesh_map.clone();

            let thread = thread_pool.spawn(async move {
                chunk_task.generate(
                    voxel_data_fn,
                    previous_chunk_data.clone(),
                    regenerate_strategy,
                );

                // No need to mesh if the chunk is empty or full
                if chunk_task.is_empty() || chunk_task.is_full() {
                    return chunk_task;
                }

                // Also no need to mesh if a matching mesh is already cached
                let mesh_cache_hit = mesh_map
                    .read()
                    .unwrap()
                    .contains_key(&chunk_task.voxels_hash());
                if !mesh_cache_hit {
                    chunk_task.mesh(chunk_meshing_fn, texture_index_mapper);
                }

                chunk_task
            });

            commands
                .entity(chunk.entity)
                .try_insert(ChunkThread::<C, C::MaterialIndex>::new(
                    thread,
                    chunk.position,
                ))
                .remove::<NeedsRemesh>()
                .remove::<NeedsRemeshLowPriority>();

            ev_chunk_will_remesh
                .write(ChunkWillRemesh::<C>::new(chunk.position, chunk.entity));
        }
    }

    /// Inserts new meshes for chunks that have just finished remeshing
    #[allow(clippy::type_complexity)]
    pub fn spawn_meshes(
        mut commands: Commands,
        mut chunking_threads: Query<
            (Entity, &mut ChunkThread<C, C::MaterialIndex>, &mut Chunk<C>),
            Without<NeedsRemesh>,
        >,
        mut mesh_assets: ResMut<Assets<Mesh>>,
        buffers: (
            ResMut<ChunkMapUpdateBuffer<C, C::MaterialIndex>>,
            ResMut<MeshCacheInsertBuffer<C>>,
        ),
        res: (Res<MeshCache<C>>, Res<LoadingTexture>),
    ) {
        let (mesh_cache, loading_texture) = res;

        if !loading_texture.is_loaded {
            return;
        }

        let (mut chunk_map_update_buffer, mut mesh_cache_insert_buffer) = buffers;
        let mesh_handles = mesh_cache.mesh_handles();
        let user_bundles = mesh_cache.user_bundles();

        for (entity, mut thread, chunk) in &mut chunking_threads {
            if !thread.0.is_finished() {
                continue;
            }

            let chunk_task = future::block_on(&mut thread.0);
            let mut entity_commands = commands.entity(entity);

            if chunk_task.is_empty() || chunk_task.is_full() {
                entity_commands
                    .remove::<Mesh3d>()
                    .remove::<MeshRef>()
                    .remove::<NeedsMaterial<C>>()
                    .remove::<C::ChunkUserBundle>();
            } else {
                let hash = chunk_task.voxels_hash();
                let mesh_handle = {
                    if let Some(mesh_handle) = mesh_handles.get(&hash) {
                        if let Some(user_bundle) = user_bundles.get(&hash).cloned() {
                            entity_commands.insert(user_bundle);
                        }

                        mesh_handle
                    } else {
                        let Some(mesh) = chunk_task.mesh else {
                            entity_commands
                                .try_insert(NeedsRemesh)
                                .remove::<NeedsRemeshLowPriority>()
                                .remove::<ChunkThread<C, C::MaterialIndex>>();
                            continue;
                        };

                        // Bevy 0.19 mesh slab allocator can emit use-after-free
                        // errors if zero-vertex meshes are uploaded.
                        if mesh.count_vertices() == 0 {
                            entity_commands
                                .remove::<Mesh3d>()
                                .remove::<MeshRef>()
                                .remove::<NeedsMaterial<C>>()
                                .remove::<C::ChunkUserBundle>();
                            chunk_map_update_buffer.push((
                                chunk.position,
                                chunk_task.chunk_data,
                                ChunkWillSpawn::<C>::new(chunk_task.position, entity),
                            ));
                            entity_commands.remove::<ChunkThread<C, C::MaterialIndex>>();
                            continue;
                        }

                        let mesh_ref = Arc::new(mesh_assets.add(mesh));
                        let user_bundle = chunk_task.user_bundle;

                        mesh_cache_insert_buffer.push((
                            hash,
                            mesh_ref.clone(),
                            user_bundle.clone(),
                        ));
                        if let Some(bundle) = user_bundle {
                            entity_commands.insert(bundle);
                        }
                        mesh_ref
                    }
                };

                entity_commands
                    .try_insert((MeshRef(mesh_handle), NeedsMaterial::<C>(PhantomData)));
            }

            chunk_map_update_buffer.push((
                chunk.position,
                chunk_task.chunk_data,
                ChunkWillSpawn::<C>::new(chunk_task.position, entity),
            ));

            entity_commands.remove::<ChunkThread<C, C::MaterialIndex>>();
        }
    }

    pub fn flush_voxel_write_buffer(
        mut commands: Commands,
        mut buffer: ResMut<VoxelWriteBuffer<C, C::MaterialIndex>>,
        mut ev_chunk_will_update: MessageWriter<ChunkWillUpdate<C>>,
        chunk_map: Res<ChunkMap<C, C::MaterialIndex>>,
        modified_voxels: ResMut<ModifiedVoxels<C, C::MaterialIndex>>,
    ) {
        if buffer.is_empty() {
            return;
        }

        let Some(chunk_map_read_lock) = chunk_map.try_get_read_lock() else {
            return;
        };
        let mut modified_voxels = match modified_voxels.try_write() {
            Ok(guard) => guard,
            Err(TryLockError::WouldBlock) => return,
            Err(TryLockError::Poisoned(err)) => {
                panic!("ModifiedVoxels write lock poisoned: {err}");
            }
        };

        for (position, voxel) in buffer.iter() {
            // Skip writes that don't actually change the voxel value.
            // Without this guard, repeatedly setting a voxel to the same value
            // every frame causes perpetual chunk remesh cancellation — the
            // in-progress async meshing task gets dropped before it can finish,
            // so the chunk never renders.
            if modified_voxels.get(position) == Some(voxel) {
                continue;
            }

            modified_voxels.insert(*position, *voxel);

            for affected_chunk_pos in get_affected_chunk_positions(*position) {
                if let Some(chunk_data) = ChunkMap::<C, C::MaterialIndex>::get(
                    &affected_chunk_pos,
                    &chunk_map_read_lock,
                ) {
                    if let Ok(mut ent) = commands.get_entity(chunk_data.entity) {
                        ent.try_insert(NeedsRemesh)
                            .remove::<NeedsRemeshLowPriority>();
                        ent.remove::<ChunkThread<C, C::MaterialIndex>>();
                        ev_chunk_will_update.write(ChunkWillUpdate::<C>::new(
                            affected_chunk_pos,
                            chunk_data.entity,
                        ));
                    }
                }
            }
        }

        buffer.clear();
    }

    pub fn flush_mesh_cache_buffers(
        mut mesh_cache_insert_buffer: ResMut<MeshCacheInsertBuffer<C>>,
        mesh_cache: Res<MeshCache<C>>,
    ) {
        mesh_cache.apply_buffers(&mut mesh_cache_insert_buffer);
    }

    pub fn flush_chunk_map_buffers(
        mut chunk_map_insert_buffer: ResMut<ChunkMapInsertBuffer<C, C::MaterialIndex>>,
        mut chunk_map_update_buffer: ResMut<ChunkMapUpdateBuffer<C, C::MaterialIndex>>,
        mut chunk_map_remove_buffer: ResMut<ChunkMapRemoveBuffer<C>>,
        mut ev_chunk_will_spawn: MessageWriter<ChunkWillSpawn<C>>,
        chunk_map: Res<ChunkMap<C, C::MaterialIndex>>,
    ) {
        chunk_map.apply_buffers(
            &mut chunk_map_insert_buffer,
            &mut chunk_map_update_buffer,
            &mut chunk_map_remove_buffer,
            &mut ev_chunk_will_spawn,
        );
    }

    pub(crate) fn assign_material<M: Material>(
        mut commands: Commands,
        mut needs_material: Query<(Entity, &MeshRef), With<NeedsMaterial<C>>>,
        material_handle: Option<Res<VoxelWorldMaterialHandle<M>>>,
    ) {
        let Some(material_handle) = material_handle else {
            return;
        };

        for (entity, mesh_ref) in needs_material.iter_mut() {
            commands
                .entity(entity)
                .insert(Mesh3d((*mesh_ref.0).clone()))
                .insert(MeshMaterial3d(material_handle.handle.clone()))
                .remove::<NeedsMaterial<C>>();
        }
    }
}

struct TrackedCamera {
    camera: Camera,
    transform: GlobalTransform,
    visibility: ChunkVisibilityVolume,
    viewport_size: UVec2,
    position: Vec3,
    chunk_position: IVec3,
}

impl TrackedCamera {
    fn new(
        camera: &Camera,
        transform: &GlobalTransform,
        projection: &Projection,
        frustum: &Frustum,
    ) -> Self {
        let position = transform.translation();
        Self {
            camera: camera.clone(),
            transform: *transform,
            visibility: ChunkVisibilityVolume::new(transform, projection, frustum),
            viewport_size: camera.physical_viewport_size().unwrap_or_default(),
            position,
            chunk_position: position.as_ivec3() / CHUNK_SIZE_I,
        }
    }

    #[inline]
    fn has_same_view(&self, other: &Self) -> bool {
        self.viewport_size == other.viewport_size
            && self.visibility.has_same_shape(&other.visibility)
    }
}

#[derive(Resource)]
pub(crate) struct TrackedCameras<C: VoxelWorldConfig> {
    cameras: SmallVec<[TrackedCamera; 2]>,
    _marker: PhantomData<C>,
}

impl<C: VoxelWorldConfig> Default for TrackedCameras<C> {
    fn default() -> Self {
        Self {
            cameras: SmallVec::new(),
            _marker: PhantomData,
        }
    }
}

impl<C: VoxelWorldConfig> TrackedCameras<C> {
    #[inline]
    fn refresh(&mut self, camera_info: &CameraInfo<C>) {
        self.cameras.clear();
        for (camera, transform, projection, frustum) in camera_info.iter() {
            let tracked = TrackedCamera::new(camera, transform, projection, frustum);
            if !self
                .cameras
                .iter()
                .any(|camera| camera.has_same_view(&tracked))
            {
                self.cameras.push(tracked);
            }
        }
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.cameras.is_empty()
    }

    #[inline(always)]
    fn camera_for_ray(&self, ray_index: usize) -> Option<&TrackedCamera> {
        match self.cameras.len() {
            0 => None,
            1 => Some(&self.cameras[0]),
            len => Some(&self.cameras[ray_index % len]),
        }
    }

    #[inline(always)]
    fn for_each_unique_chunk_position(&self, mut f: impl FnMut(IVec3)) {
        let mut seen = SmallVec::<[IVec3; 2]>::new();
        for camera in &self.cameras {
            if seen.contains(&camera.chunk_position) {
                continue;
            }
            seen.push(camera.chunk_position);
            f(camera.chunk_position);
        }
    }

    #[inline(always)]
    fn viewport_margin_to_ndc(&self, margin: u32) -> Vec2 {
        match self.cameras.len() {
            0 => Vec2::ZERO,
            1 => viewport_margin_to_ndc(self.cameras[0].viewport_size, margin),
            _ => cameras_viewport_margin_to_ndc(&self.cameras, margin),
        }
    }

    #[inline(always)]
    fn is_chunk_close(&self, chunk_position: IVec3, distance_squared: i32) -> bool {
        match self.cameras.len() {
            0 => false,
            1 => {
                chunk_position.distance_squared(self.cameras[0].chunk_position)
                    <= distance_squared
            }
            _ => chunk_is_close_to_any_camera(
                &self.cameras,
                chunk_position,
                distance_squared,
            ),
        }
    }

    #[inline(always)]
    fn closest_camera_to_chunk(&self, chunk_position: IVec3) -> &TrackedCamera {
        match self.cameras.len() {
            0 => unreachable!("closest_camera_to_chunk requires at least one camera"),
            1 => &self.cameras[0],
            _ => closest_camera_to_chunk(&self.cameras, chunk_position),
        }
    }

    // This intentionally ignores distance. Callers pair it with their own
    // close/protected checks so spawn and despawn share the same visibility math.
    #[inline(always)]
    fn is_chunk_visible(&self, chunk_position: IVec3, ndc_margin: Vec2) -> bool {
        match self.cameras.len() {
            0 => false,
            1 => self.cameras[0]
                .visibility
                .contains_chunk(chunk_position, ndc_margin),
            _ => self.cameras.iter().any(|camera| {
                camera.visibility.contains_chunk(chunk_position, ndc_margin)
            }),
        }
    }

    #[inline(always)]
    fn is_chunk_close_and_visible(
        &self,
        chunk_position: IVec3,
        distance_squared: i32,
        protected_distance_squared: i32,
        ndc_margin: Vec2,
    ) -> bool {
        match self.cameras.len() {
            0 => false,
            1 => camera_protects_or_sees_close_chunk(
                &self.cameras[0],
                chunk_position,
                distance_squared,
                protected_distance_squared,
                ndc_margin,
            ),
            _ => self.cameras.iter().any(|camera| {
                camera_protects_or_sees_close_chunk(
                    camera,
                    chunk_position,
                    distance_squared,
                    protected_distance_squared,
                    ndc_margin,
                )
            }),
        }
    }
}

fn chunk_is_close_to_any_camera(
    cameras: &[TrackedCamera],
    chunk_position: IVec3,
    distance_squared: i32,
) -> bool {
    cameras.iter().any(|camera| {
        chunk_position.distance_squared(camera.chunk_position) <= distance_squared
    })
}

#[inline(always)]
fn camera_protects_or_sees_close_chunk(
    camera: &TrackedCamera,
    chunk_position: IVec3,
    distance_squared: i32,
    protected_distance_squared: i32,
    ndc_margin: Vec2,
) -> bool {
    let chunk_distance_squared = chunk_position.distance_squared(camera.chunk_position);
    chunk_distance_squared <= protected_distance_squared
        || (chunk_distance_squared <= distance_squared
            && camera.visibility.contains_chunk(chunk_position, ndc_margin))
}

fn closest_camera_to_chunk(
    cameras: &[TrackedCamera],
    chunk_position: IVec3,
) -> &TrackedCamera {
    let chunk_world_position = chunk_position.as_vec3() * CHUNK_SIZE_F;
    let mut closest_camera = &cameras[0];
    let mut closest_distance_squared = closest_camera
        .position
        .distance_squared(chunk_world_position);

    for camera in &cameras[1..] {
        let distance_squared = camera.position.distance_squared(chunk_world_position);
        if distance_squared < closest_distance_squared {
            closest_camera = camera;
            closest_distance_squared = distance_squared;
        }
    }

    closest_camera
}

fn cameras_viewport_margin_to_ndc(cameras: &[TrackedCamera], margin: u32) -> Vec2 {
    cameras.iter().fold(Vec2::ZERO, |max_margin, camera| {
        max_margin.max(viewport_margin_to_ndc(camera.viewport_size, margin))
    })
}

const SQRT_3: f32 = 1.732_050_8;
const CHUNK_CENTER_OFFSET: Vec3 = Vec3::splat(CHUNK_SIZE_F * 0.5);
const CHUNK_BOUNDING_SPHERE_RADIUS: f32 = 0.5 * CHUNK_SIZE_F * SQRT_3;

#[inline]
fn world_position_to_chunk_position(position: Vec3) -> IVec3 {
    IVec3::new(
        (position.x / CHUNK_SIZE_F).floor() as i32,
        (position.y / CHUNK_SIZE_F).floor() as i32,
        (position.z / CHUNK_SIZE_F).floor() as i32,
    )
}

#[inline]
fn axis_step(direction: f32) -> i32 {
    if direction > 0.0 {
        1
    } else if direction < 0.0 {
        -1
    } else {
        0
    }
}

#[inline]
fn initial_chunk_boundary_t(
    origin: f32,
    direction: f32,
    chunk_coordinate: i32,
    step: i32,
) -> f32 {
    if step == 0 {
        return f32::INFINITY;
    }

    let next_chunk_boundary = if step > 0 {
        chunk_coordinate + 1
    } else {
        chunk_coordinate
    } as f32
        * CHUNK_SIZE_F;

    (next_chunk_boundary - origin) / direction
}

#[inline]
fn chunk_t_delta(direction: f32) -> f32 {
    if direction == 0.0 {
        f32::INFINITY
    } else {
        CHUNK_SIZE_F / direction.abs()
    }
}

#[inline]
fn viewport_margin_to_ndc(viewport_size: UVec2, margin: u32) -> Vec2 {
    if margin == 0 || viewport_size.x == 0 || viewport_size.y == 0 {
        return Vec2::ZERO;
    }

    let margin = margin as f32 * 2.0;
    Vec2::new(
        margin / viewport_size.x as f32,
        margin / viewport_size.y as f32,
    )
}

enum ChunkVisibilityVolume {
    Perspective {
        origin: Vec3,
        forward: Vec3,
        right: Vec3,
        up: Vec3,
        tan_half_fov_y: f32,
        tan_half_fov_x: f32,
        near: f32,
        far: f32,
    },
    Frustum(Frustum),
}

impl ChunkVisibilityVolume {
    fn new(
        camera_transform: &GlobalTransform,
        projection: &Projection,
        frustum: &Frustum,
    ) -> Self {
        match projection {
            Projection::Perspective(projection) => {
                let tan_half_fov_y = (projection.fov * 0.5).tan();
                Self::Perspective {
                    origin: camera_transform.translation(),
                    forward: *camera_transform.forward(),
                    right: *camera_transform.right(),
                    up: *camera_transform.up(),
                    tan_half_fov_y,
                    tan_half_fov_x: tan_half_fov_y * projection.aspect_ratio,
                    near: projection.near,
                    far: projection.far,
                }
            }
            _ => Self::Frustum(*frustum),
        }
    }

    #[inline]
    fn has_same_shape(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Perspective {
                    origin,
                    forward,
                    right,
                    up,
                    tan_half_fov_y,
                    tan_half_fov_x,
                    near,
                    far,
                },
                Self::Perspective {
                    origin: other_origin,
                    forward: other_forward,
                    right: other_right,
                    up: other_up,
                    tan_half_fov_y: other_tan_half_fov_y,
                    tan_half_fov_x: other_tan_half_fov_x,
                    near: other_near,
                    far: other_far,
                },
            ) => {
                origin == other_origin
                    && forward == other_forward
                    && right == other_right
                    && up == other_up
                    && tan_half_fov_y == other_tan_half_fov_y
                    && tan_half_fov_x == other_tan_half_fov_x
                    && near == other_near
                    && far == other_far
            }
            (Self::Frustum(_), Self::Frustum(_)) => false,
            _ => false,
        }
    }

    #[inline]
    fn contains_chunk(&self, chunk_position: IVec3, ndc_margin: Vec2) -> bool {
        let chunk_center = chunk_position.as_vec3() * CHUNK_SIZE_F + CHUNK_CENTER_OFFSET;
        let radius = CHUNK_BOUNDING_SPHERE_RADIUS;

        match self {
            Self::Perspective {
                origin,
                forward,
                right,
                up,
                tan_half_fov_y,
                tan_half_fov_x,
                near,
                far,
            } => {
                let to_chunk = chunk_center - *origin;
                let depth = to_chunk.dot(*forward);
                if depth + radius < *near || depth - radius > *far {
                    return false;
                }

                let projected_depth = depth.max(0.0);
                let max_x =
                    projected_depth * *tan_half_fov_x * (1.0 + ndc_margin.x) + radius;
                if to_chunk.dot(*right).abs() > max_x {
                    return false;
                }

                let max_y =
                    projected_depth * *tan_half_fov_y * (1.0 + ndc_margin.y) + radius;
                to_chunk.dot(*up).abs() <= max_y
            }
            Self::Frustum(frustum) => {
                let radius = radius + ndc_margin.max_element() * CHUNK_SIZE_F;
                let sphere = Sphere {
                    center: Vec3A::from(chunk_center),
                    radius,
                };

                frustum.intersects_sphere(&sphere, true)
            }
        }
    }
}

/// Check if the given world point is within the camera's view
#[inline]
#[allow(dead_code)]
fn is_in_view(
    world_point: Vec3,
    camera: &Camera,
    cam_global_transform: &GlobalTransform,
) -> bool {
    if let Some(chunk_vp) = camera.world_to_ndc(cam_global_transform, world_point) {
        // When the position is within the viewport the values returned will be between
        // -1.0 and 1.0 on the X and Y axes, and between 0.0 and 1.0 on the Z axis.
        chunk_vp.x >= -1.0
            && chunk_vp.x <= 1.0
            && chunk_vp.y >= -1.0
            && chunk_vp.y <= 1.0
            && chunk_vp.z >= 0.0
            && chunk_vp.z <= 1.0
    } else {
        false
    }
}
