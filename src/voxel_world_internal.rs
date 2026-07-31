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
use std::{
    collections::VecDeque,
    marker::PhantomData,
    sync::{Arc, RwLock, TryLockError},
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
        (&'static Camera, &'static GlobalTransform, &'static Frustum),
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

#[derive(Component)]
pub struct WorldRoot<C>(PhantomData<C>);

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

    /// Find and spawn chunks in need of spawning
    pub fn spawn_chunks(
        mut commands: Commands,
        mut chunk_map_insert_buffer: ResMut<ChunkMapInsertBuffer<C, C::MaterialIndex>>,
        world_root: Query<Entity, With<WorldRoot<C>>>,
        chunk_map: Res<ChunkMap<C, C::MaterialIndex>>,
        chunk_low_priority: Query<(), With<NeedsRemeshLowPriority>>,
        configuration: Res<C>,
        camera_info: CameraInfo<C>,
    ) {
        // Panic if no root exists as it is already inserted in the setup.
        let world_root = world_root.single().unwrap();
        let attach_to_root = configuration.attach_chunks_to_root();

        let cameras: Vec<_> = camera_info
            .iter()
            .map(|(camera, transform, frustum)| {
                TrackedCamera::new(camera, transform, frustum)
            })
            .collect();
        if cameras.is_empty() {
            return;
        }

        let spawning_distance = configuration.spawning_distance() as i32;
        let spawning_distance_squared = spawning_distance.pow(2);
        let spawn_strategy = configuration.chunk_spawn_strategy();
        let max_spawn_per_frame = configuration.max_spawn_per_frame();

        let visibility_margin = 0.0f32;
        let protected_chunk_radius_sq =
            (configuration.min_despawn_distance() as i32).pow(2);

        let mut visited = HashSet::new();
        let mut chunks_deque = VecDeque::with_capacity(
            configuration.spawning_rays() * spawning_distance as usize,
        );

        let Some(chunk_map_read_lock) = chunk_map.try_get_read_lock() else {
            return;
        };
        let mut promote_low_priority = |chunk_data: &ChunkData<C::MaterialIndex>| {
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
            |camera: &TrackedCamera, point: Vec2, queue: &mut VecDeque<IVec3>| {
                let Ok(ray) = camera.camera.viewport_to_world(camera.transform, point)
                else {
                    return;
                };
                let mut current = ray.origin;
                let mut t = 0.0;
                while t < (spawning_distance * CHUNK_SIZE_I) as f32 {
                    let chunk_pos = current.as_ivec3() / CHUNK_SIZE_I;
                    if let Some(chunk) = ChunkMap::<C, C::MaterialIndex>::get(
                        &chunk_pos,
                        &chunk_map_read_lock,
                    ) {
                        promote_low_priority(&chunk);
                        if chunk.is_full {
                            // If we hit a full chunk, we can stop the ray early
                            break;
                        }
                    } else {
                        queue.push_back(chunk_pos);
                    }
                    t += CHUNK_SIZE_F;
                    current = ray.origin + ray.direction * t;
                }
            };

        // Each frame we pick some random points on the screen
        let m = configuration.spawning_ray_margin();
        for _ in 0..configuration.spawning_rays() {
            let camera = &cameras[rand::random_range(0..cameras.len())];
            let viewport_size =
                camera.camera.physical_viewport_size().unwrap_or_default();
            let random_point_in_viewport = {
                let x =
                    rand::random::<f32>() * (viewport_size.x + m * 2) as f32 - m as f32;
                let y =
                    rand::random::<f32>() * (viewport_size.y + m * 2) as f32 - m as f32;
                Vec2::new(x, y)
            };

            // Then, for each point, we cast a ray, picking up any unspawned chunks along the ray
            queue_chunks_intersecting_ray_from_point(
                camera,
                random_point_in_viewport,
                &mut chunks_deque,
            );
        }

        // We also queue the chunks closest to every camera to make sure they will always spawn early.
        let protected_distance = configuration.min_despawn_distance() as i32;
        for camera in &cameras {
            for x in -protected_distance..=protected_distance {
                for y in -protected_distance..=protected_distance {
                    for z in -protected_distance..=protected_distance {
                        let queue_pos = camera.chunk_position + IVec3::new(x, y, z);
                        chunks_deque.push_front(queue_pos);
                    }
                }
            }
        }

        // Then, when we have a queue of chunks, we can set them up for spawning
        let mut spawned_this_frame = 0;
        while let Some(chunk_position) = chunks_deque.pop_front() {
            if spawned_this_frame >= max_spawn_per_frame {
                break;
            }
            if visited.contains(&chunk_position) {
                continue;
            }
            visited.insert(chunk_position);

            if !chunk_is_close_to_any_camera(
                &cameras,
                chunk_position,
                spawning_distance_squared,
            ) {
                continue;
            }

            if spawn_strategy == ChunkSpawnStrategy::CloseAndInView
                && !chunk_visible_to_any_camera(
                    &cameras,
                    chunk_position,
                    visibility_margin,
                )
                && !chunk_is_close_to_any_camera(
                    &cameras,
                    chunk_position,
                    protected_chunk_radius_sq,
                )
            {
                continue;
            }

            let has_chunk = ChunkMap::<C, C::MaterialIndex>::contains_chunk(
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
                let camera_position =
                    closest_camera_position_to_chunk(&cameras, chunk_position);
                let lod_level =
                    configuration.chunk_lod(chunk_position, None, camera_position);
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
            for x in -1..=1 {
                for y in -1..=1 {
                    for z in -1..=1 {
                        let queue_pos = chunk_position + IVec3::new(x, y, z);
                        if queue_pos == chunk_position {
                            continue;
                        }
                        chunks_deque.push_back(queue_pos);
                    }
                }
            }
        }
    }

    /// Update chunk LOD assignments and schedule remeshing when a change occurs.
    pub fn update_chunk_lods(
        mut commands: Commands,
        mut chunks: Query<(Entity, &mut Chunk<C>), Without<NeedsDespawn>>,
        configuration: Res<C>,
        camera_info: CameraInfo<C>,
        mut ev_chunk_will_change_lod: MessageWriter<ChunkWillChangeLod<C>>,
    ) {
        let cameras: Vec<_> = camera_info
            .iter()
            .map(|(camera, transform, frustum)| {
                TrackedCamera::new(camera, transform, frustum)
            })
            .collect();
        if cameras.is_empty() {
            return;
        }

        let min_despawn_distance_sq =
            (configuration.min_despawn_distance() as i32).pow(2);

        for (entity, mut chunk) in chunks.iter_mut() {
            let camera_position =
                closest_camera_position_to_chunk(&cameras, chunk.position);
            let target_lod = configuration.chunk_lod(
                chunk.position,
                Some(chunk.lod_level),
                camera_position,
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
            if chunk_is_close_to_any_camera(
                &cameras,
                chunk.position,
                min_despawn_distance_sq,
            ) {
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
    pub fn retire_chunks(
        mut commands: Commands,
        all_chunks: Query<(&Chunk<C>, Option<&ViewVisibility>), Without<NeedsDespawn>>,
        configuration: Res<C>,
        camera_info: CameraInfo<C>,
        mut ev_chunk_will_despawn: MessageWriter<ChunkWillDespawn<C>>,
    ) {
        if configuration.max_chunk_despawns_per_frame() == 0 {
            return;
        }

        let cameras: Vec<_> = camera_info
            .iter()
            .map(|(camera, transform, frustum)| {
                TrackedCamera::new(camera, transform, frustum)
            })
            .collect();
        if cameras.is_empty() {
            return;
        }

        let spawning_distance = configuration.spawning_distance() as i32;
        let spawning_distance_squared = spawning_distance.pow(2);
        let near_distance_squared = (configuration.min_despawn_distance() as i32).pow(2);
        let strategy = configuration.chunk_despawn_strategy();

        for (chunk, view_visibility) in all_chunks.iter() {
            if chunk_should_retire(
                strategy,
                &cameras,
                chunk.position,
                view_visibility.map(|vis| vis.get()),
                spawning_distance_squared,
                near_distance_squared,
            ) {
                commands
                    .entity(chunk.entity)
                    .try_insert(NeedsDespawn)
                    .remove::<NeedsRemesh>()
                    .remove::<NeedsRemeshLowPriority>();
                ev_chunk_will_despawn
                    .write(ChunkWillDespawn::<C>::new(chunk.position, chunk.entity));
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

        for chunk in dirty_chunks
            .iter()
            .chain(dirty_chunks_low.iter())
            .take(available_threads)
        {
            let previous_chunk_data = {
                let Some(read_lock) = chunk_map.try_get_read_lock() else {
                    return;
                };
                ChunkMap::<C, C::MaterialIndex>::get(&chunk.position, &read_lock)
            };

            let lod_level = chunk.lod_level;

            let regenerate_strategy = configuration.chunk_regenerate_strategy();

            let voxel_data_fn = (configuration.voxel_lookup_delegate())(
                chunk.position,
                lod_level,
                previous_chunk_data.clone(),
            );
            let data_shape = chunk.data_shape;
            let mesh_shape = chunk.mesh_shape;
            let chunk_meshing_fn = (configuration
                .chunk_meshing_delegate()
                .unwrap_or(Box::new(default_chunk_meshing_delegate)))(
                chunk.position,
                lod_level,
                data_shape,
                mesh_shape,
                previous_chunk_data.clone(),
            );
            let texture_index_mapper = configuration.texture_index_mapper().clone();

            let mut chunk_task = ChunkTask::<C, C::MaterialIndex>::new(
                chunk.entity,
                chunk.position,
                lod_level,
                data_shape,
                mesh_shape,
                modified_voxels.clone(),
            );

            let mesh_map = mesh_cache.get_mesh_map();

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
            (
                Entity,
                &mut ChunkThread<C, C::MaterialIndex>,
                &mut Chunk<C>,
                &Transform,
            ),
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

        for (entity, mut thread, chunk, transform) in &mut chunking_threads {
            let thread_result = future::block_on(future::poll_once(&mut thread.0));

            if thread_result.is_none() {
                continue;
            }

            let chunk_task = thread_result.unwrap();

            if chunk_task.is_empty() || chunk_task.is_full() {
                commands
                    .entity(entity)
                    .remove::<Mesh3d>()
                    .remove::<MeshRef>()
                    .remove::<NeedsMaterial<C>>()
                    .remove::<C::ChunkUserBundle>();
            } else {
                let mesh_handle = {
                    if let Some(mesh_handle) =
                        mesh_cache.get_mesh_handle(&chunk_task.voxels_hash())
                    {
                        if let Some(user_bundle) =
                            mesh_cache.get_user_bundle(&chunk_task.voxels_hash())
                        {
                            commands.entity(entity).insert(user_bundle);
                        }

                        mesh_handle
                    } else {
                        let hash = chunk_task.voxels_hash();
                        let Some(mesh) = chunk_task.mesh else {
                            commands
                                .entity(chunk.entity)
                                .try_insert(NeedsRemesh)
                                .remove::<NeedsRemeshLowPriority>()
                                .remove::<ChunkThread<C, C::MaterialIndex>>();
                            continue;
                        };

                        // Bevy 0.19 mesh slab allocator can emit use-after-free
                        // errors if zero-vertex meshes are uploaded.
                        if mesh.count_vertices() == 0 {
                            commands
                                .entity(entity)
                                .remove::<Mesh3d>()
                                .remove::<MeshRef>()
                                .remove::<NeedsMaterial<C>>()
                                .remove::<C::ChunkUserBundle>();
                            chunk_map_update_buffer.push((
                                chunk.position,
                                chunk_task.chunk_data,
                                ChunkWillSpawn::<C>::new(chunk_task.position, entity),
                            ));
                            commands
                                .entity(chunk.entity)
                                .remove::<ChunkThread<C, C::MaterialIndex>>();
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
                            commands.entity(entity).insert(bundle);
                        }
                        mesh_ref
                    }
                };

                commands.entity(entity).try_insert((
                    *transform,
                    MeshRef(mesh_handle),
                    NeedsMaterial::<C>(PhantomData),
                ));
            }

            chunk_map_update_buffer.push((
                chunk.position,
                chunk_task.chunk_data,
                ChunkWillSpawn::<C>::new(chunk_task.position, entity),
            ));

            commands
                .entity(chunk.entity)
                .remove::<ChunkThread<C, C::MaterialIndex>>();
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
        mut needs_material: Query<(Entity, &MeshRef, &Transform), With<NeedsMaterial<C>>>,
        material_handle: Option<Res<VoxelWorldMaterialHandle<M>>>,
    ) {
        let Some(material_handle) = material_handle else {
            return;
        };

        for (entity, mesh_ref, transform) in needs_material.iter_mut() {
            commands
                .entity(entity)
                .insert(Mesh3d((*mesh_ref.0).clone()))
                .insert(MeshMaterial3d(material_handle.handle.clone()))
                .insert(*transform)
                .remove::<NeedsMaterial<C>>();
        }
    }
}

struct TrackedCamera<'a> {
    camera: &'a Camera,
    transform: &'a GlobalTransform,
    frustum: &'a Frustum,
    position: Vec3,
    chunk_position: IVec3,
}

impl<'a> TrackedCamera<'a> {
    fn new(
        camera: &'a Camera,
        transform: &'a GlobalTransform,
        frustum: &'a Frustum,
    ) -> Self {
        let position = transform.translation();
        Self {
            camera,
            transform,
            frustum,
            position,
            chunk_position: position.as_ivec3() / CHUNK_SIZE_I,
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

fn closest_camera_position_to_chunk(
    cameras: &[TrackedCamera],
    chunk_position: IVec3,
) -> Vec3 {
    let chunk_world_position = chunk_position.as_vec3() * CHUNK_SIZE_F;
    cameras
        .iter()
        .min_by(|a, b| {
            a.position
                .distance_squared(chunk_world_position)
                .total_cmp(&b.position.distance_squared(chunk_world_position))
        })
        .map(|camera| camera.position)
        .unwrap_or(Vec3::ZERO)
}

fn chunk_visible_to_any_camera(
    cameras: &[TrackedCamera],
    chunk_position: IVec3,
    ndc_margin: f32,
) -> bool {
    cameras.iter().any(|camera| {
        chunk_visible_to_camera(
            camera.frustum,
            camera.position,
            chunk_position,
            ndc_margin,
        )
    })
}

fn chunk_should_retire(
    strategy: ChunkDespawnStrategy,
    cameras: &[TrackedCamera],
    chunk_position: IVec3,
    view_visible: Option<bool>,
    spawning_distance_squared: i32,
    near_distance_squared: i32,
) -> bool {
    let should_be_culled = match strategy {
        ChunkDespawnStrategy::FarAway => false,
        ChunkDespawnStrategy::FarAwayOrOutOfView => {
            let frustum_culled =
                !chunk_visible_to_any_camera(cameras, chunk_position, 0.0);
            if let Some(visible) = view_visible {
                !visible || frustum_culled
            } else {
                frustum_culled
            }
        }
    };

    let near_camera =
        chunk_is_close_to_any_camera(cameras, chunk_position, near_distance_squared);

    (should_be_culled && !near_camera)
        || !chunk_is_close_to_any_camera(
            cameras,
            chunk_position,
            spawning_distance_squared + 1,
        )
}

const SQRT_3: f32 = 1.732_050_8;
const CHUNK_BOUNDING_SPHERE_RADIUS: f32 = 0.5 * CHUNK_SIZE_F * SQRT_3;

fn chunk_visible_to_camera(
    frustum: &Frustum,
    camera_position: Vec3,
    chunk_position: IVec3,
    ndc_margin: f32,
) -> bool {
    let chunk_min = chunk_position.as_vec3() * CHUNK_SIZE_F;
    let chunk_max = chunk_min + Vec3::splat(CHUNK_SIZE_F);

    if camera_position.x >= chunk_min.x
        && camera_position.x <= chunk_max.x
        && camera_position.y >= chunk_min.y
        && camera_position.y <= chunk_max.y
        && camera_position.z >= chunk_min.z
        && camera_position.z <= chunk_max.z
    {
        return true;
    }

    let chunk_center = (chunk_min + chunk_max) * 0.5;
    let mut radius = CHUNK_BOUNDING_SPHERE_RADIUS;
    if ndc_margin > 0.0 {
        radius += ndc_margin * CHUNK_SIZE_F;
    }

    let sphere = Sphere {
        center: Vec3A::from(chunk_center),
        radius,
    };

    frustum.intersects_sphere(&sphere, true)
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
