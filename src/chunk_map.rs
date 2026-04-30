use std::{
    marker::PhantomData,
    sync::{Arc, RwLock, RwLockReadGuard, TryLockError},
};

use crate::{
    chunk::{self, CHUNK_SIZE_F},
    configuration::VoxelWorldConfig,
    voxel::VOXEL_SIZE,
    voxel_world::ChunkWillSpawn,
};
use bevy::{
    math::{bounding::Aabb3d, Vec3A},
    prelude::*,
};
use hashbrown::HashMap;

#[derive(Deref, DerefMut)]
pub struct ChunkMapData<I> {
    #[deref]
    data: HashMap<IVec3, chunk::ChunkData<I>>,
    min: IVec3,
    max: IVec3,
    axis_counts: [HashMap<i32, usize>; 3],
}

/// Holds a map of all chunks that are currently spawned spawned
/// The chunks also exist as entities that can be queried in the ECS,
/// but having this map in addition allows for faster spatial lookups
#[derive(Resource)]
pub struct ChunkMap<C, I> {
    map: Arc<RwLock<ChunkMapData<I>>>,
    _marker: PhantomData<C>,
}

impl<C: VoxelWorldConfig, I: Copy> ChunkMap<C, I> {
    pub fn get(
        position: &IVec3,
        read_lock: &RwLockReadGuard<ChunkMapData<I>>,
    ) -> Option<chunk::ChunkData<I>> {
        read_lock.data.get(position).cloned()
    }

    pub fn get_ref<'a>(
        position: &IVec3,
        read_lock: &'a RwLockReadGuard<ChunkMapData<I>>,
    ) -> Option<&'a chunk::ChunkData<I>> {
        read_lock.data.get(position)
    }

    pub fn contains_chunk(
        position: &IVec3,
        read_lock: &RwLockReadGuard<ChunkMapData<I>>,
    ) -> bool {
        read_lock.data.contains_key(position)
    }

    /// Get the current bounding box of loaded chunks in this map.
    ///
    /// Expressed in **chunk coordinates**. Bounds are **inclusive**.
    pub fn get_bounds(read_lock: &RwLockReadGuard<ChunkMapData<I>>) -> Aabb3d {
        Aabb3d {
            min: Vec3A::from(read_lock.min.as_vec3()),
            max: Vec3A::from(read_lock.max.as_vec3()),
        }
    }

    /// Get the current bounding box of loaded chunks in this map.
    ///
    /// Expressed in **world units**. Bounds are **inclusive**.
    pub fn get_world_bounds(read_lock: &RwLockReadGuard<ChunkMapData<I>>) -> Aabb3d {
        let mut world_bounds = ChunkMap::<C, I>::get_bounds(read_lock);
        world_bounds.min *= CHUNK_SIZE_F * VOXEL_SIZE;
        world_bounds.max = (world_bounds.max + Vec3A::ONE) * CHUNK_SIZE_F * VOXEL_SIZE;
        world_bounds
    }

    pub fn get_read_lock(&self) -> RwLockReadGuard<'_, ChunkMapData<I>> {
        self.map.read().unwrap()
    }

    pub fn try_get_read_lock(&self) -> Option<RwLockReadGuard<'_, ChunkMapData<I>>> {
        match self.map.try_read() {
            Ok(guard) => Some(guard),
            Err(TryLockError::WouldBlock) => None,
            Err(TryLockError::Poisoned(err)) => {
                panic!("ChunkMap read lock poisoned: {err}");
            }
        }
    }

    pub fn get_map(&self) -> Arc<RwLock<ChunkMapData<I>>> {
        self.map.clone()
    }

    pub(crate) fn apply_buffers(
        &self,
        insert_buffer: &mut ChunkMapInsertBuffer<C, I>,
        update_buffer: &mut ChunkMapUpdateBuffer<C, I>,
        remove_buffer: &mut ChunkMapRemoveBuffer<C>,
        ev_chunk_will_spawn: &mut MessageWriter<ChunkWillSpawn<C>>,
    ) -> bool {
        if insert_buffer.is_empty()
            && update_buffer.is_empty()
            && remove_buffer.is_empty()
        {
            return false;
        }

        if let Ok(mut write_lock) = self.map.try_write() {
            write_lock.data.reserve(insert_buffer.len());

            for (position, chunk_data) in insert_buffer.drain(..) {
                if write_lock.data.insert(position, chunk_data).is_none() {
                    write_lock.include_position_in_bounds(position);
                }
            }

            for (position, chunk_data, evt) in update_buffer.drain(..) {
                if let Some(existing_chunk_data) = write_lock.data.get_mut(&position) {
                    *existing_chunk_data = chunk_data;
                } else {
                    write_lock.data.insert(position, chunk_data);
                    write_lock.include_position_in_bounds(position);
                }

                ev_chunk_will_spawn.write(evt);
            }

            let mut need_rebuild_aabb = false;
            for position in remove_buffer.drain(..) {
                if write_lock.data.remove(&position).is_some() {
                    need_rebuild_aabb |= write_lock.remove_position(position);
                }
            }

            if need_rebuild_aabb {
                write_lock.rebuild_bounds_from_positions();
            }

            need_rebuild_aabb
        } else {
            false
        }
    }
}

impl<I> ChunkMapData<I> {
    fn include_position_in_bounds(&mut self, position: IVec3) {
        let first_position = self.axis_counts[0].is_empty();
        *self.axis_counts[0].entry(position.x).or_default() += 1;
        *self.axis_counts[1].entry(position.y).or_default() += 1;
        *self.axis_counts[2].entry(position.z).or_default() += 1;

        if first_position {
            self.min = position;
            self.max = position;
            return;
        }

        self.min = self.min.min(position);
        self.max = self.max.max(position);
    }

    fn remove_position(&mut self, position: IVec3) -> bool {
        let boundary_removed =
            position.cmpeq(self.min).any() || position.cmpeq(self.max).any();

        Self::decrement_axis_count(&mut self.axis_counts[0], position.x);
        Self::decrement_axis_count(&mut self.axis_counts[1], position.y);
        Self::decrement_axis_count(&mut self.axis_counts[2], position.z);

        boundary_removed
    }

    fn decrement_axis_count(axis_counts: &mut HashMap<i32, usize>, coordinate: i32) {
        let Some(count) = axis_counts.get_mut(&coordinate) else {
            return;
        };
        *count -= 1;
        if *count == 0 {
            axis_counts.remove(&coordinate);
        }
    }

    fn rebuild_bounds_from_positions(&mut self) {
        let Some((min_x, max_x)) = Self::axis_bounds(&self.axis_counts[0]) else {
            self.min = IVec3::ZERO;
            self.max = IVec3::ZERO;
            return;
        };

        let (min_y, max_y) = Self::axis_bounds(&self.axis_counts[1]).unwrap();
        let (min_z, max_z) = Self::axis_bounds(&self.axis_counts[2]).unwrap();

        self.min = IVec3::new(min_x, min_y, min_z);
        self.max = IVec3::new(max_x, max_y, max_z);
    }

    fn axis_bounds(axis_counts: &HashMap<i32, usize>) -> Option<(i32, i32)> {
        let mut coordinates = axis_counts.keys();
        let first = *coordinates.next()?;
        let mut min = first;
        let mut max = first;
        for coordinate in coordinates {
            min = min.min(*coordinate);
            max = max.max(*coordinate);
        }

        Some((min, max))
    }
}

impl<C, I> Default for ChunkMap<C, I> {
    fn default() -> Self {
        Self {
            map: Arc::new(RwLock::new(ChunkMapData {
                data: HashMap::with_capacity(1000),
                min: IVec3::ZERO,
                max: IVec3::ZERO,
                axis_counts: Default::default(),
            })),
            _marker: PhantomData,
        }
    }
}

#[derive(Resource, Deref, DerefMut, Default, Debug)]
pub(crate) struct ChunkMapInsertBuffer<C, I>(
    #[deref] Vec<(IVec3, chunk::ChunkData<I>)>,
    PhantomData<C>,
);

#[derive(Resource, Deref, DerefMut, Default)]
pub(crate) struct ChunkMapUpdateBuffer<C: VoxelWorldConfig, I>(
    #[deref] Vec<(IVec3, chunk::ChunkData<I>, ChunkWillSpawn<C>)>,
    PhantomData<C>,
);

#[derive(Resource, Deref, DerefMut, Default)]
pub(crate) struct ChunkMapRemoveBuffer<C>(#[deref] Vec<IVec3>, PhantomData<C>);
