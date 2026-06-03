mod chunk;
mod chunk_map;
mod configuration;
mod debug_draw;
mod mesh_cache;
mod meshing;
mod plugin;
mod voxel;
mod voxel_material;
mod voxel_traversal;
mod voxel_world;
mod voxel_world_internal;

pub mod prelude {
    pub use crate::chunk::{Chunk, NeedsDespawn};
    pub use crate::configuration::*;
    pub use crate::plugin::VoxelWorldPlugin;
    pub use crate::voxel::{VoxelFace, WorldVoxel, VOXEL_SIZE};
    pub use crate::voxel_world::{
        get_chunk_voxel_position, VoxelRaycastResult, VoxelWorld, VoxelWorldCamera,
    };
    pub use crate::voxel_world::{
        ChunkWillChangeLod, ChunkWillDespawn, ChunkWillRemesh, ChunkWillSpawn,
        ChunkWillUpdate,
    };
    pub use crate::voxel_world_internal::{
        VoxelWorldDiagnostics, VoxelWorldDiagnosticsFrame,
    };
}

pub mod custom_meshing {
    pub use crate::chunk::PaddedChunkShape;
    pub use crate::chunk::VoxelArray;
    pub use crate::chunk::CHUNK_SIZE_F;
    pub use crate::chunk::CHUNK_SIZE_I;
    pub use crate::chunk::CHUNK_SIZE_U;
    pub use crate::meshing::generate_chunk_mesh;
    pub use crate::meshing::generate_chunk_mesh_for_shape;
    pub use crate::meshing::mesh_from_quads;
}

pub mod debug {
    pub use crate::debug_draw::*;
}

pub mod rendering {
    pub use crate::plugin::VoxelWorldMaterialHandle;
    pub use crate::voxel_material::vertex_layout;
    pub use crate::voxel_material::ATTRIBUTE_TEX_INDEX;
    pub use crate::voxel_material::VOXEL_TEXTURE_SHADER_HANDLE;
}

pub mod traversal_alg {
    pub use crate::voxel_traversal::*;
}

#[doc(hidden)]
pub mod benchmark {
    use bevy::prelude::*;

    use crate::{
        chunk::{ChunkTask, CHUNK_SIZE_I},
        configuration::{ChunkRegenerateStrategy, DefaultWorld, VoxelWorldConfig},
        voxel::WorldVoxel,
        voxel_world_internal::ModifiedVoxels,
    };

    #[derive(Clone, Copy)]
    pub enum GenerationPattern {
        Empty,
        Full,
        SingleVoxel,
    }

    #[derive(Clone, Copy)]
    pub struct GenerationBenchResult {
        pub is_empty: bool,
        pub is_full: bool,
        pub voxels_len: usize,
        pub voxels_hash: u64,
    }

    pub fn generate_chunk_for_bench(
        data_shape: UVec3,
        pattern: GenerationPattern,
    ) -> GenerationBenchResult {
        type Mat = <DefaultWorld as VoxelWorldConfig>::MaterialIndex;

        let modified_voxels = ModifiedVoxels::<DefaultWorld, Mat>::default();
        let mut chunk_task = ChunkTask::<DefaultWorld, Mat>::new(
            Entity::PLACEHOLDER,
            IVec3::ZERO,
            0,
            data_shape,
            data_shape,
            modified_voxels,
        );

        chunk_task.generate(
            move |world_pos, _previous| match pattern {
                GenerationPattern::Empty => WorldVoxel::Air,
                GenerationPattern::Full => WorldVoxel::Solid(1),
                GenerationPattern::SingleVoxel => {
                    if world_pos == IVec3::splat(CHUNK_SIZE_I / 2) {
                        WorldVoxel::Solid(1)
                    } else {
                        WorldVoxel::Air
                    }
                }
            },
            None,
            ChunkRegenerateStrategy::Repopulate,
        );

        GenerationBenchResult {
            is_empty: chunk_task.is_empty(),
            is_full: chunk_task.is_full(),
            voxels_len: chunk_task
                .chunk_data
                .voxels
                .as_ref()
                .map(|voxels| voxels.len())
                .unwrap_or_default(),
            voxels_hash: chunk_task.voxels_hash(),
        }
    }
}

#[cfg(test)]
mod test;
