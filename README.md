# bevy_voxel_world

[![Crates.io](https://img.shields.io/crates/v/bevy_voxel_world.svg)](https://crates.io/crates/bevy_voxel_world)
[![License](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](https://github.com/bevyengine/bevy#license)
[![Bevy tracking](https://img.shields.io/badge/Bevy%20tracking-released%20version-lightblue)](https://github.com/bevyengine/bevy/blob/main/docs/plugins_guidelines.md#main-branch-tracking)

---

## What is `bevy_voxel_world`

This plugin makes it easy to generate and modify voxel terrains in Bevy. `bevy_voxel_world` handles multithreaded meshing, chunk spawning/despawning, texture mapping and provides an easy to use API that can be accessed from any system.

![bvw_480](https://github.com/splashdust/bevy_voxel_world/assets/428824/98d25cd1-0a6c-4618-b0de-0e16ca5af636)

```bash
$ cargo run -r --example noise_terrain
```

The world can be controlled in two main ways: through a terrain lookup function, and directly by `set_voxel` and `get_voxel` functions. The world has two "layers" of voxel information, one that is procedural and determined by the terrain lookup function, and one that is controlled by `set_voxel` and persisted in a `HashMap`. The persistent layer always overrides the procedural layer. This way, the world can be infinitely large, but we only need to store information about voxels that are deliberately changed. In the current implementation, the proceduaral layer is cached for spawned chunks, so it may still use a lot of memory if the spawning distance is large.

For an example on how to use a terrain lookup function, see [this example](https://github.com/splashdust/bevy_voxel_world/blob/main/examples/noise_terrain.rs).

## Basic setup

Create a configuration struct for your world:

```rust
#[derive(Resource, Clone, Default)]
struct MyWorld;

impl VoxelWorldConfig for MyWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    // All options have defaults, so you only need to add the ones you want to modify.
    // For a full list, see src/configuration.rs
    fn spawning_distance(&self) -> u32 {
        25
    }
}
```

Then add the plugin with your config:

```rust
.add_plugins(VoxelWorldPlugin::with_config(MyWorld))
```

The config struct does two things:

- It supplies the configuration values
- Its type also acts as a world instance identifier. This means that you can create multiple worlds by adding multiple instances of the plugin as long as each instance has a unique configuration struct. [Here's an example of two worlds using different materials](https://github.com/splashdust/bevy_voxel_world/blob/main/examples/multiple_worlds.rs)

## Accessing the world

To access a voxel world instance in a system, you can use the `VoxelWorld` system param. `VoxelWorld` take one type parameter, which is the configuration struct for the world you want to access.

The `set_voxel` and `get_voxel` access functions can be used to manipulate the voxel data in the world.

```rust
fn my_system(mut voxel_world: VoxelWorld<MyWorld>) {
    voxel_world.set_voxel(IVec3 { ... }, WorldVoxel::Solid(0));
}
```

This will update the voxel value at the given location in the persisting `HashMap`, and cause `bevy_voxel_world` to queue the affected chunk for re-meshing.

Voxels are keyed by their XYZ coordinate in the world, specified by an `IVec3`. The type of voxel is specified by the `WorldVoxel` type. A voxel can be `Unset`, `Air` or `Solid`.

## Voxel materials

`Solid` voxels holds a material type value. You can configure the type of the material index, but it's advisable to keep it small (like `u8`), since each voxel will hold one of these values. Material types can easily be mapped to indexes in a 2d texture array though a mapping callback. The mapping callback always returns a `[u32; 3]` which is passed along to the shader.

A custom array texture can be supplied in the config. It should be image with a size of `W x (W * n)`, where `n` is the number of indexes. So an array of 4 16x16 px textures would be 16x64 px in size. The number of indexes is specified in the second parameter.

Then, to map out which indexes belong to which material type, you can supply a `texture_index_mapper` callback:

```rust
impl VoxelWorldConfig for MyWorld {
    // In this example we use a `u8` for the index.
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    fn texture_index_mapper(&self) -> Arc<dyn Fn(u8) -> [u32; 3] + Send + Sync> {
        Arc::new(|vox_mat: u8| match vox_mat {
            SNOWY_BRICK => [0, 1, 2],
            FULL_BRICK => [2, 2, 2],
            GRASS | _ => [3, 3, 3],
        })
    }

    fn voxel_texture(&self) -> Option<(String, u32)> {
        Some(("example_voxel_texture.png".into(), 4)) // Array texture with 4 indexes
    }
}
```

The `texture_index_mapper` callback is supplied with a material type and should return an array with three values. The values indicate which texture index maps to `[top, sides, bottom]` of a voxel.

See the [textures example](https://github.com/splashdust/bevy_voxel_world/blob/main/examples/textures.rs) for a runnable example of this.

<img width="558" alt src="https://github.com/splashdust/bevy_voxel_world/assets/428824/382fdcf7-9d70-4432-b2ba-18479d34346f">

### Custom shader support

If you need to customize materials futher, you can use `.with_material(MyCustomVoxelMaterial)`, when adding the plugin, to register your own Bevy material. This allows you to use your own custom shader with `bevy_voxel_world`. See [this example](https://github.com/splashdust/bevy_voxel_world/blob/main/examples/custom_material.rs) for more details.

## Ray casting

To find a voxel location in the world from a pixel location on the screen, for example the mouse location, you can ray cast into the voxel world.

```rust
fn do_something_with_mouse_voxel_pos(
    voxel_world: VoxelWorld<MyWorld>,
    camera_info: Query<(&Camera, &GlobalTransform), With<VoxelWorldCamera<MyWorld>>>,
    mut cursor_evr: EventReader<CursorMoved>,
) {
    for ev in cursor_evr.read() {
        // Get a ray from the cursor position into the world
        let (camera, cam_gtf) = camera_info.single();
        let Some(ray) = camera.viewport_to_world(cam_gtf, ev.position) else {
            return;
        };

        if let Some(result) = voxel_world.raycast(ray, &|(_pos, _vox)| true) {
            // result.position will be the world location of the voxel as a Vec3
            // To get the empty location next to the voxel in the direction of the surface where the ray intersected you can use result.normal:
            // let empty_pos = result.position + result.normal;
        }
    }
}
```

See this [full example of ray casting](https://github.com/splashdust/bevy_voxel_world/blob/main/examples/ray_cast.rs) for more details.

## Custom meshing

Custom meshing is supported though a delegate function that can be specified in the config. The function recives an array of voxel data for the chunk to be meshed, and a reference to the texture index mapper functon. This delegate is then called from a thread in the task pool and needs to return a Bevy Mesh that can be inserted on the chunk entity.

The meshing delegate function can also return a Bevy component or bundle that will be inserted along with the mesh. This is useful for generating additional data based on the meshing data, such as nav meshes. It can be beneficial to do all such processing at once, while the voxel data is already hot in the caches.

See this [full example of custom meshing](https://github.com/splashdust/bevy_voxel_world/blob/main/examples/custom_meshing.rs) for more details.

<img width="558" src="https://github.com/user-attachments/assets/13f46fd6-81a8-4c93-943c-9c5fc7b9b38e"/>

## Gotchas

`bevy_voxel_world` began as an internal part of a game that I'm working on, but I figured that it could be useful as a standalone plugin, for myself and perhaps for others, so I decided to break it out and make it public as a crate.

In its current state, there are still various hard-coded assumptions that works well enough for my usecase, but may not suit everyone. Over time, the aim is to generalize and make `bevy_voxel_world` more configurable. There are also many potential performance optimizations that I have not prioritized yet at this point.

Default meshing is handled by [block-mesh-rs](https://github.com/bonsairobo/block-mesh-rs), using the "simple" algorithm.

Feedback, issues and pull requests are welcomed!

---

### Bevy compatibility

| bevy | bevy_voxel_world |
| ---- | ---------------- |
| 0.17 | ^0.13.0          |
| 0.16 | ^0.12.0          |
| 0.15 | 0.10.0           |
| 0.14 | 0.8.0            |
| 0.13 | 0.4.0            |
| 0.12 | 0.3.6            |
| 0.11 | 0.2.2            |
