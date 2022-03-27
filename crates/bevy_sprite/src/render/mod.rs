use std::borrow::BorrowMut;
use std::collections::BTreeMap;
use std::hash::{BuildHasher, Hash, Hasher};
use std::time::Instant;
use ahash::AHasher;

use crate::{
    texture_atlas::{TextureAtlas, TextureAtlasSprite},
    Rect, Sprite, SPRITE_SHADER_HANDLE,
};
use bevy_asset::{AssetEvent, Assets, Handle, HandleId};
use bevy_core::FloatOrd;
use bevy_core_pipeline::Transparent2d;
use bevy_ecs::{
    prelude::*,
    system::{lifetimeless::*, SystemParamItem},
};
use bevy_math::{const_vec2, Vec2};
use bevy_render::{
    color::Color,
    render_asset::RenderAssets,
    render_phase::{
        BatchedPhaseItem, DrawFunctions, EntityRenderCommand, RenderCommand, RenderCommandResult,
        RenderPhase, SetItemPipeline, TrackedRenderPass,
    },
    render_resource::{std140::AsStd140, *},
    renderer::{RenderDevice, RenderQueue},
    texture::{BevyDefault, Image},
    view::{Msaa, ViewUniform, ViewUniformOffset, ViewUniforms, Visibility},
    RenderWorld,
};
use bevy_transform::components::GlobalTransform;
use bevy_utils::{FullPassHasher, FullPreHashMap, hashbrown, Hashed, HashMap, PassHasher, PreHashMap};
use bytemuck::{Pod, Zeroable};
use copyless::VecHelper;
use partition::{partition, partition_index};
use rdst::{RadixKey, RadixSort};
use bevy_render::render_phase::BatchRange;

pub struct SpritePipeline {
    view_layout: BindGroupLayout,
    material_layout: BindGroupLayout,
}

impl FromWorld for SpritePipeline {
    fn from_world(world: &mut World) -> Self {
        let world = world.cell();
        let render_device = world.get_resource::<RenderDevice>().unwrap();

        let view_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: BufferSize::new(ViewUniform::std140_size_static() as u64),
                },
                count: None,
            }],
            label: Some("sprite_view_layout"),
        });

        let material_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        multisampled: false,
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
            label: Some("sprite_material_layout"),
        });

        SpritePipeline {
            view_layout,
            material_layout,
        }
    }
}

bitflags::bitflags! {
    #[repr(transparent)]
    // NOTE: Apparently quadro drivers support up to 64x MSAA.
    // MSAA uses the highest 6 bits for the MSAA sample count - 1 to support up to 64x MSAA.
    pub struct SpritePipelineKey: u32 {
        const NONE                        = 0;
        const COLORED                     = (1 << 0);
        const MSAA_RESERVED_BITS          = SpritePipelineKey::MSAA_MASK_BITS << SpritePipelineKey::MSAA_SHIFT_BITS;
    }
}

impl SpritePipelineKey {
    const MSAA_MASK_BITS: u32 = 0b111111;
    const MSAA_SHIFT_BITS: u32 = 32 - 6;

    pub fn from_msaa_samples(msaa_samples: u32) -> Self {
        let msaa_bits = ((msaa_samples - 1) & Self::MSAA_MASK_BITS) << Self::MSAA_SHIFT_BITS;
        SpritePipelineKey::from_bits(msaa_bits).unwrap()
    }

    pub fn msaa_samples(&self) -> u32 {
        ((self.bits >> Self::MSAA_SHIFT_BITS) & Self::MSAA_MASK_BITS) + 1
    }
}

impl SpecializedRenderPipeline for SpritePipeline {
    type Key = SpritePipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let mut formats = vec![
            // position
            VertexFormat::Float32x3,
            // uv
            VertexFormat::Float32x2,
        ];

        if key.contains(SpritePipelineKey::COLORED) {
            // color
            formats.push(VertexFormat::Uint32);
        }

        let vertex_layout =
            VertexBufferLayout::from_vertex_formats(VertexStepMode::Vertex, formats);

        let mut shader_defs = Vec::new();
        if key.contains(SpritePipelineKey::COLORED) {
            shader_defs.push("COLORED".to_string());
        }

        RenderPipelineDescriptor {
            vertex: VertexState {
                shader: SPRITE_SHADER_HANDLE.typed::<Shader>(),
                entry_point: "vertex".into(),
                shader_defs: shader_defs.clone(),
                buffers: vec![vertex_layout],
            },
            fragment: Some(FragmentState {
                shader: SPRITE_SHADER_HANDLE.typed::<Shader>(),
                shader_defs,
                entry_point: "fragment".into(),
                targets: vec![ColorTargetState {
                    format: TextureFormat::bevy_default(),
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                }],
            }),
            layout: Some(vec![self.view_layout.clone(), self.material_layout.clone()]),
            primitive: PrimitiveState {
                front_face: FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
            },
            depth_stencil: None,
            multisample: MultisampleState {
                count: key.msaa_samples(),
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            label: Some("sprite_pipeline".into()),
        }
    }
}

#[derive(Component, Clone, Copy)]
pub struct ExtractedSprite {
    pub transform: GlobalTransform,
    pub color: Color,
    /// Select an area of the texture
    pub rect: Option<Rect>,
    /// Change the on-screen size of the sprite
    pub custom_size: Option<Vec2>,
    /// Handle to the `Image` of this sprite
    /// PERF: storing a `HandleId` instead of `Handle<Image>` enables some optimizations (`ExtractedSprite` becomes `Copy` and doesn't need to be dropped)
    pub image_handle_id: HandleId,
    pub flip_x: bool,
    pub flip_y: bool,
}

impl RadixKey for ExtractedSprite {
    const LEVELS: usize = 4;

    fn get_level(&self, level: usize) -> u8 {
        self.transform.translation.z.get_level(level)
    }
}

#[derive(Default)]
pub struct ExtractedSprites {
    pub sprites: Vec<ExtractedSprite>,
}

#[derive(Default)]
pub struct SpriteAssetEvents {
    pub images: Vec<AssetEvent<Image>>,
}

pub fn extract_sprite_events(
    mut render_world: ResMut<RenderWorld>,
    mut image_events: EventReader<AssetEvent<Image>>,
) {
    let mut events = render_world.resource_mut::<SpriteAssetEvents>();
    let SpriteAssetEvents { ref mut images } = *events;
    images.clear();

    for image in image_events.iter() {
        // AssetEvent: !Clone
        images.push(match image {
            AssetEvent::Created { handle } => AssetEvent::Created {
                handle: handle.clone_weak(),
            },
            AssetEvent::Modified { handle } => AssetEvent::Modified {
                handle: handle.clone_weak(),
            },
            AssetEvent::Removed { handle } => AssetEvent::Removed {
                handle: handle.clone_weak(),
            },
        });
    }
}

pub fn extract_sprites(
    mut render_world: ResMut<RenderWorld>,
    texture_atlases: Res<Assets<TextureAtlas>>,
    sprite_query: Query<(&Visibility, &Sprite, &GlobalTransform, &Handle<Image>)>,
    atlas_query: Query<(
        &Visibility,
        &TextureAtlasSprite,
        &GlobalTransform,
        &Handle<TextureAtlas>,
    )>,
) {
    let mut extracted_sprites = render_world.resource_mut::<ExtractedSprites>();
    extracted_sprites.sprites.clear();
    for (visibility, sprite, transform, handle) in sprite_query.iter() {
        if !visibility.is_visible {
            continue;
        }
        // PERF: we don't check in this function that the `Image` asset is ready, since it should be in most cases and hashing the handle is expensive
        extracted_sprites.sprites.alloc().init(ExtractedSprite {
            color: sprite.color,
            transform: *transform,
            // Use the full texture
            rect: None,
            // Pass the custom size
            custom_size: sprite.custom_size,
            flip_x: sprite.flip_x,
            flip_y: sprite.flip_y,
            image_handle_id: handle.id,
        });
    }
    for (visibility, atlas_sprite, transform, texture_atlas_handle) in atlas_query.iter() {
        if !visibility.is_visible {
            continue;
        }
        if let Some(texture_atlas) = texture_atlases.get(texture_atlas_handle) {
            let rect = Some(texture_atlas.textures[atlas_sprite.index as usize]);
            extracted_sprites.sprites.alloc().init(ExtractedSprite {
                color: atlas_sprite.color,
                transform: *transform,
                // Select the area in the texture atlas
                rect,
                // Pass the custom size
                custom_size: atlas_sprite.custom_size,
                flip_x: atlas_sprite.flip_x,
                flip_y: atlas_sprite.flip_y,
                image_handle_id: texture_atlas.texture.id,
            });
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SpriteVertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ColoredSpriteVertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
    pub color: u32,
}

pub struct SpriteMeta {
    vertices: BufferVec<SpriteVertex>,
    colored_vertices: BufferVec<ColoredSpriteVertex>,
    view_bind_group: Option<BindGroup>,
}

impl SpriteMeta {
    fn buffers_mut(&mut self) -> (&mut BufferVec<SpriteVertex>, &mut BufferVec<ColoredSpriteVertex>) {
        (
            &mut self.vertices,
            &mut self.colored_vertices,
        )
    }
}

impl Default for SpriteMeta {
    fn default() -> Self {
        Self {
            vertices: BufferVec::new(BufferUsages::VERTEX),
            colored_vertices: BufferVec::new(BufferUsages::VERTEX),
            view_bind_group: None,
        }
    }
}

const QUAD_INDICES: [usize; 6] = [0, 2, 3, 0, 1, 2];

const QUAD_VERTEX_POSITIONS: [Vec2; 4] = [
    const_vec2!([-0.5, -0.5]),
    const_vec2!([0.5, -0.5]),
    const_vec2!([0.5, 0.5]),
    const_vec2!([-0.5, 0.5]),
];

const QUAD_UVS: [Vec2; 4] = [
    const_vec2!([0., 1.]),
    const_vec2!([1., 1.]),
    const_vec2!([1., 0.]),
    const_vec2!([0., 0.]),
];

#[derive(Component, Eq, PartialEq, Copy, Clone, Hash)]
pub struct SpriteBatch {
    image_handle_id: HandleId,
    colored: bool,
}

#[derive(Default)]
pub struct ImageBindGroups {
    values: HashMap<Handle<Image>, BindGroup>,
}

#[derive(Clone, Copy)]
struct ExtractedSpriteWithKey<'a>(u128, &'a ExtractedSprite);

impl RadixKey for ExtractedSpriteWithKey<'_> {
    const LEVELS: usize = 16;

    fn get_level(&self, level: usize) -> u8 {
        self.0.get_level(level)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn queue_sprites(
    mut commands: Commands,
    draw_functions: Res<DrawFunctions<Transparent2d>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut sprite_meta: ResMut<SpriteMeta>,
    view_uniforms: Res<ViewUniforms>,
    sprite_pipeline: Res<SpritePipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<SpritePipeline>>,
    mut pipeline_cache: ResMut<PipelineCache>,
    mut image_bind_groups: ResMut<ImageBindGroups>,
    gpu_images: Res<RenderAssets<Image>>,
    msaa: Res<Msaa>,
    mut extracted_sprites: ResMut<ExtractedSprites>,
    mut views: Query<&mut RenderPhase<Transparent2d>>,
    events: Res<SpriteAssetEvents>,
) {
    // If an image has changed, the GpuImage has (probably) changed
    for event in &events.images {
        match event {
            AssetEvent::Created { .. } => None,
            AssetEvent::Modified { handle } => image_bind_groups.values.remove(handle),
            AssetEvent::Removed { handle } => image_bind_groups.values.remove(handle),
        };
    }

    if let Some(view_binding) = view_uniforms.uniforms.binding() {
        let sprite_meta = &mut sprite_meta;

        // Clear the vertex buffers
        sprite_meta.vertices.clear();
        sprite_meta.colored_vertices.clear();

        sprite_meta.view_bind_group = Some(render_device.create_bind_group(&BindGroupDescriptor {
            entries: &[BindGroupEntry {
                binding: 0,
                resource: view_binding,
            }],
            label: Some("sprite_view_bind_group"),
            layout: &sprite_pipeline.view_layout,
        }));

        let draw_sprite_function = draw_functions.read().get_id::<DrawSprite>().unwrap();
        let key = SpritePipelineKey::from_msaa_samples(msaa.samples);
        let pipeline = pipelines.specialize(&mut pipeline_cache, &sprite_pipeline, key);
        let colored_pipeline = pipelines.specialize(
            &mut pipeline_cache,
            &sprite_pipeline,
            key | SpritePipelineKey::COLORED,
        );

        // Vertex buffer indices
        let mut index = 0;
        let mut colored_index = 0;

        // FIXME: VisibleEntities is ignored
        for mut transparent_phase in views.iter_mut() {
            let extracted_sprites = &mut extracted_sprites.sprites;
            let image_bind_groups = &mut *image_bind_groups;

            let start = Instant::now();
            // Sort sprites by z for correct transparency and then by handle to improve batching

            let mut batch_key_map: FullPreHashMap<SpriteBatch> = hashbrown::HashMap::with_capacity_and_hasher(extracted_sprites.len(), FullPassHasher::default());
            let hash = Instant::now();
            let batch_keys: Vec<_> = extracted_sprites
                .iter()
                .map(|extracted_sprite| {
                    let batch = SpriteBatch {
                        image_handle_id: extracted_sprite.image_handle_id,
                        colored: extracted_sprite.color != Color::WHITE,
                    };

                    let mut hasher = AHasher::default();

                    hasher.write_u32(extracted_sprite.transform.translation.z.to_bits());
                    batch.hash(&mut hasher);

                    let batch_key = hasher.finish();

                    batch_key_map.insert(batch_key, batch);

                    batch_key
                })
                .collect();
            println!("Hash: {}us", hash.elapsed().as_micros());

            let mut batch_entities: FullPreHashMap<(_, _)> = hashbrown::HashMap::with_capacity_and_hasher(extracted_sprites.len(), FullPassHasher::default());

            let spawn_batches = Instant::now();
            batch_key_map
                .into_iter()
                .for_each(|(k, batch)| {
                    gpu_images
                        .get(&Handle::weak(batch.image_handle_id))
                        .map(|gpu_image| {
                            let size = Vec2::new(gpu_image.size.width, gpu_image.size.height);
                            let entity = commands.spawn_bundle((batch,)).id();

                            image_bind_groups
                                .values
                                .entry(Handle::weak(batch.image_handle_id))
                                .or_insert_with(|| {
                                    render_device.create_bind_group(&BindGroupDescriptor {
                                        entries: &[
                                            BindGroupEntry {
                                                binding: 0,
                                                resource: BindingResource::TextureView(
                                                    &gpu_image.texture_view,
                                                ),
                                            },
                                            BindGroupEntry {
                                                binding: 1,
                                                resource: BindingResource::Sampler(&gpu_image.sampler),
                                            },
                                        ],
                                        label: Some("sprite_material_bind_group"),
                                        layout: &sprite_pipeline.material_layout,
                                    })
                                });

                            (entity, size)
                        })
                        .map(|img| {
                            batch_entities.insert(k, img);
                        });
                });

            println!("Spawn batches: {}us", spawn_batches.elapsed().as_micros());

            let uv_pos = Instant::now();

            // Extract UVs and positions for each sprite
            let mut new_extracted_sprites = Vec::with_capacity(extracted_sprites.len());
            extracted_sprites
                .into_iter()
                .zip(batch_keys.into_iter())
                .for_each(|(extracted_sprite, batch_key)| {
                    let _ = batch_entities
                        .get(&batch_key)
                        .map(|(_, image_size)| {
                            let mut uvs = QUAD_UVS;
                            if extracted_sprite.flip_x {
                                uvs = [uvs[1], uvs[0], uvs[3], uvs[2]];
                            }
                            if extracted_sprite.flip_y {
                                uvs = [uvs[3], uvs[2], uvs[1], uvs[0]];
                            }

                            let quad_size = if let Some(custom_size) = extracted_sprite.custom_size {
                                // Override the size if a custom one is specified
                                custom_size
                            } else if let Some(rect) = extracted_sprite.rect {
                                // If a rect is specified, adjust UVs and the size of the quad
                                let rect_size = rect.size();
                                for uv in &mut uvs {
                                    *uv = (rect.min + *uv * rect_size) / *image_size;
                                }
                                rect_size
                            } else {
                                // By default, the size of the quad is the size of the texture
                                *image_size
                            };

                            // Apply size and global transform
                            let positions = QUAD_VERTEX_POSITIONS.map(|quad_pos| {
                                extracted_sprite
                                    .transform
                                    .mul_vec3((quad_pos * quad_size).extend(0.))
                                    .into()
                            });

                            let z = extracted_sprite.transform.translation.z;

                            new_extracted_sprites.push((batch_key, z, extracted_sprite.color, uvs, positions));
                        });
                });

            println!("UV / Pos: {}us", uv_pos.elapsed().as_micros());

            transparent_phase.items.reserve(new_extracted_sprites.len());

            let part = Instant::now();
            // Partition sprites into colored / non-colored vertices
            let (color, no_color) = partition(&mut new_extracted_sprites, |(_, _, color, _, _)| *color != Color::WHITE);
            println!("Partition: {}us", part.elapsed().as_micros());

            let (verts, color_verts) = sprite_meta.buffers_mut();

            // Submit colored vertices
            let submit_color = Instant::now();
            let color_iter = color
                .into_iter()
                .map(|(batch_key, z, color, uvs, positions)| {
                    let color = color.as_linear_rgba_f32();
                    // encode color as a single u32 to save space
                    let color = (color[0] * 255.0) as u32
                        | ((color[1] * 255.0) as u32) << 8
                        | ((color[2] * 255.0) as u32) << 16
                        | ((color[3] * 255.0) as u32) << 24;
                    for i in QUAD_INDICES.iter() {
                        color_verts.push(ColoredSpriteVertex {
                            position: positions[*i],
                            uv: uvs[*i].into(),
                            color,
                        });
                    }
                    let item_start = colored_index;
                    colored_index += QUAD_INDICES.len() as u32;
                    let item_end = colored_index;

                    (batch_key, z, colored_pipeline, item_start, item_end)
                });

            println!("Submit color: {}us", submit_color.elapsed().as_micros());

            // Submit non-colored vertices
            let submit_no_color = Instant::now();
            let no_color_iter = no_color
                .into_iter()
                .map(|(batch_key, z, _, uvs, positions)| {
                    for i in QUAD_INDICES.iter() {
                        verts.push(SpriteVertex {
                            position: positions[*i],
                            uv: uvs[*i].into(),
                        });
                    }
                    let item_start = index;
                    index += QUAD_INDICES.len() as u32;
                    let item_end = index;

                    (batch_key, z, pipeline, item_start, item_end)
                });
            println!("Submit no color: {}us", submit_no_color.elapsed().as_micros());

            let phase_add = Instant::now();
            no_color_iter
                .chain(color_iter)
                .for_each(|(batch_key, z, pipeline, item_start, item_end)| {
                    let sort_key = *z;

                    if let Some((entity, _)) = batch_entities.remove(batch_key) {
                        // TODO(nathan): Not easy to reserve space in transparent_phase for
                        // entities which have an image ready.
                        transparent_phase.add(Transparent2d {
                            draw_function: draw_sprite_function,
                            pipeline,
                            entity,
                            sort_key,
                            batch_range: Some(BatchRange::new(item_start, item_end)),
                        });
                    }
                });
            println!("Phase add: {}us", phase_add.elapsed().as_micros());

            // extracted_sprites.sort_unstable_by_key(|s| s.transform.translation.z);
            // extracted_sprites.radix_sort_unstable();

            println!("ELP: {}us", start.elapsed().as_micros());
            println!("-------");
        }
        sprite_meta
            .vertices
            .write_buffer(&render_device, &render_queue);
        sprite_meta
            .colored_vertices
            .write_buffer(&render_device, &render_queue);
    }
}

pub type DrawSprite = (
    SetItemPipeline,
    SetSpriteViewBindGroup<0>,
    SetSpriteTextureBindGroup<1>,
    DrawSpriteBatch,
);

pub struct SetSpriteViewBindGroup<const I: usize>;
impl<const I: usize> EntityRenderCommand for SetSpriteViewBindGroup<I> {
    type Param = (SRes<SpriteMeta>, SQuery<Read<ViewUniformOffset>>);

    fn render<'w>(
        view: Entity,
        _item: Entity,
        (sprite_meta, view_query): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let view_uniform = view_query.get(view).unwrap();
        pass.set_bind_group(
            I,
            sprite_meta.into_inner().view_bind_group.as_ref().unwrap(),
            &[view_uniform.offset],
        );
        RenderCommandResult::Success
    }
}
pub struct SetSpriteTextureBindGroup<const I: usize>;
impl<const I: usize> EntityRenderCommand for SetSpriteTextureBindGroup<I> {
    type Param = (SRes<ImageBindGroups>, SQuery<Read<SpriteBatch>>);

    fn render<'w>(
        _view: Entity,
        item: Entity,
        (image_bind_groups, query_batch): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let sprite_batch = query_batch.get(item).unwrap();
        let image_bind_groups = image_bind_groups.into_inner();

        pass.set_bind_group(
            I,
            image_bind_groups
                .values
                .get(&Handle::weak(sprite_batch.image_handle_id))
                .unwrap(),
            &[],
        );
        RenderCommandResult::Success
    }
}

pub struct DrawSpriteBatch;
impl<P: BatchedPhaseItem> RenderCommand<P> for DrawSpriteBatch {
    type Param = (SRes<SpriteMeta>, SQuery<Read<SpriteBatch>>);

    fn render<'w>(
        _view: Entity,
        item: &P,
        (sprite_meta, query_batch): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let sprite_batch = query_batch.get(item.entity()).unwrap();
        let sprite_meta = sprite_meta.into_inner();
        if sprite_batch.colored {
            pass.set_vertex_buffer(0, sprite_meta.colored_vertices.buffer().unwrap().slice(..));
        } else {
            pass.set_vertex_buffer(0, sprite_meta.vertices.buffer().unwrap().slice(..));
        }
        pass.draw(item.batch_range().as_ref().unwrap().as_range(), 0..1);
        RenderCommandResult::Success
    }
}
