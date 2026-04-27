use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::deformation::{
    DeformationNetwork, PackedDeformationNetwork, PackedMlpLayerDesc, PackedPlaneDesc,
};
use crate::texture::Texture;
use crate::utils::transmute_slice;

pub const WORKGROUP_SIZE: u32 = 128;
pub const MAX_GRID_FEATURES: u32 = 512;
pub const MAX_MLP_WIDTH: u32 = 512;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuPlaneDesc {
    width: u32,
    height: u32,
    channels: u32,
    data_offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuLayerDesc {
    in_features: u32,
    out_features: u32,
    has_relu_before: u32,
    weight_offset: u32,
    bias_offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct NetworkMetaUniform {
    counts0: [u32; 4],
    counts1: [u32; 4],
    aabb_max: [f32; 4],
    aabb_min: [f32; 4],
    scale_and_pad: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AnimationUniform {
    time01: f32,
    splat_count: u32,
    gaussian_tex_width: u32,
    _pad0: u32,
}

pub struct GpuDeformationRuntime {
    compute_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    animation_uniform_buffer: wgpu::Buffer,
    output_texture: Texture,
    splat_count: u32,
    gaussian_tex_width: u32,
}

impl GpuDeformationRuntime {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        net: &DeformationNetwork,
        splat_count: usize,
        gaussian_tex_width: u32,
        gaussian_tex_height: u32,
        base_tex_data: &[u32],
        base_tile_means: &[[f32; 3]],
        base_scales: &[[f32; 3]],
        orig_means: &[[f32; 3]],
        orig_quats: &[[f32; 4]],
    ) -> Result<Self, String> {
        let packed = net
            .pack_for_gpu()
            .map_err(|e| format!("pack_for_gpu() failed: {}", e))?;
        Self::validate_packed_dims(&packed)?;

        if splat_count != base_tile_means.len()
            || splat_count != base_scales.len()
            || splat_count != orig_means.len()
            || splat_count != orig_quats.len()
        {
            return Err(format!(
                "splat input length mismatch: splats={}, base_tile_means={}, base_scales={}, orig_means={}, orig_quats={}",
                splat_count,
                base_tile_means.len(),
                base_scales.len(),
                orig_means.len(),
                orig_quats.len()
            ));
        }
        if base_tex_data.len() < 8 * splat_count {
            return Err(format!(
                "base_tex_data too short: len={} expected_at_least={}",
                base_tex_data.len(),
                8 * splat_count
            ));
        }

        let meta_uniform = NetworkMetaUniform {
            counts0: [
                packed.metadata.n_grid_levels as u32,
                packed.metadata.n_planes_per_level as u32,
                packed.metadata.feature_dim as u32,
                packed.metadata.net_width as u32,
            ],
            counts1: [
                packed.metadata.n_time_frames as u32,
                packed.feature_layers.len() as u32,
                packed.pos_layers.len() as u32,
                packed.rot_layers.len() as u32,
            ],
            aabb_max: [
                packed.metadata.aabb_max[0],
                packed.metadata.aabb_max[1],
                packed.metadata.aabb_max[2],
                0.0,
            ],
            aabb_min: [
                packed.metadata.aabb_min[0],
                packed.metadata.aabb_min[1],
                packed.metadata.aabb_min[2],
                0.0,
            ],
            scale_and_pad: [packed.metadata.scale_factor, 0.0, 0.0, 0.0],
        };
        let animation_uniform = AnimationUniform {
            time01: 0.0,
            splat_count: splat_count as u32,
            gaussian_tex_width,
            _pad0: 0,
        };

        let orig_means_vec4 = Self::to_vec4_from_vec3(orig_means);
        let base_tile_means_vec4 = Self::to_vec4_from_vec3(base_tile_means);
        let base_scales_vec4 = Self::to_vec4_from_vec3(base_scales);
        let base_rgba = Self::extract_base_rgba(base_tex_data, splat_count);
        let plane_descs = Self::pack_plane_descs(packed.plane_descs.as_slice());
        let feature_layers = Self::pack_layer_descs(packed.feature_layers.as_slice());
        let pos_layers = Self::pack_layer_descs(packed.pos_layers.as_slice());
        let rot_layers = Self::pack_layer_descs(packed.rot_layers.as_slice());

        let network_meta_buffer = Self::create_buffer_init_or_dummy(
            device,
            "deformation_meta_uniform",
            bytemuck::bytes_of(&meta_uniform),
            wgpu::BufferUsages::UNIFORM,
        );
        let animation_uniform_buffer = Self::create_buffer_init_or_dummy(
            device,
            "deformation_animation_uniform",
            bytemuck::bytes_of(&animation_uniform),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let orig_means_buffer = Self::create_buffer_init_or_dummy(
            device,
            "deformation_orig_means",
            bytemuck::cast_slice(orig_means_vec4.as_slice()),
            wgpu::BufferUsages::STORAGE,
        );
        let orig_quats_buffer = Self::create_buffer_init_or_dummy(
            device,
            "deformation_orig_quats",
            bytemuck::cast_slice(orig_quats),
            wgpu::BufferUsages::STORAGE,
        );
        let base_tile_means_buffer = Self::create_buffer_init_or_dummy(
            device,
            "deformation_base_tile_means",
            bytemuck::cast_slice(base_tile_means_vec4.as_slice()),
            wgpu::BufferUsages::STORAGE,
        );
        let base_scales_buffer = Self::create_buffer_init_or_dummy(
            device,
            "deformation_base_scales",
            bytemuck::cast_slice(base_scales_vec4.as_slice()),
            wgpu::BufferUsages::STORAGE,
        );
        let base_rgba_buffer = Self::create_buffer_init_or_dummy(
            device,
            "deformation_base_rgba",
            bytemuck::cast_slice(base_rgba.as_slice()),
            wgpu::BufferUsages::STORAGE,
        );
        let plane_descs_buffer = Self::create_buffer_init_or_dummy(
            device,
            "deformation_plane_descs",
            bytemuck::cast_slice(plane_descs.as_slice()),
            wgpu::BufferUsages::STORAGE,
        );
        let plane_data_buffer = Self::create_buffer_init_or_dummy(
            device,
            "deformation_plane_data",
            bytemuck::cast_slice(packed.plane_data.as_slice()),
            wgpu::BufferUsages::STORAGE,
        );
        let feature_layers_buffer = Self::create_buffer_init_or_dummy(
            device,
            "deformation_feature_layers",
            bytemuck::cast_slice(feature_layers.as_slice()),
            wgpu::BufferUsages::STORAGE,
        );
        let feature_weights_buffer = Self::create_buffer_init_or_dummy(
            device,
            "deformation_feature_weights",
            bytemuck::cast_slice(packed.feature_weights.as_slice()),
            wgpu::BufferUsages::STORAGE,
        );
        let feature_bias_buffer = Self::create_buffer_init_or_dummy(
            device,
            "deformation_feature_bias",
            bytemuck::cast_slice(packed.feature_bias.as_slice()),
            wgpu::BufferUsages::STORAGE,
        );
        let pos_layers_buffer = Self::create_buffer_init_or_dummy(
            device,
            "deformation_pos_layers",
            bytemuck::cast_slice(pos_layers.as_slice()),
            wgpu::BufferUsages::STORAGE,
        );
        let pos_weights_buffer = Self::create_buffer_init_or_dummy(
            device,
            "deformation_pos_weights",
            bytemuck::cast_slice(packed.pos_weights.as_slice()),
            wgpu::BufferUsages::STORAGE,
        );
        let pos_bias_buffer = Self::create_buffer_init_or_dummy(
            device,
            "deformation_pos_bias",
            bytemuck::cast_slice(packed.pos_bias.as_slice()),
            wgpu::BufferUsages::STORAGE,
        );
        let rot_layers_buffer = Self::create_buffer_init_or_dummy(
            device,
            "deformation_rot_layers",
            bytemuck::cast_slice(rot_layers.as_slice()),
            wgpu::BufferUsages::STORAGE,
        );
        let rot_weights_buffer = Self::create_buffer_init_or_dummy(
            device,
            "deformation_rot_weights",
            bytemuck::cast_slice(packed.rot_weights.as_slice()),
            wgpu::BufferUsages::STORAGE,
        );
        let rot_bias_buffer = Self::create_buffer_init_or_dummy(
            device,
            "deformation_rot_bias",
            bytemuck::cast_slice(packed.rot_bias.as_slice()),
            wgpu::BufferUsages::STORAGE,
        );

        let output_texture = Self::create_output_texture(
            device,
            queue,
            gaussian_tex_width,
            gaussian_tex_height,
            base_tex_data,
        );

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("deformation_compute_bind_group_layout"),
            entries: &[
                Self::uniform_entry(0),
                Self::uniform_entry(1),
                Self::storage_entry(2),
                Self::storage_entry(3),
                Self::storage_entry(4),
                Self::storage_entry(5),
                Self::storage_entry(6),
                Self::storage_entry(7),
                Self::storage_entry(8),
                Self::storage_entry(9),
                Self::storage_entry(10),
                Self::storage_entry(11),
                Self::storage_entry(12),
                Self::storage_entry(13),
                Self::storage_entry(14),
                Self::storage_entry(15),
                Self::storage_entry(16),
                Self::storage_entry(17),
                wgpu::BindGroupLayoutEntry {
                    binding: 18,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba32Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("deformation_compute_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: network_meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: animation_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: orig_means_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: orig_quats_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: base_tile_means_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: base_scales_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: base_rgba_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: plane_descs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: plane_data_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: feature_layers_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: feature_weights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: feature_bias_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: pos_layers_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: pos_weights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: pos_bias_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: rot_layers_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: rot_weights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: rot_bias_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 18,
                    resource: wgpu::BindingResource::TextureView(&output_texture.view),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("deformation_compute_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("deformation_compute.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("deformation_compute_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("deformation_compute_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            compute_pipeline,
            bind_group,
            animation_uniform_buffer,
            output_texture,
            splat_count: splat_count as u32,
            gaussian_tex_width,
        })
    }

    pub fn output_texture(&self) -> &Texture {
        &self.output_texture
    }

    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        time01: f32,
    ) -> Result<(), String> {
        let anim = AnimationUniform {
            time01: time01.clamp(0.0, 1.0),
            splat_count: self.splat_count,
            gaussian_tex_width: self.gaussian_tex_width,
            _pad0: 0,
        };
        queue.write_buffer(&self.animation_uniform_buffer, 0, bytemuck::bytes_of(&anim));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("deformation_compute_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("deformation_compute_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.compute_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let workgroups = (self.splat_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    fn validate_packed_dims(packed: &PackedDeformationNetwork) -> Result<(), String> {
        let grid_feature_count = packed.metadata.n_grid_levels * packed.metadata.feature_dim;
        if grid_feature_count as u32 > MAX_GRID_FEATURES {
            return Err(format!(
                "grid feature size {} exceeds MAX_GRID_FEATURES {}",
                grid_feature_count, MAX_GRID_FEATURES
            ));
        }
        if packed.metadata.net_width as u32 > MAX_MLP_WIDTH {
            return Err(format!(
                "net_width {} exceeds MAX_MLP_WIDTH {}",
                packed.metadata.net_width, MAX_MLP_WIDTH
            ));
        }

        for (label, layers) in [
            ("feature_out", packed.feature_layers.as_slice()),
            ("pos_deform", packed.pos_layers.as_slice()),
            ("rotations_deform", packed.rot_layers.as_slice()),
        ] {
            for (i, layer) in layers.iter().enumerate() {
                if layer.in_features > MAX_MLP_WIDTH || layer.out_features > MAX_MLP_WIDTH {
                    return Err(format!(
                        "{} layer {} has dims in={} out={} exceeding MAX_MLP_WIDTH {}",
                        label, i, layer.in_features, layer.out_features, MAX_MLP_WIDTH
                    ));
                }
            }
        }
        Ok(())
    }

    fn extract_base_rgba(base_tex_data: &[u32], splat_count: usize) -> Vec<[u32; 4]> {
        let mut out = Vec::with_capacity(splat_count);
        for i in 0..splat_count {
            let idx = 8 * i;
            out.push([base_tex_data[idx + 3], base_tex_data[idx + 7], 0, 0]);
        }
        out
    }

    fn pack_plane_descs(descs: &[PackedPlaneDesc]) -> Vec<GpuPlaneDesc> {
        descs
            .iter()
            .map(|d| GpuPlaneDesc {
                width: d.width,
                height: d.height,
                channels: d.channels,
                data_offset: d.data_offset,
            })
            .collect()
    }

    fn pack_layer_descs(descs: &[PackedMlpLayerDesc]) -> Vec<GpuLayerDesc> {
        descs
            .iter()
            .map(|d| GpuLayerDesc {
                in_features: d.in_features,
                out_features: d.out_features,
                has_relu_before: d.has_relu_before,
                weight_offset: d.weight_offset,
                bias_offset: d.bias_offset,
            })
            .collect()
    }

    fn to_vec4_from_vec3(values: &[[f32; 3]]) -> Vec<[f32; 4]> {
        values.iter().map(|v| [v[0], v[1], v[2], 0.0_f32]).collect()
    }

    fn create_output_texture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        base_tex_data: &[u32],
    ) -> Texture {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("deformation_output_gaussian_texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            transmute_slice::<_, u8>(base_tex_data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 16),
                rows_per_image: Some(height),
            },
            size,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Texture {
            texture,
            view,
            sampler: None,
        }
    }

    fn create_buffer_init_or_dummy(
        device: &wgpu::Device,
        label: &str,
        bytes: &[u8],
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        const DUMMY: [u8; 4] = [0_u8; 4];
        let data = if bytes.is_empty() {
            DUMMY.as_slice()
        } else {
            bytes
        };
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: data,
            usage,
        })
    }

    fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    fn storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }
}
