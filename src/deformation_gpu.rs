use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::deformation::{
    DeformationNetwork, PackedDeformationNetwork, PackedMlpLayerDesc, PackedPlaneDesc,
};
use crate::log;
use crate::texture::Texture;
use crate::utils::{get_time_milliseconds, pack_half_2x16, transmute_slice};

pub const WORKGROUP_SIZE: u32 = 128;
pub const MAX_GRID_FEATURES: u32 = 512;
pub const MAX_MLP_WIDTH: u32 = 512;
pub const DEFORMATION_DEBUG_VOLUME: u32 = 2;
const VOLUME_WORDS_PER_SAMPLE: u32 = 4;

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
    volume_counts: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AnimationUniform {
    time01: f32,
    splat_count: u32,
    gaussian_tex_width: u32,
    debug_mode: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct VolumeBakeUniform {
    base_sample: u32,
    sample_count: u32,
    _pad0: [u32; 2],
}

pub struct GpuDeformationRuntime {
    compute_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    animation_uniform_buffer: wgpu::Buffer,
    output_texture: Texture,
    splat_count: u32,
    gaussian_tex_width: u32,
    debug_mode: u32,
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
        debug_mode: u32,
        volume_res: u32,
        volume_keys: u32,
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

        let volume_res = volume_res.max(2);
        let volume_keys = volume_keys.max(1);
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
            volume_counts: if debug_mode == DEFORMATION_DEBUG_VOLUME {
                [volume_res, volume_keys, VOLUME_WORDS_PER_SAMPLE, 0]
            } else {
                [0, 0, 0, 0]
            },
        };
        let animation_uniform = AnimationUniform {
            time01: 0.0,
            splat_count: splat_count as u32,
            gaussian_tex_width,
            debug_mode,
        };

        let orig_means_vec4 = Self::to_vec4_from_vec3(orig_means);
        let base_tile_means_vec4 = Self::to_vec4_from_vec3(base_tile_means);
        let base_scales_vec4 = Self::to_vec4_from_vec3(base_scales);
        let base_rgba = Self::extract_base_rgba(base_tex_data, splat_count);
        let plane_descs = Self::pack_plane_descs(packed.plane_descs.as_slice());
        let plane_data_bits: Vec<u32> = packed.plane_data.iter().map(|v| v.to_bits()).collect();
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
        let plane_source_data_buffer = Self::create_buffer_init_or_dummy(
            device,
            "deformation_plane_data_source",
            bytemuck::cast_slice(plane_data_bits.as_slice()),
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
        let plane_data_buffer = if debug_mode == DEFORMATION_DEBUG_VOLUME {
            match Self::generate_volume_cache_gpu(
                device,
                queue,
                &network_meta_buffer,
                volume_res,
                volume_keys,
                &plane_descs_buffer,
                &plane_source_data_buffer,
                &feature_layers_buffer,
                &feature_weights_buffer,
                &feature_bias_buffer,
                &pos_layers_buffer,
                &pos_weights_buffer,
                &pos_bias_buffer,
                &rot_layers_buffer,
                &rot_weights_buffer,
                &rot_bias_buffer,
            ) {
                Ok(buffer) => buffer,
                Err(err) => {
                    log!(
                        "GPU volume bake unavailable, fallback to CPU volume generation: {}",
                        err
                    );
                    let cpu_words = Self::generate_volume_cache(net, volume_res, volume_keys)?;
                    Self::create_buffer_init_or_dummy(
                        device,
                        "deformation_plane_data_volume_cpu",
                        bytemuck::cast_slice(cpu_words.as_slice()),
                        wgpu::BufferUsages::STORAGE,
                    )
                }
            }
        } else {
            plane_source_data_buffer.clone()
        };

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
            debug_mode,
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
            debug_mode: self.debug_mode,
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

    #[allow(clippy::too_many_arguments)]
    fn generate_volume_cache_gpu(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        network_meta_buffer: &wgpu::Buffer,
        volume_res: u32,
        volume_keys: u32,
        plane_descs_buffer: &wgpu::Buffer,
        plane_source_data_buffer: &wgpu::Buffer,
        feature_layers_buffer: &wgpu::Buffer,
        feature_weights_buffer: &wgpu::Buffer,
        feature_bias_buffer: &wgpu::Buffer,
        pos_layers_buffer: &wgpu::Buffer,
        pos_weights_buffer: &wgpu::Buffer,
        pos_bias_buffer: &wgpu::Buffer,
        rot_layers_buffer: &wgpu::Buffer,
        rot_weights_buffer: &wgpu::Buffer,
        rot_bias_buffer: &wgpu::Buffer,
    ) -> Result<wgpu::Buffer, String> {
        let sample_count = volume_res as usize * volume_res as usize * volume_res as usize;
        let total_samples = sample_count
            .checked_mul(volume_keys as usize)
            .ok_or_else(|| "deformation volume sample count overflow".to_string())?;
        let word_count = total_samples
            .checked_mul(VOLUME_WORDS_PER_SAMPLE as usize)
            .ok_or_else(|| "deformation volume word count overflow".to_string())?;
        let sample_count_u32 = u32::try_from(total_samples)
            .map_err(|_| "deformation volume sample count exceeds u32".to_string())?;
        let word_count_u64 = u64::try_from(word_count)
            .map_err(|_| "deformation volume word count exceeds u64".to_string())?;
        let byte_count = word_count_u64
            .checked_mul(std::mem::size_of::<u32>() as u64)
            .ok_or_else(|| "deformation volume byte count overflow".to_string())?;

        let max_workgroups_per_dim = device.limits().max_compute_workgroups_per_dimension;
        let max_chunk_workgroups = 4096_u32.min(max_workgroups_per_dim).max(1);
        let chunk_samples = max_chunk_workgroups
            .checked_mul(WORKGROUP_SIZE)
            .ok_or_else(|| "deformation volume chunk sample count overflow".to_string())?;
        let total_workgroups = (sample_count_u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        if total_workgroups > max_workgroups_per_dim {
            // We chunk submit below, but each individual dispatch must still fit.
        }
        if max_chunk_workgroups > max_workgroups_per_dim {
            return Err(format!(
                "max chunk workgroups {} exceeds device limit {}",
                max_chunk_workgroups,
                max_workgroups_per_dim
            ));
        }

        let bake_uniform = VolumeBakeUniform {
            base_sample: 0,
            sample_count: 0,
            _pad0: [0; 2],
        };
        let bake_uniform_buffer = Self::create_buffer_init_or_dummy(
            device,
            "deformation_volume_bake_uniform",
            bytemuck::bytes_of(&bake_uniform),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let volume_data_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("deformation_plane_data_volume_gpu"),
            size: byte_count,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let bake_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("deformation_volume_bake_bind_group_layout"),
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
                Self::storage_rw_entry(13),
            ],
        });
        let bake_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("deformation_volume_bake_bind_group"),
            layout: &bake_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: network_meta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bake_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: plane_descs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: plane_source_data_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: feature_layers_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: feature_weights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: feature_bias_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: pos_layers_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: pos_weights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: pos_bias_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: rot_layers_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: rot_weights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: rot_bias_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: volume_data_buffer.as_entire_binding(),
                },
            ],
        });

        let bake_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("deformation_volume_bake_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("deformation_volume_bake.wgsl").into()),
        });
        let bake_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("deformation_volume_bake_pipeline_layout"),
            bind_group_layouts: &[&bake_layout],
            push_constant_ranges: &[],
        });
        let bake_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("deformation_volume_bake_pipeline"),
            layout: Some(&bake_pipeline_layout),
            module: &bake_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let start_ms = get_time_milliseconds();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("deformation_volume_bake_encoder"),
        });
        let mut dispatched_chunks = 0_u32;
        let mut base_sample = 0_u32;
        while base_sample < sample_count_u32 {
            let remaining = sample_count_u32 - base_sample;
            let cur_sample_count = remaining.min(chunk_samples);
            let workgroups = (cur_sample_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

            let chunk_uniform = VolumeBakeUniform {
                base_sample,
                sample_count: cur_sample_count,
                _pad0: [0; 2],
            };
            queue.write_buffer(
                &bake_uniform_buffer,
                0,
                bytemuck::bytes_of(&chunk_uniform),
            );

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("deformation_volume_bake_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&bake_pipeline);
                pass.set_bind_group(0, &bake_bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            queue.submit(std::iter::once(encoder.finish()));
            encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("deformation_volume_bake_encoder"),
            });
            base_sample = base_sample
                .checked_add(cur_sample_count)
                .ok_or_else(|| "deformation volume base sample overflow".to_string())?;
            dispatched_chunks = dispatched_chunks.saturating_add(1);
        }
        log!(
            "deform_mode=volume volume_res={} key_frames={} volume_bytes={} gpu_bake_chunks={} gpu_bake_submit_ms={:.3}",
            volume_res,
            volume_keys,
            byte_count,
            dispatched_chunks,
            get_time_milliseconds() - start_ms
        );
        Ok(volume_data_buffer)
    }

    fn generate_volume_cache(
        net: &DeformationNetwork,
        volume_res: u32,
        volume_keys: u32,
    ) -> Result<Vec<u32>, String> {
        let sample_count = volume_res as usize * volume_res as usize * volume_res as usize;
        let total_samples = sample_count
            .checked_mul(volume_keys as usize)
            .ok_or_else(|| "deformation volume sample count overflow".to_string())?;
        let word_count = total_samples
            .checked_mul(VOLUME_WORDS_PER_SAMPLE as usize)
            .ok_or_else(|| "deformation volume word count overflow".to_string())?;

        let start_ms = get_time_milliseconds();
        let mut words = Vec::with_capacity(word_count);
        let meta = net.metadata();
        let denom = (volume_res - 1).max(1) as f32;
        let time_denom = (volume_keys - 1).max(1) as f32;

        for key in 0..volume_keys {
            let t = if volume_keys <= 1 {
                0.0
            } else {
                key as f32 / time_denom
            };
            for z in 0..volume_res {
                let zf = z as f32 / denom;
                let oz = meta.aabb_max[2] + (meta.aabb_min[2] - meta.aabb_max[2]) * zf;
                for y in 0..volume_res {
                    let yf = y as f32 / denom;
                    let oy = meta.aabb_max[1] + (meta.aabb_min[1] - meta.aabb_max[1]) * yf;
                    for x in 0..volume_res {
                        let xf = x as f32 / denom;
                        let ox = meta.aabb_max[0] + (meta.aabb_min[0] - meta.aabb_max[0]) * xf;
                        let (dx, dr) = net
                            .deform_delta_single([ox, oy, oz], t)
                            .map_err(|e| format!("deformation volume generation failed: {}", e))?;

                        words.push(pack_half_2x16(dx[0], dx[1]));
                        words.push(pack_half_2x16(dx[2], dr[0]));
                        words.push(pack_half_2x16(dr[1], dr[2]));
                        words.push(pack_half_2x16(dr[3], 0.0));
                    }
                }
            }
        }

        let elapsed_ms = get_time_milliseconds() - start_ms;
        let bytes = words.len() * std::mem::size_of::<u32>();
        log!(
            "deform_mode=volume volume_res={} key_frames={} volume_bytes={} generation_ms={:.3}",
            volume_res,
            volume_keys,
            bytes,
            elapsed_ms
        );
        Ok(words)
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

    fn storage_rw_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }
}
