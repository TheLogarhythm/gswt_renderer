use std::sync::{Arc, Mutex};

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::catmull_rom_motion::{CatmullRomMotionSet, CatmullRomMotionTeacher};
use crate::deformation_gpu::{GpuDeformationRuntime, WORKGROUP_SIZE};
use crate::structure::{
    MotionCompatibilityResult, MotionCompatibilityScope, MotionTextureCompareResult,
};
use crate::texture::Texture;
use crate::utils::transmute_slice;

const PREFERRED_KNOT_TEXTURE_WIDTH: u32 = 4096;
const KNOT_TEXEL_BYTES: u32 = 16;
const MAX_KNOT_TEXTURE_UPLOAD_BYTES: usize = 64 * 1024 * 1024;
const MAX_COMPATIBILITY_SAMPLE_ERRORS: u32 = 131_072;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MotionUniform {
    time01: f32,
    splat_count: u32,
    gaussian_tex_width: u32,
    knot_count: u32,
    knot_texture_width: u32,
    time_sampling: u32,
    _pad0: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CompatibilityUniform {
    item_count: u32,
    splat_count: u32,
    knot_count: u32,
    knot_texture_width: u32,
    selected_knot: u32,
    compare_all: u32,
    sample_stride: u32,
    sample_count: u32,
    comparison_knot_count: u32,
    source_volume_key_count: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CompatibilityPartialStats {
    count: u32,
    worst_item: u32,
    nonzero_error_count: u32,
    nonzero_spline_count: u32,
    nonzero_volume_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    sum_error: f32,
    sum_error_sq: f32,
    max_error: f32,
    sum_spline_magnitude: f32,
    max_spline_magnitude: f32,
    sum_volume_magnitude: f32,
    max_volume_magnitude: f32,
    _pad3: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TextureCompareUniform {
    item_count: u32,
    splat_count: u32,
    gaussian_tex_width: u32,
    sample_stride: u32,
    sample_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TextureComparePartialStats {
    count: u32,
    worst_item: u32,
    nonzero_error_count: u32,
    _pad0: u32,
    sum_error: f32,
    sum_error_sq: f32,
    max_error: f32,
    sum_cat_magnitude: f32,
    max_cat_magnitude: f32,
    sum_volume_magnitude: f32,
    max_volume_magnitude: f32,
    _pad1: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct KnotTextureLayout {
    width: u32,
    height: u32,
    total_knots: u32,
}

pub struct GpuCatmullRomMotionRuntime {
    compute_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
    _base_texture: Texture,
    knot_texture: Texture,
    sample_times_buffer: wgpu::Buffer,
    output_texture: Texture,
    splat_count: u32,
    gaussian_tex_width: u32,
    knot_count: u32,
    knot_texture_width: u32,
    volume_comparison_knot_count: u32,
    source_volume_key_count: u32,
    time_sampling: u32,
}

pub struct MotionCompatibilityPending {
    result: Arc<Mutex<Option<Result<MotionCompatibilityResult, String>>>>,
    _readback_buffer: wgpu::Buffer,
}

impl MotionCompatibilityPending {
    pub fn take_result(&mut self) -> Option<Result<MotionCompatibilityResult, String>> {
        self.result.lock().ok()?.take()
    }
}

pub struct MotionTextureComparePending {
    result: Arc<Mutex<Option<Result<MotionTextureCompareResult, String>>>>,
    _readback_buffer: wgpu::Buffer,
}

impl MotionTextureComparePending {
    pub fn take_result(&mut self) -> Option<Result<MotionTextureCompareResult, String>> {
        self.result.lock().ok()?.take()
    }
}

impl GpuCatmullRomMotionRuntime {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        motion: &CatmullRomMotionSet,
        gaussian_tex_width: u32,
        gaussian_tex_height: u32,
        base_tex_data: &[u32],
    ) -> Result<Self, String> {
        if motion.total_splats > u32::MAX as usize {
            return Err("Catmull-Rom splat count exceeds u32".to_string());
        }
        let knot_count = motion.meta.knot_count;
        if knot_count == 0 || knot_count > u32::MAX as usize {
            return Err(format!("invalid Catmull-Rom knot count {}", knot_count));
        }
        let expected_values = motion
            .total_splats
            .checked_mul(knot_count)
            .and_then(|v| v.checked_mul(3))
            .ok_or_else(|| "Catmull-Rom knot value count overflow".to_string())?;
        if motion.global_knots.len() != expected_values {
            return Err(format!(
                "Catmull-Rom global knot length mismatch: got {}, expected {}",
                motion.global_knots.len(),
                expected_values
            ));
        }

        let base_texture = create_uint_texture(
            device,
            queue,
            "catmull_rom_base_gaussian_texture",
            gaussian_tex_width,
            gaussian_tex_height,
            base_tex_data,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        let output_texture = create_uint_texture(
            device,
            queue,
            "catmull_rom_output_gaussian_texture",
            gaussian_tex_width,
            gaussian_tex_height,
            base_tex_data,
            wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST,
        );

        let knot_layout = knot_texture_layout(
            motion.total_splats,
            knot_count,
            device.limits().max_texture_dimension_2d,
        )?;
        let knot_texture_mib =
            knot_layout.width as f64 * knot_layout.height as f64 * KNOT_TEXEL_BYTES as f64
                / (1024.0 * 1024.0);
        crate::log!(
            "Catmull-Rom knot texture: {}x{} rgba32float ({:.1} MiB, knots={}, splats={})",
            knot_layout.width,
            knot_layout.height,
            knot_texture_mib,
            knot_count,
            motion.total_splats
        );
        let knot_texture = create_knot_texture(
            device,
            queue,
            "catmull_rom_delta_xyz_knots",
            motion.global_knots.as_slice(),
            knot_layout,
        )?;
        let sample_times_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("catmull_rom_sample_times"),
            contents: bytemuck::cast_slice(motion.meta.sample_times.as_slice()),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let uniform = MotionUniform {
            time01: 0.0,
            splat_count: motion.total_splats as u32,
            gaussian_tex_width,
            knot_count: knot_count as u32,
            knot_texture_width: knot_layout.width,
            time_sampling: motion.meta.time_sampling.shader_value(),
            _pad0: [0; 2],
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("catmull_rom_motion_uniform"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = Self::create_bind_group_layout(device);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("catmull_rom_motion_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&base_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&knot_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&output_texture.view),
                },
            ],
        });
        let compute_pipeline = Self::create_compute_pipeline(device, &bind_group_layout);

        Ok(Self {
            compute_pipeline,
            bind_group,
            uniform_buffer,
            _base_texture: base_texture,
            knot_texture,
            sample_times_buffer,
            output_texture,
            splat_count: motion.total_splats as u32,
            gaussian_tex_width,
            knot_count: knot_count as u32,
            knot_texture_width: knot_layout.width,
            volume_comparison_knot_count: volume_comparison_knot_count(motion),
            source_volume_key_count: source_volume_key_count(motion),
            time_sampling: motion.meta.time_sampling.shader_value(),
        })
    }

    pub fn output_texture(&self) -> &Texture {
        &self.output_texture
    }

    pub fn knot_count(&self) -> u32 {
        self.knot_count
    }

    pub fn uses_volume_key_times(&self) -> bool {
        self.time_sampling == 1
    }

    pub fn uses_periodic_times(&self) -> bool {
        !self.uses_volume_key_times()
    }

    pub fn knot_preview_time(&self, selected_knot: u32) -> f32 {
        knot_preview_time_for_sampling(selected_knot, self.knot_count, self.uses_volume_key_times())
    }

    pub fn volume_comparison_time(&self, selected_knot: u32) -> f32 {
        if self.source_volume_key_count > 1 {
            let knot = selected_knot.min(self.volume_comparison_knot_count.saturating_sub(1));
            knot as f32 / (self.source_volume_key_count - 1) as f32
        } else {
            self.knot_preview_time(selected_knot)
        }
    }

    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        time01: f32,
    ) -> Result<(), String> {
        let uniform = MotionUniform {
            time01: if self.uses_volume_key_times() {
                time01.clamp(0.0, 1.0)
            } else {
                time01.rem_euclid(1.0)
            },
            splat_count: self.splat_count,
            gaussian_tex_width: self.gaussian_tex_width,
            knot_count: self.knot_count,
            knot_texture_width: self.knot_texture_width,
            time_sampling: self.time_sampling,
            _pad0: [0; 2],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniform));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("catmull_rom_motion_compute_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("catmull_rom_motion_compute_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.compute_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(
                (self.splat_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE,
                1,
                1,
            );
        }
        queue.submit(Some(encoder.finish()));
        Ok(())
    }

    pub fn compare_to_volume_async(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        volume_runtime: &GpuDeformationRuntime,
        scope: MotionCompatibilityScope,
        selected_knot: u32,
    ) -> Result<MotionCompatibilityPending, String> {
        let volume_data_buffer = volume_runtime.volume_data_buffer().ok_or_else(|| {
            "compatibility comparison requires deform_mode=volume cache".to_string()
        })?;
        if volume_runtime.volume_key_count() == 0 || volume_runtime.volume_res() < 2 {
            return Err("invalid volume cache dimensions for compatibility comparison".to_string());
        }
        let compare_all = scope == MotionCompatibilityScope::AllKnots;
        let comparison_knot_count = self
            .volume_comparison_knot_count
            .max(1)
            .min(self.knot_count);
        let selected_knot = selected_knot.min(comparison_knot_count.saturating_sub(1));
        let item_count = if compare_all {
            self.splat_count
                .checked_mul(comparison_knot_count)
                .ok_or_else(|| "compatibility item count overflow".to_string())?
        } else {
            self.splat_count
        };
        if item_count == 0 {
            return Err("no splats available for compatibility comparison".to_string());
        }

        let partial_count = (item_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        let sample_stride = ((item_count + MAX_COMPATIBILITY_SAMPLE_ERRORS - 1)
            / MAX_COMPATIBILITY_SAMPLE_ERRORS)
            .max(1);
        let sample_count = (item_count + sample_stride - 1) / sample_stride;
        let partial_bytes =
            partial_count as u64 * std::mem::size_of::<CompatibilityPartialStats>() as u64;
        let sample_bytes = sample_count as u64 * std::mem::size_of::<f32>() as u64;
        let readback_size = partial_bytes
            .checked_add(sample_bytes)
            .ok_or_else(|| "compatibility readback size overflow".to_string())?;

        let uniform = CompatibilityUniform {
            item_count,
            splat_count: self.splat_count,
            knot_count: self.knot_count,
            knot_texture_width: self.knot_texture_width,
            selected_knot,
            compare_all: u32::from(compare_all),
            sample_stride,
            sample_count,
            comparison_knot_count,
            source_volume_key_count: self.source_volume_key_count,
            _pad0: 0,
            _pad1: 0,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("motion_compatibility_uniform"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let partial_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("motion_compatibility_partial_stats"),
            size: partial_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let sample_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("motion_compatibility_sample_errors"),
            size: sample_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("motion_compatibility_readback"),
            size: readback_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = create_compatibility_bind_group_layout(device);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("motion_compatibility_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: volume_runtime.network_meta_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: volume_runtime.orig_means_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: volume_data_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.knot_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.sample_times_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: partial_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: sample_buffer.as_entire_binding(),
                },
            ],
        });
        let pipeline = create_compatibility_pipeline(device, &bind_group_layout);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("motion_compatibility_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("motion_compatibility_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(partial_count, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&partial_buffer, 0, &readback_buffer, 0, partial_bytes);
        encoder.copy_buffer_to_buffer(
            &sample_buffer,
            0,
            &readback_buffer,
            partial_bytes,
            sample_bytes,
        );
        queue.submit(Some(encoder.finish()));

        let result = Arc::new(Mutex::new(None));
        let callback_result = result.clone();
        let readback_for_callback = readback_buffer.clone();
        let result_splat_count = self.splat_count;
        let result_knot_count = self.knot_count;
        let result_selected_knot = selected_knot;
        let result_compared_knots = if compare_all {
            comparison_knot_count
        } else {
            1
        };
        readback_buffer.map_async(wgpu::MapMode::Read, .., move |map_result| {
            let parsed = map_result
                .map_err(|err| format!("compatibility readback map failed: {}", err))
                .and_then(|_| {
                    let view = readback_for_callback.get_mapped_range(..);
                    let partial_len = partial_bytes as usize;
                    let partials: &[CompatibilityPartialStats] =
                        bytemuck::cast_slice(&view[..partial_len]);
                    let samples: &[f32] = bytemuck::cast_slice(&view[partial_len..]);
                    let result = reduce_compatibility_stats(
                        partials,
                        samples,
                        scope,
                        result_splat_count,
                        result_knot_count,
                        result_selected_knot,
                        result_compared_knots,
                    );
                    drop(view);
                    readback_for_callback.unmap();
                    Ok(result)
                });
            if let Ok(mut slot) = callback_result.lock() {
                *slot = Some(parsed);
            }
        });
        let _ = device.poll(wgpu::PollType::Poll);

        Ok(MotionCompatibilityPending {
            result,
            _readback_buffer: readback_buffer,
        })
    }

    pub fn compare_final_means_to_volume_async(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        volume_runtime: &GpuDeformationRuntime,
        catmull_rom_time01: f32,
        volume_time01: f32,
    ) -> Result<MotionTextureComparePending, String> {
        self.dispatch(device, queue, catmull_rom_time01)?;
        volume_runtime.dispatch(device, queue, volume_time01, false)?;

        let item_count = self.splat_count;
        if item_count == 0 {
            return Err("no splats available for texture comparison".to_string());
        }
        let partial_count = (item_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        let sample_stride = ((item_count + MAX_COMPATIBILITY_SAMPLE_ERRORS - 1)
            / MAX_COMPATIBILITY_SAMPLE_ERRORS)
            .max(1);
        let sample_count = (item_count + sample_stride - 1) / sample_stride;
        let partial_bytes =
            partial_count as u64 * std::mem::size_of::<TextureComparePartialStats>() as u64;
        let sample_bytes = sample_count as u64 * std::mem::size_of::<f32>() as u64;
        let readback_size = partial_bytes
            .checked_add(sample_bytes)
            .ok_or_else(|| "texture comparison readback size overflow".to_string())?;

        let uniform = TextureCompareUniform {
            item_count,
            splat_count: self.splat_count,
            gaussian_tex_width: self.gaussian_tex_width,
            sample_stride,
            sample_count,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("motion_texture_compare_uniform"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let partial_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("motion_texture_compare_partial_stats"),
            size: partial_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let sample_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("motion_texture_compare_sample_errors"),
            size: sample_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("motion_texture_compare_readback"),
            size: readback_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = create_texture_compare_bind_group_layout(device);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("motion_texture_compare_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.output_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        &volume_runtime.output_texture().view,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: partial_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: sample_buffer.as_entire_binding(),
                },
            ],
        });
        let pipeline = create_texture_compare_pipeline(device, &bind_group_layout);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("motion_texture_compare_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("motion_texture_compare_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(partial_count, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&partial_buffer, 0, &readback_buffer, 0, partial_bytes);
        encoder.copy_buffer_to_buffer(
            &sample_buffer,
            0,
            &readback_buffer,
            partial_bytes,
            sample_bytes,
        );
        queue.submit(Some(encoder.finish()));

        let result = Arc::new(Mutex::new(None));
        let callback_result = result.clone();
        let readback_for_callback = readback_buffer.clone();
        let result_splat_count = self.splat_count;
        readback_buffer.map_async(wgpu::MapMode::Read, .., move |map_result| {
            let parsed = map_result
                .map_err(|err| format!("texture comparison readback map failed: {}", err))
                .and_then(|_| {
                    let view = readback_for_callback.get_mapped_range(..);
                    let partial_len = partial_bytes as usize;
                    let partials: &[TextureComparePartialStats] =
                        bytemuck::cast_slice(&view[..partial_len]);
                    let samples: &[f32] = bytemuck::cast_slice(&view[partial_len..]);
                    let result = reduce_texture_compare_stats(
                        partials,
                        samples,
                        result_splat_count,
                        catmull_rom_time01,
                        volume_time01,
                    );
                    drop(view);
                    readback_for_callback.unmap();
                    Ok(result)
                });
            if let Ok(mut slot) = callback_result.lock() {
                *slot = Some(parsed);
            }
        });
        let _ = device.poll(wgpu::PollType::Poll);

        Ok(MotionTextureComparePending {
            result,
            _readback_buffer: readback_buffer,
        })
    }

    fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("catmull_rom_motion_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Uint,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba32Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        })
    }

    fn create_compute_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("catmull_rom_motion_compute_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("catmull_rom_motion_compute.wgsl").into(),
            ),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("catmull_rom_motion_pipeline_layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("catmull_rom_motion_compute_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    }
}

fn create_compatibility_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("motion_compatibility_bind_group_layout"),
        entries: &[
            buffer_entry(0, wgpu::BufferBindingType::Uniform),
            buffer_entry(1, wgpu::BufferBindingType::Uniform),
            buffer_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
            buffer_entry(3, wgpu::BufferBindingType::Storage { read_only: true }),
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                },
                count: None,
            },
            buffer_entry(5, wgpu::BufferBindingType::Storage { read_only: true }),
            buffer_entry(6, wgpu::BufferBindingType::Storage { read_only: false }),
            buffer_entry(7, wgpu::BufferBindingType::Storage { read_only: false }),
        ],
    })
}

fn buffer_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn create_compatibility_pipeline(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("motion_compatibility_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("motion_compatibility_compute.wgsl").into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("motion_compatibility_pipeline_layout"),
        bind_group_layouts: &[bind_group_layout],
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("motion_compatibility_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

fn create_texture_compare_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("motion_texture_compare_bind_group_layout"),
        entries: &[
            buffer_entry(0, wgpu::BufferBindingType::Uniform),
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Uint,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Uint,
                },
                count: None,
            },
            buffer_entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
            buffer_entry(4, wgpu::BufferBindingType::Storage { read_only: false }),
        ],
    })
}

fn create_texture_compare_pipeline(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("motion_texture_compare_shader"),
        source: wgpu::ShaderSource::Wgsl(
            include_str!("motion_texture_compare_compute.wgsl").into(),
        ),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("motion_texture_compare_pipeline_layout"),
        bind_group_layouts: &[bind_group_layout],
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("motion_texture_compare_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

fn reduce_compatibility_stats(
    partials: &[CompatibilityPartialStats],
    samples: &[f32],
    scope: MotionCompatibilityScope,
    splat_count: u32,
    knot_count: u32,
    selected_knot: u32,
    compared_knots: u32,
) -> MotionCompatibilityResult {
    let mut count = 0_u64;
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut max_error = 0.0_f32;
    let mut worst_item = 0_u32;
    let mut nonzero_error_count = 0_u64;
    let mut nonzero_spline_delta_count = 0_u64;
    let mut nonzero_volume_delta_count = 0_u64;
    let mut sum_spline_magnitude = 0.0_f64;
    let mut max_spline_magnitude = 0.0_f32;
    let mut sum_volume_magnitude = 0.0_f64;
    let mut max_volume_magnitude = 0.0_f32;
    for partial in partials {
        count += partial.count as u64;
        sum += partial.sum_error as f64;
        sum_sq += partial.sum_error_sq as f64;
        nonzero_error_count += partial.nonzero_error_count as u64;
        nonzero_spline_delta_count += partial.nonzero_spline_count as u64;
        nonzero_volume_delta_count += partial.nonzero_volume_count as u64;
        sum_spline_magnitude += partial.sum_spline_magnitude as f64;
        sum_volume_magnitude += partial.sum_volume_magnitude as f64;
        if partial.max_error > max_error {
            max_error = partial.max_error;
            worst_item = partial.worst_item;
        }
        max_spline_magnitude = max_spline_magnitude.max(partial.max_spline_magnitude);
        max_volume_magnitude = max_volume_magnitude.max(partial.max_volume_magnitude);
    }
    let mut sample_errors = samples
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect::<Vec<_>>();
    sample_errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let sampled_p95_error = if sample_errors.is_empty() {
        0.0
    } else {
        let idx = ((sample_errors.len() - 1) as f32 * 0.95).round() as usize;
        sample_errors[idx.min(sample_errors.len() - 1)]
    };
    let denom = (count.max(1)) as f64;
    let (worst_knot, worst_splat) = if scope == MotionCompatibilityScope::AllKnots {
        (
            worst_item / splat_count.max(1),
            worst_item % splat_count.max(1),
        )
    } else {
        (selected_knot, worst_item)
    };
    MotionCompatibilityResult {
        scope,
        actual_compared_count: count,
        mean_error: (sum / denom) as f32,
        rms_error: (sum_sq / denom).sqrt() as f32,
        max_error,
        sampled_p95_error,
        mean_spline_delta_magnitude: (sum_spline_magnitude / denom) as f32,
        max_spline_delta_magnitude: max_spline_magnitude,
        mean_volume_delta_magnitude: (sum_volume_magnitude / denom) as f32,
        max_volume_delta_magnitude: max_volume_magnitude,
        nonzero_error_count,
        nonzero_spline_delta_count,
        nonzero_volume_delta_count,
        splat_count,
        knot_count,
        compared_knots,
        worst_knot,
        worst_splat,
        sampled_error_count: sample_errors.len() as u32,
    }
}

fn reduce_texture_compare_stats(
    partials: &[TextureComparePartialStats],
    samples: &[f32],
    splat_count: u32,
    time01: f32,
    volume_time01: f32,
) -> MotionTextureCompareResult {
    let mut count = 0_u64;
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut max_error = 0.0_f32;
    let mut worst_splat = 0_u32;
    let mut nonzero_error_count = 0_u64;
    let mut sum_cat_magnitude = 0.0_f64;
    let mut max_cat_magnitude = 0.0_f32;
    let mut sum_volume_magnitude = 0.0_f64;
    let mut max_volume_magnitude = 0.0_f32;
    for partial in partials {
        count += partial.count as u64;
        sum += partial.sum_error as f64;
        sum_sq += partial.sum_error_sq as f64;
        nonzero_error_count += partial.nonzero_error_count as u64;
        sum_cat_magnitude += partial.sum_cat_magnitude as f64;
        sum_volume_magnitude += partial.sum_volume_magnitude as f64;
        if partial.max_error > max_error {
            max_error = partial.max_error;
            worst_splat = partial.worst_item;
        }
        max_cat_magnitude = max_cat_magnitude.max(partial.max_cat_magnitude);
        max_volume_magnitude = max_volume_magnitude.max(partial.max_volume_magnitude);
    }
    let mut sample_errors = samples
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect::<Vec<_>>();
    sample_errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let sampled_p95_error = if sample_errors.is_empty() {
        0.0
    } else {
        let idx = ((sample_errors.len() - 1) as f32 * 0.95).round() as usize;
        sample_errors[idx.min(sample_errors.len() - 1)]
    };
    let denom = (count.max(1)) as f64;
    MotionTextureCompareResult {
        time01,
        volume_time01,
        actual_compared_count: count,
        mean_error: (sum / denom) as f32,
        rms_error: (sum_sq / denom).sqrt() as f32,
        max_error,
        sampled_p95_error,
        mean_catmull_rom_mean_magnitude: (sum_cat_magnitude / denom) as f32,
        max_catmull_rom_mean_magnitude: max_cat_magnitude,
        mean_volume_mean_magnitude: (sum_volume_magnitude / denom) as f32,
        max_volume_mean_magnitude: max_volume_magnitude,
        nonzero_error_count,
        splat_count,
        worst_splat: worst_splat.min(splat_count.saturating_sub(1)),
        sampled_error_count: sample_errors.len() as u32,
    }
}

fn knot_preview_time_for_sampling(
    selected_knot: u32,
    knot_count: u32,
    uses_volume_keys: bool,
) -> f32 {
    if knot_count == 0 {
        return 0.0;
    }
    let knot = selected_knot.min(knot_count - 1) as f32;
    if uses_volume_keys {
        knot / (knot_count - 1).max(1) as f32
    } else {
        knot / knot_count as f32
    }
}

fn volume_comparison_knot_count(motion: &CatmullRomMotionSet) -> u32 {
    motion
        .meta
        .source_knot_count
        .unwrap_or(motion.meta.knot_count)
        .min(motion.meta.knot_count)
        .max(1) as u32
}

fn source_volume_key_count(motion: &CatmullRomMotionSet) -> u32 {
    if motion.meta.motion_teacher == CatmullRomMotionTeacher::Volume {
        motion
            .meta
            .source_knot_count
            .or(motion.meta.volume_key_count)
            .unwrap_or(0) as u32
    } else {
        0
    }
}

fn create_uint_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    width: u32,
    height: u32,
    tex_data: &[u32],
    usage: wgpu::TextureUsages,
) -> Texture {
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Uint,
        usage,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        transmute_slice::<_, u8>(tex_data),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 16),
            rows_per_image: Some(height),
        },
        size,
    );
    Texture {
        texture: texture.clone(),
        view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
        sampler: None,
    }
}

fn knot_texture_layout(
    splat_count: usize,
    knot_count: usize,
    max_texture_dimension_2d: u32,
) -> Result<KnotTextureLayout, String> {
    let total_knots = splat_count
        .checked_mul(knot_count)
        .ok_or_else(|| "Catmull-Rom knot count overflow".to_string())?;
    if total_knots > u32::MAX as usize {
        return Err("Catmull-Rom knot count exceeds u32".to_string());
    }
    if max_texture_dimension_2d < 16 {
        return Err(format!(
            "max_texture_dimension_2d {} is too small for Catmull-Rom knot texture",
            max_texture_dimension_2d
        ));
    }
    let mut width = PREFERRED_KNOT_TEXTURE_WIDTH.min(max_texture_dimension_2d);
    let min_width_for_height =
        ((total_knots as u32) + max_texture_dimension_2d - 1) / max_texture_dimension_2d;
    width = width.max(min_width_for_height);
    width = ((width + 15) / 16) * 16;
    if width > max_texture_dimension_2d {
        return Err(format!(
            "Catmull-Rom knot texture requires width {}, exceeding max_texture_dimension_2d {}",
            width, max_texture_dimension_2d
        ));
    }
    let height = ((total_knots as u32) + width - 1) / width;
    if height > max_texture_dimension_2d {
        return Err(format!(
            "Catmull-Rom knot texture height {} exceeds max_texture_dimension_2d {}",
            height, max_texture_dimension_2d
        ));
    }
    Ok(KnotTextureLayout {
        width,
        height,
        total_knots: total_knots as u32,
    })
}

fn create_knot_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    knots_xyz: &[f32],
    layout: KnotTextureLayout,
) -> Result<Texture, String> {
    if knots_xyz.len() != layout.total_knots as usize * 3 {
        return Err(format!(
            "{} knot float count {} does not match layout total {}",
            label,
            knots_xyz.len(),
            layout.total_knots as usize * 3
        ));
    }
    let size = wgpu::Extent3d {
        width: layout.width,
        height: layout.height,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    write_knot_texture_in_chunks(queue, &texture, knots_xyz, layout)?;
    Ok(Texture {
        texture: texture.clone(),
        view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
        sampler: None,
    })
}

fn write_knot_texture_in_chunks(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    knots_xyz: &[f32],
    layout: KnotTextureLayout,
) -> Result<(), String> {
    let bytes_per_row = layout.width * KNOT_TEXEL_BYTES;
    let rows_per_chunk = ((MAX_KNOT_TEXTURE_UPLOAD_BYTES / bytes_per_row as usize).max(1)) as u32;
    let row_texels = layout.width as usize;
    let mut row_start = 0_u32;
    while row_start < layout.height {
        let row_count = rows_per_chunk.min(layout.height - row_start);
        let chunk_texel_count = row_count as usize * row_texels;
        let chunk_start_flat = row_start as usize * row_texels;
        let mut texels = vec![[0.0_f32; 4]; chunk_texel_count];
        for (local_idx, texel) in texels.iter_mut().enumerate() {
            let flat_idx = chunk_start_flat + local_idx;
            if flat_idx >= layout.total_knots as usize {
                break;
            }
            let src = flat_idx * 3;
            *texel = [knots_xyz[src], knots_xyz[src + 1], knots_xyz[src + 2], 0.0];
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: row_start,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(texels.as_slice()),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(row_count),
            },
            wgpu::Extent3d {
                width: layout.width,
                height: row_count,
                depth_or_array_layers: 1,
            },
        );
        row_start += row_count;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catmull_rom_motion_uniform_is_32_bytes() {
        assert_eq!(std::mem::size_of::<MotionUniform>(), 32);
    }

    #[test]
    fn compatibility_uniform_and_partial_stats_have_expected_layout() {
        assert_eq!(std::mem::size_of::<CompatibilityUniform>(), 48);
        assert_eq!(std::mem::size_of::<CompatibilityPartialStats>(), 64);
        assert_eq!(std::mem::size_of::<TextureCompareUniform>(), 32);
        assert_eq!(std::mem::size_of::<TextureComparePartialStats>(), 48);
    }

    #[test]
    fn compatibility_reduction_reports_aggregate_errors() {
        let partials = [
            CompatibilityPartialStats {
                count: 2,
                worst_item: 1,
                nonzero_error_count: 2,
                nonzero_spline_count: 1,
                nonzero_volume_count: 2,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
                sum_error: 3.0,
                sum_error_sq: 5.0,
                max_error: 2.0,
                sum_spline_magnitude: 10.0,
                max_spline_magnitude: 7.0,
                sum_volume_magnitude: 12.0,
                max_volume_magnitude: 8.0,
                _pad3: 0.0,
            },
            CompatibilityPartialStats {
                count: 1,
                worst_item: 4,
                nonzero_error_count: 1,
                nonzero_spline_count: 1,
                nonzero_volume_count: 1,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
                sum_error: 4.0,
                sum_error_sq: 16.0,
                max_error: 4.0,
                sum_spline_magnitude: 5.0,
                max_spline_magnitude: 5.0,
                sum_volume_magnitude: 6.0,
                max_volume_magnitude: 6.0,
                _pad3: 0.0,
            },
        ];
        let samples = [1.0, 4.0, 2.0];

        let result = reduce_compatibility_stats(
            &partials,
            &samples,
            MotionCompatibilityScope::AllKnots,
            3,
            2,
            0,
            2,
        );

        assert_eq!(result.scope, MotionCompatibilityScope::AllKnots);
        assert_eq!(result.splat_count, 3);
        assert_eq!(result.knot_count, 2);
        assert_eq!(result.compared_knots, 2);
        assert_eq!(result.worst_knot, 1);
        assert_eq!(result.worst_splat, 1);
        assert_eq!(result.actual_compared_count, 3);
        assert_eq!(result.nonzero_error_count, 3);
        assert_eq!(result.nonzero_spline_delta_count, 2);
        assert_eq!(result.nonzero_volume_delta_count, 3);
        assert_eq!(result.sampled_error_count, 3);
        assert!((result.mean_error - 7.0 / 3.0).abs() < 1e-6);
        assert!((result.rms_error - (21.0_f32 / 3.0).sqrt()).abs() < 1e-6);
        assert!((result.mean_spline_delta_magnitude - 5.0).abs() < 1e-6);
        assert_eq!(result.max_spline_delta_magnitude, 7.0);
        assert!((result.mean_volume_delta_magnitude - 6.0).abs() < 1e-6);
        assert_eq!(result.max_volume_delta_magnitude, 8.0);
        assert_eq!(result.max_error, 4.0);
        assert_eq!(result.sampled_p95_error, 4.0);
    }

    #[test]
    fn selected_knot_reduction_reports_selected_knot_as_worst_knot() {
        let partials = [CompatibilityPartialStats {
            count: 1,
            worst_item: 42,
            nonzero_error_count: 1,
            nonzero_spline_count: 0,
            nonzero_volume_count: 1,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            sum_error: 0.25,
            sum_error_sq: 0.0625,
            max_error: 0.25,
            sum_spline_magnitude: 0.0,
            max_spline_magnitude: 0.0,
            sum_volume_magnitude: 0.25,
            max_volume_magnitude: 0.25,
            _pad3: 0.0,
        }];

        let result = reduce_compatibility_stats(
            &partials,
            &[0.25],
            MotionCompatibilityScope::SelectedKnot,
            100,
            25,
            7,
            1,
        );

        assert_eq!(result.worst_knot, 7);
        assert_eq!(result.worst_splat, 42);
        assert_eq!(result.compared_knots, 1);
    }

    #[test]
    fn texture_compare_reduction_reports_zero_for_identical_means() {
        let partials = [TextureComparePartialStats {
            count: 2,
            worst_item: 0,
            nonzero_error_count: 0,
            _pad0: 0,
            sum_error: 0.0,
            sum_error_sq: 0.0,
            max_error: 0.0,
            sum_cat_magnitude: 10.0,
            max_cat_magnitude: 6.0,
            sum_volume_magnitude: 10.0,
            max_volume_magnitude: 6.0,
            _pad1: 0.0,
        }];

        let result = reduce_texture_compare_stats(&partials, &[0.0, 0.0], 2, 0.5, 0.75);

        assert_eq!(result.actual_compared_count, 2);
        assert_eq!(result.nonzero_error_count, 0);
        assert_eq!(result.mean_error, 0.0);
        assert_eq!(result.rms_error, 0.0);
        assert_eq!(result.max_error, 0.0);
        assert_eq!(result.sampled_p95_error, 0.0);
        assert!((result.mean_catmull_rom_mean_magnitude - 5.0).abs() < 1e-6);
        assert!((result.mean_volume_mean_magnitude - 5.0).abs() < 1e-6);
        assert_eq!(result.worst_splat, 0);
        assert_eq!(result.time01, 0.5);
        assert_eq!(result.volume_time01, 0.75);
    }

    #[test]
    fn texture_compare_reduction_reports_nonzero_final_mean_errors() {
        let partials = [
            TextureComparePartialStats {
                count: 2,
                worst_item: 1,
                nonzero_error_count: 1,
                _pad0: 0,
                sum_error: 3.0,
                sum_error_sq: 9.0,
                max_error: 3.0,
                sum_cat_magnitude: 12.0,
                max_cat_magnitude: 7.0,
                sum_volume_magnitude: 11.0,
                max_volume_magnitude: 6.0,
                _pad1: 0.0,
            },
            TextureComparePartialStats {
                count: 1,
                worst_item: 2,
                nonzero_error_count: 1,
                _pad0: 0,
                sum_error: 4.0,
                sum_error_sq: 16.0,
                max_error: 4.0,
                sum_cat_magnitude: 5.0,
                max_cat_magnitude: 5.0,
                sum_volume_magnitude: 9.0,
                max_volume_magnitude: 9.0,
                _pad1: 0.0,
            },
        ];

        let result = reduce_texture_compare_stats(&partials, &[3.0, 0.0, 4.0], 3, 1.0, 1.0);

        assert_eq!(result.actual_compared_count, 3);
        assert_eq!(result.nonzero_error_count, 2);
        assert!((result.mean_error - 7.0 / 3.0).abs() < 1e-6);
        assert!((result.rms_error - (25.0_f32 / 3.0).sqrt()).abs() < 1e-6);
        assert_eq!(result.max_error, 4.0);
        assert_eq!(result.sampled_p95_error, 4.0);
        assert_eq!(result.worst_splat, 2);
        assert!((result.mean_catmull_rom_mean_magnitude - 17.0 / 3.0).abs() < 1e-6);
        assert_eq!(result.max_catmull_rom_mean_magnitude, 7.0);
        assert!((result.mean_volume_mean_magnitude - 20.0 / 3.0).abs() < 1e-6);
        assert_eq!(result.max_volume_mean_magnitude, 9.0);
    }

    #[test]
    fn selected_knot_time_handles_volume_key_and_periodic_assets() {
        assert!((knot_preview_time_for_sampling(24, 25, true) - 1.0).abs() < 1e-6);
        assert!((knot_preview_time_for_sampling(24, 25, false) - 24.0 / 25.0).abs() < 1e-6);
        assert_eq!(knot_preview_time_for_sampling(100, 25, true), 1.0);
        assert_eq!(knot_preview_time_for_sampling(0, 0, true), 0.0);
    }

    #[test]
    fn volume_comparison_uses_source_knots_for_loop_closure_assets() {
        let motion = CatmullRomMotionSet {
            meta: crate::catmull_rom_motion::CatmullRomMotionMeta {
                knot_count: 28,
                include_lods: vec![0],
                sample_times: (0..28).map(|i| i as f32 / 28.0).collect(),
                time_sampling: crate::catmull_rom_motion::CatmullRomTimeSampling::Periodic,
                sample_time_grid: crate::catmull_rom_motion::CatmullRomSampleTimeGrid::Periodic,
                has_time_sampling_field: true,
                periodic_flag: true,
                motion_teacher: CatmullRomMotionTeacher::Volume,
                volume_res: Some(64),
                volume_key_count: Some(25),
                source_knot_count: Some(25),
                exported_knot_count: Some(28),
                loop_closure_knots: Some(3),
                loop_closure_method: Some("cubic_hermite".to_string()),
                source_frame_count: Some(75),
                source_fps: Some(30.0),
                duration_seconds: Some(2.5),
            },
            total_splats: 1,
            global_knots: vec![0.0; 28 * 3],
        };

        assert_eq!(volume_comparison_knot_count(&motion), 25);
        assert_eq!(source_volume_key_count(&motion), 25);
    }

    #[test]
    fn catmull_rom_knot_texture_layout_covers_all_knots() {
        let layout = knot_texture_layout(1000, 32, 8192).unwrap();
        assert_eq!(layout.width, 4096);
        assert!(layout.width * layout.height >= layout.total_knots);
        assert_eq!(layout.total_knots, 32_000);
    }

    #[test]
    fn catmull_rom_knot_texture_layout_expands_width_to_fit_height_limit() {
        let layout = knot_texture_layout(2_641_692, 32, 16_384).unwrap();
        assert!(layout.width > 4096);
        assert!(layout.width <= 16_384);
        assert!(layout.height <= 16_384);
        assert!(layout.width * layout.height >= layout.total_knots);
    }
}
