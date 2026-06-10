use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::basis_bank_motion::BasisBankMotionSet;
use crate::deformation_gpu::WORKGROUP_SIZE;
use crate::texture::Texture;
use crate::utils::transmute_slice;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MotionUniform {
    time01: f32,
    splat_count: u32,
    gaussian_tex_width: u32,
    knot_count: u32,
    top_k: u32,
    _pad0: [u32; 3],
}

pub struct GpuBasisBankMotionRuntime {
    compute_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
    _base_texture: Texture,
    _basis_knots_buffer: wgpu::Buffer,
    _basis_ids_buffer: wgpu::Buffer,
    _weights_buffer: wgpu::Buffer,
    output_texture: Texture,
    splat_count: u32,
    gaussian_tex_width: u32,
    knot_count: u32,
    top_k: u32,
    global_basis_count: u32,
}

impl GpuBasisBankMotionRuntime {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        motion: &BasisBankMotionSet,
        gaussian_tex_width: u32,
        gaussian_tex_height: u32,
        base_tex_data: &[u32],
    ) -> Result<Self, String> {
        if motion.total_splats > u32::MAX as usize {
            return Err("basis-bank splat count exceeds u32".to_string());
        }
        if motion.global_basis_count == 0 || motion.global_basis_count > u32::MAX as usize {
            return Err(format!(
                "invalid basis-bank global basis count {}",
                motion.global_basis_count
            ));
        }
        if motion.meta.exported_knot_count == 0
            || motion.meta.exported_knot_count > u32::MAX as usize
        {
            return Err(format!(
                "invalid basis-bank knot count {}",
                motion.meta.exported_knot_count
            ));
        }
        if motion.meta.top_k == 0 || motion.meta.top_k > u32::MAX as usize {
            return Err(format!("invalid basis-bank top_k {}", motion.meta.top_k));
        }

        let expected_basis_values = motion
            .global_basis_count
            .checked_mul(motion.meta.exported_knot_count)
            .and_then(|v| v.checked_mul(3))
            .ok_or_else(|| "basis-bank basis value count overflow".to_string())?;
        if motion.global_basis_knots.len() != expected_basis_values {
            return Err(format!(
                "basis-bank basis value mismatch: got {}, expected {}",
                motion.global_basis_knots.len(),
                expected_basis_values
            ));
        }
        let expected_coeff_values = motion
            .total_splats
            .checked_mul(motion.meta.top_k)
            .ok_or_else(|| "basis-bank coefficient count overflow".to_string())?;
        if motion.global_basis_ids.len() != expected_coeff_values
            || motion.global_weights.len() != expected_coeff_values
        {
            return Err(format!(
                "basis-bank coefficient length mismatch: ids={}, weights={}, expected {}",
                motion.global_basis_ids.len(),
                motion.global_weights.len(),
                expected_coeff_values
            ));
        }

        let base_texture = create_uint_texture(
            device,
            queue,
            "basis_bank_base_gaussian_texture",
            gaussian_tex_width,
            gaussian_tex_height,
            base_tex_data,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        let output_texture = create_uint_texture(
            device,
            queue,
            "basis_bank_output_gaussian_texture",
            gaussian_tex_width,
            gaussian_tex_height,
            base_tex_data,
            wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST,
        );

        let basis_knots = pack_basis_knots(motion.global_basis_knots.as_slice());
        let basis_knots_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("basis_bank_knots"),
            contents: bytemuck::cast_slice(basis_knots.as_slice()),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let basis_ids_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("basis_bank_ids"),
            contents: bytemuck::cast_slice(motion.global_basis_ids.as_slice()),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let weights_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("basis_bank_weights"),
            contents: bytemuck::cast_slice(motion.global_weights.as_slice()),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let uniform = MotionUniform {
            time01: 0.0,
            splat_count: motion.total_splats as u32,
            gaussian_tex_width,
            knot_count: motion.meta.exported_knot_count as u32,
            top_k: motion.meta.top_k as u32,
            _pad0: [0; 3],
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("basis_bank_motion_uniform"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = Self::create_bind_group_layout(device);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("basis_bank_motion_bind_group"),
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
                    resource: basis_knots_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: basis_ids_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: weights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&output_texture.view),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("basis_bank_motion_compute_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("basis_bank_motion_compute.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("basis_bank_motion_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("basis_bank_motion_compute_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        crate::log!(
            "Basis-bank GPU runtime: splats={}, global_basis={}, knots={}, top_k={}, basis_buffer={:.1} MiB, coeff_buffers={:.1} MiB",
            motion.total_splats,
            motion.global_basis_count,
            motion.meta.exported_knot_count,
            motion.meta.top_k,
            basis_knots.len() as f64 * 16.0 / (1024.0 * 1024.0),
            expected_coeff_values as f64 * 8.0 / (1024.0 * 1024.0)
        );

        Ok(Self {
            compute_pipeline,
            bind_group,
            uniform_buffer,
            _base_texture: base_texture,
            _basis_knots_buffer: basis_knots_buffer,
            _basis_ids_buffer: basis_ids_buffer,
            _weights_buffer: weights_buffer,
            output_texture,
            splat_count: motion.total_splats as u32,
            gaussian_tex_width,
            knot_count: motion.meta.exported_knot_count as u32,
            top_k: motion.meta.top_k as u32,
            global_basis_count: motion.global_basis_count as u32,
        })
    }

    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        time01: f32,
    ) -> Result<f64, String> {
        let uniform = MotionUniform {
            time01,
            splat_count: self.splat_count,
            gaussian_tex_width: self.gaussian_tex_width,
            knot_count: self.knot_count,
            top_k: self.top_k,
            _pad0: [0; 3],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniform));

        let start = crate::utils::get_time_milliseconds();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("basis_bank_motion_compute_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("basis_bank_motion_compute_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.compute_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let workgroups = self.splat_count.div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
        Ok(crate::utils::get_time_milliseconds() - start)
    }

    pub fn output_texture(&self) -> &Texture {
        &self.output_texture
    }

    pub fn knot_count(&self) -> u32 {
        self.knot_count
    }

    pub fn top_k(&self) -> u32 {
        self.top_k
    }

    pub fn global_basis_count(&self) -> u32 {
        self.global_basis_count
    }

    fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("basis_bank_motion_bind_group_layout"),
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
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                storage_buffer_layout_entry(2),
                storage_buffer_layout_entry(3),
                storage_buffer_layout_entry(4),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
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
}

fn storage_buffer_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

fn pack_basis_knots(knots: &[f32]) -> Vec<[f32; 4]> {
    let mut packed = Vec::with_capacity(knots.len() / 3);
    for chunk in knots.chunks_exact(3) {
        packed.push([chunk[0], chunk[1], chunk[2], 0.0]);
    }
    packed
}

fn create_uint_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    width: u32,
    height: u32,
    data: &[u32],
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
        transmute_slice(data),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basis_motion_uniform_is_32_bytes() {
        assert_eq!(std::mem::size_of::<MotionUniform>(), 32);
    }

    #[test]
    fn packs_basis_knots_as_vec4_rows() {
        let packed = pack_basis_knots(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(packed, vec![[1.0, 2.0, 3.0, 0.0], [4.0, 5.0, 6.0, 0.0]]);
    }
}
