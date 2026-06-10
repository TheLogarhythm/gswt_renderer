use wgpu::util::DeviceExt;
use wgpu::BufferAddress;

use crate::basis_bank_motion_gpu::GpuBasisBankMotionRuntime;
use crate::camera::{Camera, CameraUniforms};
use crate::catmull_rom_motion::MotionMode;
use crate::catmull_rom_motion_gpu::{
    GpuCatmullRomMotionRuntime, MotionCompatibilityPending, MotionTextureComparePending,
};
use crate::deformation::DeformationNetwork;
use crate::deformation_gpu::{GpuDeformationRuntime, DEFORMATION_DEBUG_VOLUME};
use crate::log;
use crate::motion::{pack_motion_spline_knots, MOTION_PACKED_KNOT_COUNT};
use crate::structure::*;
use crate::texture::Texture;
use crate::utils::*;

#[derive(Clone, Copy, PartialEq, Eq)]
enum DeformationDebugMode {
    Full,
    Identity,
    Volume,
}

const SOURCE_MOTION_FPS: f32 = 30.0;

#[derive(Clone, Copy)]
struct DeformationDebugConfig {
    mode: DeformationDebugMode,
    volume_res: u32,
    volume_keys: u32,
}

impl DeformationDebugConfig {
    const DEFAULT_VOLUME_RES: u32 = 64;
    const DEFAULT_VOLUME_KEYS: u32 = 25;

    fn from_url_query() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Self {
                mode: DeformationDebugMode::Volume,
                volume_res: Self::DEFAULT_VOLUME_RES,
                volume_keys: Self::DEFAULT_VOLUME_KEYS,
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            let mut config = Self {
                mode: DeformationDebugMode::Volume,
                volume_res: Self::DEFAULT_VOLUME_RES,
                volume_keys: Self::DEFAULT_VOLUME_KEYS,
            };
            let global = web_sys::js_sys::global();
            let search = web_sys::js_sys::Reflect::get(
                &global,
                &wasm_bindgen::JsValue::from_str("location"),
            )
            .ok()
            .and_then(|location| {
                web_sys::js_sys::Reflect::get(&location, &wasm_bindgen::JsValue::from_str("search"))
                    .ok()
            })
            .and_then(|value| value.as_string())
            .unwrap_or_default();
            for part in search.trim_start_matches('?').split('&') {
                match part {
                    "deform_mode=identity" | "deform_debug=identity" => {
                        config.mode = DeformationDebugMode::Identity
                    }
                    "deform_mode=volume" | "deform_debug=volume" => {
                        config.mode = DeformationDebugMode::Volume
                    }
                    "deform_mode=hexplane_mlp" | "deform_debug=full" => {
                        config.mode = DeformationDebugMode::Full
                    }
                    _ => {
                        if let Some(raw) = part.strip_prefix("deform_volume_res=") {
                            if let Ok(value) = raw.parse::<u32>() {
                                config.volume_res = value.max(2);
                            }
                        } else if let Some(raw) = part
                            .strip_prefix("deform_volume_keys=")
                            .or_else(|| part.strip_prefix("deform_key_frames="))
                        {
                            if let Ok(value) = raw.parse::<u32>() {
                                config.volume_keys = value.max(1);
                            }
                        }
                    }
                }
            }
            config
        }
    }
}

impl DeformationDebugMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Full => "hexplane_mlp",
            Self::Identity => "identity",
            Self::Volume => "volume",
        }
    }

    fn shader_value(self) -> u32 {
        match self {
            Self::Full => 0,
            Self::Identity => 1,
            Self::Volume => 2,
        }
    }
}

pub struct GSWTRenderer {
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,

    camera_uniforms_buffer: wgpu::Buffer,
    scene_uniforms_buffer: wgpu::Buffer,
    gaussian_texture: Texture,
    scene_bind_group_layout: wgpu::BindGroupLayout,
    scene_bind_group: Option<wgpu::BindGroup>,

    tile_uniforms_buffer: wgpu::Buffer,
    tile_bind_group: wgpu::BindGroup,

    gs_index_buffer: wgpu::Buffer,
    map_id_buffer: wgpu::Buffer,
    lod_id_buffer: wgpu::Buffer,
    buffer_base_data: Vec<Vec<Vec<BufferDataValue>>>,

    gaussian_tex_width: u32,
    gaussian_tex_height: u32,
    base_tex_data: Vec<u32>,
    work_tex_data: Vec<u32>,
    base_tile_means: Vec<[f32; 3]>,
    base_scales: Vec<[f32; 3]>,
    deformation_network: Option<DeformationNetwork>,
    deformation_gpu_runtime: Option<GpuDeformationRuntime>,
    basis_bank_runtime: Option<GpuBasisBankMotionRuntime>,
    compatibility_volume_runtime: Option<GpuDeformationRuntime>,
    motion_compatibility_pending: Option<MotionCompatibilityPending>,
    motion_texture_compare_pending: Option<MotionTextureComparePending>,
    catmull_rom_runtime: Option<GpuCatmullRomMotionRuntime>,
    merged_orig_means: Option<Vec<[f32; 3]>>,
    merged_orig_quats: Option<Vec<[f32; 4]>>,
    deformation_ready: bool,
    deformation_duration: f32,
    deformation_log_frame: u32,
    deformation_debug_mode: DeformationDebugMode,
    deformation_volume_res: u32,
    deformation_volume_keys: u32,
    motion_mode: MotionMode,

    user_data: UserData,
}
impl GSWTRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
        preload_data: PreloadData,
    ) -> Self {
        let scene_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Uint,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("scene_bind_group_layout"),
            });

        // TODO: change back to uniform
        let tile_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: true,
                        min_binding_size: std::num::NonZeroU64::new(256),
                    },
                    count: None,
                }],
                label: Some("tile_bind_group_layout"),
            });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gswt.wgsl").into()),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&scene_bind_group_layout, &tile_bind_group_layout],
                push_constant_ranges: &[],
            });

        let alpha_blend = Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    Vertex2D::desc(),
                    wgpu::VertexBufferLayout {
                        array_stride: 4 as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![1 => Uint32],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: 4 as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![2 => Uint32],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: 4 as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![3 => Uint32],
                    },
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: alpha_blend,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList, // 1.
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw, // 2.
                cull_mode: None,
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Texture::DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Vertex buffer
        let vertices = &mut [
            // quad
            Vertex2D {
                position: [-2.0, -2.0],
            },
            Vertex2D {
                position: [2.0, -2.0],
            },
            Vertex2D {
                position: [2.0, 2.0],
            },
            Vertex2D {
                position: [2.0, 2.0],
            },
            Vertex2D {
                position: [-2.0, 2.0],
            },
            Vertex2D {
                position: [-2.0, -2.0],
            },
        ];
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Scene global data
        let camera_uniforms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera Uniforms Buffer"),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            size: std::mem::size_of::<CameraUniforms>() as u64,
            mapped_at_creation: false,
        });
        let scene_uniforms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scene Uniforms Buffer"),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            size: std::mem::size_of::<SceneUniforms>() as u64,
            mapped_at_creation: false,
        });
        let mut gaussian_texture = Texture::from_bytes(
            device,
            queue,
            transmute_slice::<_, u8>(preload_data.tile_splats_merged.tex_data.as_slice()),
            preload_data.tile_splats_merged.tex_width as u32,
            preload_data.tile_splats_merged.tex_height as u32,
            16,
            wgpu::TextureFormat::Rgba32Uint,
            wgpu::FilterMode::Nearest,
            wgpu::AddressMode::ClampToEdge,
            Some("Gaussian Texture"),
        )
        .unwrap();
        let gaussian_tex_width = preload_data.tile_splats_merged.tex_width as u32;
        let gaussian_tex_height = preload_data.tile_splats_merged.tex_height as u32;
        let base_tex_data = preload_data.tile_splats_merged.tex_data.clone();
        let mut base_tile_means = Vec::with_capacity(preload_data.tile_splats_merged.splat_count);
        {
            let tex_f: &[f32] = transmute_slice(base_tex_data.as_slice());
            for i in 0..preload_data.tile_splats_merged.splat_count {
                let index_f = 8 * i;
                base_tile_means.push([tex_f[index_f], tex_f[index_f + 1], tex_f[index_f + 2]]);
            }
        }
        let mut base_scales = Vec::with_capacity(preload_data.tile_splats_merged.splat_count);
        {
            let f_buffer: &[f32] =
                transmute_slice(preload_data.tile_splats_merged.buffer.as_slice());
            for i in 0..preload_data.tile_splats_merged.splat_count {
                base_scales.push([
                    f_buffer[8 * i + 3],
                    f_buffer[8 * i + 4],
                    f_buffer[8 * i + 5],
                ]);
            }
        }
        let work_tex_data = base_tex_data.clone();
        let deformation_network = preload_data.deformation_network;
        let basis_bank_motion = preload_data.basis_bank_motion;
        let catmull_rom_motion = preload_data.catmull_rom_motion;
        let merged_orig_means = preload_data.merged_orig_means;
        let merged_orig_quats = preload_data.merged_orig_quats;
        let requested_motion_mode = MotionMode::from_url_query();
        log!("motion_mode={}", requested_motion_mode.as_str());
        let deformation_debug_config = DeformationDebugConfig::from_url_query();
        let deformation_debug_mode = deformation_debug_config.mode;
        let mut deformation_gpu_runtime: Option<GpuDeformationRuntime> = None;
        let mut basis_bank_runtime: Option<GpuBasisBankMotionRuntime> = None;
        let mut catmull_rom_runtime: Option<GpuCatmullRomMotionRuntime> = None;
        let force_cpu_deformation = std::env::var("GSWT_FORCE_CPU_DEFORMATION")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let mut deformation_ready = false;
        let mut deformation_duration = 1.0_f32;
        let mut active_motion_mode = MotionMode::Static;
        let network_motion_duration = deformation_network
            .as_ref()
            .map(|net| source_motion_duration_from_frame_count(net.metadata().n_time_frames));
        let catmull_rom_metadata_duration = catmull_rom_motion
            .as_ref()
            .and_then(|motion| valid_motion_duration(motion.meta.duration_seconds));
        let catmull_rom_playback_duration =
            catmull_rom_motion_duration(catmull_rom_metadata_duration, network_motion_duration);

        if matches!(
            requested_motion_mode,
            MotionMode::Auto | MotionMode::BasisBank
        ) {
            if let Some(motion) = basis_bank_motion.as_ref() {
                match GpuBasisBankMotionRuntime::new(
                    device,
                    queue,
                    motion,
                    gaussian_tex_width,
                    gaussian_tex_height,
                    base_tex_data.as_slice(),
                ) {
                    Ok(runtime) => {
                        gaussian_texture = runtime.output_texture().clone();
                        deformation_ready = true;
                        deformation_duration = catmull_rom_playback_duration;
                        active_motion_mode = MotionMode::BasisBank;
                        log!(
                            "GSWTRenderer::new(): motion backend=GPU basis-bank (splats={}, global_basis={}, basis_per_lod={}, top_k={}, knots={}, source_knot_count={}, loop_closure_knots={}, motion_teacher={}, volume_res={:?}, volume_key_count={:?}, duration={:.3}s)",
                            motion.total_splats,
                            motion.global_basis_count,
                            motion.meta.basis_count,
                            motion.meta.top_k,
                            motion.meta.exported_knot_count,
                            motion.meta.source_knot_count,
                            motion.meta.loop_closure_knots,
                            motion.meta.motion_teacher,
                            motion.meta.volume_res,
                            motion.meta.volume_key_count,
                            deformation_duration
                        );
                        basis_bank_runtime = Some(runtime);
                    }
                    Err(err) => {
                        log!(
                            "GSWTRenderer::new(): basis-bank backend unavailable: {}",
                            err
                        );
                    }
                }
            } else if requested_motion_mode == MotionMode::BasisBank {
                log!(
                    "GSWTRenderer::new(): motion_mode=basis_bank requested, but no valid basis-bank motion was loaded."
                );
            }
        }

        if matches!(
            requested_motion_mode,
            MotionMode::Auto | MotionMode::CatmullRom
        ) && active_motion_mode != MotionMode::BasisBank
        {
            if let Some(motion) = catmull_rom_motion.as_ref() {
                match GpuCatmullRomMotionRuntime::new(
                    device,
                    queue,
                    motion,
                    gaussian_tex_width,
                    gaussian_tex_height,
                    base_tex_data.as_slice(),
                ) {
                    Ok(runtime) => {
                        gaussian_texture = runtime.output_texture().clone();
                        deformation_ready = true;
                        deformation_duration = catmull_rom_playback_duration;
                        active_motion_mode = MotionMode::CatmullRom;
                        log!(
                            "GSWTRenderer::new(): motion backend=GPU Catmull-Rom (splats={}, knots={}, time_sampling={}, sample_time_grid={}, motion_teacher={}, volume_res={:?}, volume_key_count={:?}, source_knot_count={:?}, exported_knot_count={:?}, loop_closure_knots={:?}, loop_closure_method={:?}, included_lods={:?}, duration={:.3}s)",
                            motion.total_splats,
                            motion.meta.knot_count,
                            motion.meta.time_sampling.as_str(),
                            motion.meta.sample_time_grid.as_str(),
                            motion.meta.motion_teacher.as_str(),
                            motion.meta.volume_res,
                            motion.meta.volume_key_count,
                            motion.meta.source_knot_count,
                            motion.meta.exported_knot_count,
                            motion.meta.loop_closure_knots,
                            motion.meta.loop_closure_method,
                            motion.meta.include_lods,
                            deformation_duration
                        );
                        catmull_rom_runtime = Some(runtime);
                    }
                    Err(err) => {
                        log!(
                            "GSWTRenderer::new(): Catmull-Rom backend unavailable: {}",
                            err
                        );
                    }
                }
            } else if requested_motion_mode == MotionMode::CatmullRom {
                log!(
                    "GSWTRenderer::new(): motion_mode=catmull_rom requested, but no valid Catmull-Rom motion was loaded."
                );
            }
        }

        let allow_network_fallback = !matches!(
            active_motion_mode,
            MotionMode::CatmullRom | MotionMode::BasisBank
        ) && requested_motion_mode != MotionMode::Static;
        if allow_network_fallback {
            if deformation_debug_mode == DeformationDebugMode::Volume {
                log!(
                    "deform_mode=volume volume_res={} key_frames={}",
                    deformation_debug_config.volume_res,
                    deformation_debug_config.volume_keys
                );
            } else {
                log!("deform_mode={}", deformation_debug_mode.as_str());
            }
        }

        if allow_network_fallback {
            if let Some(net) = deformation_network.as_ref() {
                let splat_count = preload_data.tile_splats_merged.splat_count;
                match (merged_orig_means.as_ref(), merged_orig_quats.as_ref()) {
                    (Some(orig_means), Some(orig_quats))
                        if orig_means.len() == splat_count && orig_quats.len() == splat_count =>
                    {
                        deformation_ready = true;
                        deformation_duration = network_motion_duration.unwrap_or_else(|| {
                            source_motion_duration_from_frame_count(net.metadata().n_time_frames)
                        });
                        active_motion_mode = MotionMode::DeformationNetwork;
                        if force_cpu_deformation {
                            log!(
                                "GSWTRenderer::new(): GSWT_FORCE_CPU_DEFORMATION enabled; using CPU fallback."
                            );
                            log!(
                                "GSWTRenderer::new(): deformation backend=CPU (splats={}, duration={:.3}s)",
                                splat_count,
                                deformation_duration
                            );
                        } else {
                            match GpuDeformationRuntime::new(
                                device,
                                queue,
                                net,
                                splat_count,
                                gaussian_tex_width,
                                gaussian_tex_height,
                                base_tex_data.as_slice(),
                                base_tile_means.as_slice(),
                                base_scales.as_slice(),
                                orig_means.as_slice(),
                                orig_quats.as_slice(),
                                deformation_debug_mode.shader_value(),
                                deformation_debug_config.volume_res,
                                deformation_debug_config.volume_keys,
                            ) {
                                Ok(runtime) => {
                                    gaussian_texture = runtime.output_texture().clone();
                                    deformation_gpu_runtime = Some(runtime);
                                    log!(
                                        "GSWTRenderer::new(): deformation backend=GPU (splats={}, duration={:.3}s)",
                                        splat_count,
                                        deformation_duration
                                    );
                                }
                                Err(err) => {
                                    log!(
                                        "GSWTRenderer::new(): GPU deformation unavailable, fallback to CPU: {}",
                                        err
                                    );
                                    log!(
                                        "GSWTRenderer::new(): deformation backend=CPU (splats={}, duration={:.3}s)",
                                        splat_count,
                                        deformation_duration
                                    );
                                }
                            }
                        }
                    }
                    (Some(orig_means), Some(orig_quats)) => {
                        log!(
                            "GSWTRenderer::new(): deformation disabled due to length mismatch: splats={}, orig_means={}, orig_quats={}",
                            splat_count,
                            orig_means.len(),
                            orig_quats.len()
                        );
                    }
                    _ => {
                        log!(
                            "GSWTRenderer::new(): deformation disabled because merged orig inputs are missing."
                        );
                    }
                }
            } else if requested_motion_mode == MotionMode::DeformationNetwork {
                log!(
                    "GSWTRenderer::new(): motion_mode=deformation_network requested, but deformation_weights.bin was not loaded."
                );
            }
        } else if requested_motion_mode == MotionMode::Static {
            log!("GSWTRenderer::new(): static motion mode selected.");
        }

        let tile_uniforms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Per-instance Tile Uniforms Storage Buffer"),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            size: 20_000 * u64::max(256, std::mem::size_of::<TileUniforms>() as u64),
            mapped_at_creation: false,
        });
        let tile_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &tile_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &tile_uniforms_buffer,
                    offset: 0,
                    size: std::num::NonZeroU64::new(256),
                }),
            }],
            label: Some("tile_bind_group"),
        });

        // Instance buffers
        let gs_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(format!("gs_index_buffer").as_str()),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            size: 10_000_000 * 4,
            mapped_at_creation: false,
        });
        let map_id_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(format!("map_id_buffer").as_str()),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            size: 10_000_000 * 4,
            mapped_at_creation: false,
        });
        let lod_id_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(format!("lod_id_buffer").as_str()),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            size: 10_000_000 * 4,
            mapped_at_creation: false,
        });

        // Preloaded instance buffers
        let mut buffer_base_data = Vec::with_capacity(preload_data.tile_base_data.len());
        for i in 0..preload_data.tile_base_data.len() {
            let tile_data_vec = &preload_data.tile_base_data[i];
            let mut tile_buf_vec: Vec<Vec<BufferDataValue>> =
                Vec::with_capacity(tile_data_vec.len());
            for j in 0..tile_data_vec.len() {
                let view_data_vec = &tile_data_vec[j];
                let mut view_buf_vec: Vec<BufferDataValue> =
                    Vec::with_capacity(view_data_vec.len());
                for k in 0..view_data_vec.len() {
                    let base_data = &view_data_vec[k];

                    let gs_index_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some(format!("gs_index_buffer [{i}.{j}.{k}]").as_str()),
                            contents: bytemuck::cast_slice(base_data.gs_index.as_slice()),
                            usage: wgpu::BufferUsages::VERTEX,
                        });

                    let lod_id_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some(format!("lod_id_buffer [{i}.{j}.{k}]").as_str()),
                            contents: bytemuck::cast_slice(base_data.gs_lod_id.as_slice()),
                            usage: wgpu::BufferUsages::VERTEX,
                        });

                    let buffer_data = BufferDataValue {
                        splat_count: base_data.splat_count as u32,
                        gs_index_buffer,
                        map_id_buffer: None,
                        lod_id_buffer: Some(lod_id_buffer),
                    };
                    view_buf_vec.push(buffer_data);
                }
                tile_buf_vec.push(view_buf_vec);
            }
            buffer_base_data.push(tile_buf_vec);
        }

        Self {
            render_pipeline,
            vertex_buffer,

            camera_uniforms_buffer,
            scene_uniforms_buffer,
            gaussian_texture,
            scene_bind_group_layout,
            scene_bind_group: None,

            tile_uniforms_buffer,
            tile_bind_group,

            gs_index_buffer,
            map_id_buffer,
            lod_id_buffer,
            buffer_base_data,

            gaussian_tex_width,
            gaussian_tex_height,
            base_tex_data,
            work_tex_data,
            base_tile_means,
            base_scales,
            deformation_network,
            deformation_gpu_runtime,
            basis_bank_runtime,
            compatibility_volume_runtime: None,
            motion_compatibility_pending: None,
            motion_texture_compare_pending: None,
            catmull_rom_runtime,
            merged_orig_means,
            merged_orig_quats,
            deformation_ready,
            deformation_duration,
            deformation_log_frame: 0,
            deformation_debug_mode,
            deformation_volume_res: deformation_debug_config.volume_res,
            deformation_volume_keys: deformation_debug_config.volume_keys,
            motion_mode: active_motion_mode,

            user_data: UserData::new(),
        }
    }

    pub fn configure(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        user_data: &UserData,
        render_data: &RenderData,
    ) {
        self.user_data = user_data.clone();

        let mut group_entries: Vec<wgpu::BindGroupEntry> = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: self.camera_uniforms_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: self.scene_uniforms_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&self.gaussian_texture.view),
            },
        ];

        let height_map: Texture;
        height_map = Texture::from_bytes(
            device,
            queue,
            bytemuck::cast_slice(self.user_data.height_map.as_slice()),
            self.user_data.height_map_wh.x as u32,
            self.user_data.height_map_wh.y as u32,
            4,
            wgpu::TextureFormat::R32Float,
            wgpu::FilterMode::Linear,
            wgpu::AddressMode::Repeat,
            Some("Dummy Height Map Texture"),
        )
        .unwrap();
        group_entries.push(wgpu::BindGroupEntry {
            binding: 3,
            resource: wgpu::BindingResource::TextureView(&height_map.view),
        });
        group_entries.push(wgpu::BindGroupEntry {
            binding: 4,
            resource: wgpu::BindingResource::Sampler(height_map.sampler.as_ref().unwrap()),
        });

        let scene_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.scene_bind_group_layout,
            entries: &group_entries,
            label: Some("scene_bind_group"),
        });

        self.scene_bind_group = Some(scene_bind_group);
    }

    pub fn has_deformation(&self) -> bool {
        self.deformation_ready
    }

    pub fn deformation_duration(&self) -> f32 {
        self.deformation_duration
    }

    pub fn uses_periodic_motion(&self) -> bool {
        if self.basis_bank_runtime.is_some() {
            return true;
        }
        self.catmull_rom_runtime
            .as_ref()
            .map(GpuCatmullRomMotionRuntime::uses_periodic_times)
            .unwrap_or(false)
    }

    pub fn active_motion_mode(&self) -> MotionMode {
        self.motion_mode
    }

    pub fn catmull_rom_knot_count(&self) -> Option<u32> {
        if let Some(runtime) = self.basis_bank_runtime.as_ref() {
            return Some(runtime.knot_count());
        }
        self.catmull_rom_runtime
            .as_ref()
            .map(GpuCatmullRomMotionRuntime::knot_count)
    }

    pub fn catmull_rom_uses_volume_key_times(&self) -> bool {
        if self.basis_bank_runtime.is_some() {
            return false;
        }
        self.catmull_rom_runtime
            .as_ref()
            .map(GpuCatmullRomMotionRuntime::uses_volume_key_times)
            .unwrap_or(false)
    }

    pub fn basis_bank_basis_count(&self) -> Option<u32> {
        self.basis_bank_runtime
            .as_ref()
            .map(GpuBasisBankMotionRuntime::global_basis_count)
    }

    pub fn basis_bank_top_k(&self) -> Option<u32> {
        self.basis_bank_runtime
            .as_ref()
            .map(GpuBasisBankMotionRuntime::top_k)
    }

    pub fn volume_key_count(&self) -> Option<u32> {
        self.deformation_gpu_runtime
            .as_ref()
            .or(self.compatibility_volume_runtime.as_ref())
            .map(GpuDeformationRuntime::volume_key_count)
            .or(Some(self.deformation_volume_keys))
    }

    fn ensure_compatibility_volume_runtime(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(), String> {
        if self.compatibility_volume_runtime.is_some() {
            return Ok(());
        }
        let net = self.deformation_network.as_ref().ok_or_else(|| {
            "deformation network is unavailable for volume comparison".to_string()
        })?;
        let orig_means = self
            .merged_orig_means
            .as_ref()
            .ok_or_else(|| "original means are unavailable for volume comparison".to_string())?;
        let orig_quats = self.merged_orig_quats.as_ref().ok_or_else(|| {
            "original quaternions are unavailable for volume comparison".to_string()
        })?;
        self.compatibility_volume_runtime = Some(GpuDeformationRuntime::new(
            device,
            queue,
            net,
            self.base_tile_means.len(),
            self.gaussian_tex_width,
            self.gaussian_tex_height,
            self.base_tex_data.as_slice(),
            self.base_tile_means.as_slice(),
            self.base_scales.as_slice(),
            orig_means.as_slice(),
            orig_quats.as_slice(),
            DEFORMATION_DEBUG_VOLUME,
            self.deformation_volume_res,
            self.deformation_volume_keys,
        )?);
        Ok(())
    }

    pub fn start_motion_compatibility_compare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scope: MotionCompatibilityScope,
        selected_knot: u32,
    ) -> Result<(), String> {
        if self.motion_compatibility_pending.is_some() {
            return Err("motion compatibility comparison is already running".to_string());
        }
        if self.catmull_rom_runtime.is_none() {
            return Err("Catmull-Rom backend is not active".to_string());
        }
        self.ensure_compatibility_volume_runtime(device, queue)?;
        let cat_runtime = self
            .catmull_rom_runtime
            .as_ref()
            .ok_or_else(|| "Catmull-Rom backend is not active".to_string())?;
        let volume_runtime = self
            .compatibility_volume_runtime
            .as_ref()
            .ok_or_else(|| "volume comparison runtime failed to initialize".to_string())?;
        self.motion_compatibility_pending = Some(cat_runtime.compare_to_volume_async(
            device,
            queue,
            volume_runtime,
            scope,
            selected_knot,
        )?);
        Ok(())
    }

    pub fn start_motion_texture_compare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        selected_knot: u32,
    ) -> Result<(), String> {
        if self.motion_texture_compare_pending.is_some() {
            return Err("motion texture comparison is already running".to_string());
        }
        if self.catmull_rom_runtime.is_none() {
            return Err("Catmull-Rom backend is not active".to_string());
        }
        self.ensure_compatibility_volume_runtime(device, queue)?;
        let cat_runtime = self
            .catmull_rom_runtime
            .as_ref()
            .ok_or_else(|| "Catmull-Rom backend is not active".to_string())?;
        let volume_runtime = self
            .compatibility_volume_runtime
            .as_ref()
            .ok_or_else(|| "volume comparison runtime failed to initialize".to_string())?;
        let catmull_rom_time01 = cat_runtime.knot_preview_time(selected_knot);
        let volume_time01 = cat_runtime.volume_comparison_time(selected_knot);
        self.motion_texture_compare_pending =
            Some(cat_runtime.compare_final_means_to_volume_async(
                device,
                queue,
                volume_runtime,
                catmull_rom_time01,
                volume_time01,
            )?);
        Ok(())
    }

    pub fn poll_motion_compatibility_result(
        &mut self,
        device: &wgpu::Device,
    ) -> Option<Result<MotionCompatibilityResult, String>> {
        let _ = device.poll(wgpu::PollType::Poll);
        let result = self
            .motion_compatibility_pending
            .as_mut()
            .and_then(MotionCompatibilityPending::take_result);
        if result.is_some() {
            self.motion_compatibility_pending = None;
        }
        result
    }

    pub fn poll_motion_texture_compare_result(
        &mut self,
        device: &wgpu::Device,
    ) -> Option<Result<MotionTextureCompareResult, String>> {
        let _ = device.poll(wgpu::PollType::Poll);
        let result = self
            .motion_texture_compare_pending
            .as_mut()
            .and_then(MotionTextureComparePending::take_result);
        if result.is_some() {
            self.motion_texture_compare_pending = None;
        }
        result
    }

    pub fn update_deformation(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        time01: f32,
        apply_network_delta_rot: bool,
    ) {
        if !self.deformation_ready {
            return;
        }

        self.deformation_log_frame = self.deformation_log_frame.wrapping_add(1);

        if let Some(runtime) = self.basis_bank_runtime.as_ref() {
            match runtime.dispatch(device, queue, time01) {
                Ok(elapsed) => {
                    if self.deformation_log_frame % 15 == 0 {
                        log!("motion_mode=basis_bank gpu_submit={:.3}ms", elapsed);
                    }
                }
                Err(err) => {
                    log!(
                        "GSWTRenderer::update_deformation(): basis-bank GPU backend failed: {}",
                        err
                    );
                }
            }
            return;
        }

        if let Some(runtime) = self.catmull_rom_runtime.as_ref() {
            let start = get_time_milliseconds();
            if let Err(err) = runtime.dispatch(device, queue, time01) {
                log!(
                    "GSWTRenderer::update_deformation(): Catmull-Rom GPU backend failed: {}",
                    err
                );
                return;
            }
            let elapsed = get_time_milliseconds() - start;
            if self.deformation_log_frame % 15 == 0 {
                log!("motion_mode=catmull_rom gpu_submit={:.3}ms", elapsed);
            }
            return;
        }

        let gpu_result = if let Some(runtime) = self.deformation_gpu_runtime.as_ref() {
            let start = get_time_milliseconds();
            let result = runtime.dispatch(
                device,
                queue,
                time01.clamp(0.0, 1.0),
                apply_network_delta_rot,
            );
            let elapsed = get_time_milliseconds() - start;
            if self.deformation_log_frame % 15 == 0 {
                log!(
                    "deform_mode={} deform_gpu_submit={:.3}ms",
                    self.deformation_debug_mode.as_str(),
                    elapsed
                );
            }
            Some(result)
        } else {
            None
        };

        match gpu_result {
            Some(Ok(())) => return,
            Some(Err(err)) => {
                log!(
                    "GSWTRenderer::update_deformation(): GPU backend failed, switching to CPU fallback: {}",
                    err
                );
                self.deformation_gpu_runtime = None;
            }
            None => {}
        }

        let net = if let Some(net) = self.deformation_network.as_ref() {
            net
        } else {
            return;
        };
        let orig_means = if let Some(orig_means) = self.merged_orig_means.as_ref() {
            orig_means
        } else {
            return;
        };
        let orig_quats = if let Some(orig_quats) = self.merged_orig_quats.as_ref() {
            orig_quats
        } else {
            return;
        };

        let (new_tile_means, new_quats) = match net.deform_batch(
            orig_means.as_slice(),
            self.base_tile_means.as_slice(),
            orig_quats.as_slice(),
            time01.clamp(0.0, 1.0),
        ) {
            Ok(v) => v,
            Err(err) => {
                log!(
                    "GSWTRenderer::update_deformation(): deformation failed: {}",
                    err
                );
                return;
            }
        };

        self.work_tex_data
            .copy_from_slice(self.base_tex_data.as_slice());
        {
            let tex_f: &mut [f32] = transmute_slice_mut(self.work_tex_data.as_mut_slice());
            for i in 0..new_tile_means.len() {
                let index_f = 8 * i;
                tex_f[index_f + 0] = new_tile_means[i][0];
                tex_f[index_f + 1] = new_tile_means[i][1];
                tex_f[index_f + 2] = new_tile_means[i][2];
            }
        }
        if apply_network_delta_rot {
            for i in 0..new_quats.len() {
                let index_f = 8 * i;
                let cov = Self::pack_covariance(self.base_scales[i], new_quats[i]);
                self.work_tex_data[index_f + 4] = cov[0];
                self.work_tex_data[index_f + 5] = cov[1];
                self.work_tex_data[index_f + 6] = cov[2];
            }
        }

        let texture_size = wgpu::Extent3d {
            width: self.gaussian_tex_width,
            height: self.gaussian_tex_height,
            depth_or_array_layers: 1,
        };
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.gaussian_texture.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            transmute_slice::<_, u8>(self.work_tex_data.as_slice()),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.gaussian_tex_width * 16),
                rows_per_image: Some(self.gaussian_tex_height),
            },
            texture_size,
        );
    }

    fn pack_covariance(scale: [f32; 3], rot: [f32; 4]) -> [u32; 3] {
        let r = Mat3::new(
            1.0 - 2.0 * (rot[2] * rot[2] + rot[3] * rot[3]),
            2.0 * (rot[1] * rot[2] + rot[0] * rot[3]),
            2.0 * (rot[1] * rot[3] - rot[0] * rot[2]),
            2.0 * (rot[1] * rot[2] - rot[0] * rot[3]),
            1.0 - 2.0 * (rot[1] * rot[1] + rot[3] * rot[3]),
            2.0 * (rot[2] * rot[3] + rot[0] * rot[1]),
            2.0 * (rot[1] * rot[3] + rot[0] * rot[2]),
            2.0 * (rot[2] * rot[3] - rot[0] * rot[1]),
            1.0 - 2.0 * (rot[1] * rot[1] + rot[2] * rot[2]),
        );
        let s = Mat3::new(scale[0], 0.0, 0.0, 0.0, scale[1], 0.0, 0.0, 0.0, scale[2]);
        let m = r * s;
        let m = [
            m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2],
        ];
        let sigma = [
            m[0] * m[0] + m[3] * m[3] + m[6] * m[6],
            m[0] * m[1] + m[3] * m[4] + m[6] * m[7],
            m[0] * m[2] + m[3] * m[5] + m[6] * m[8],
            m[1] * m[1] + m[4] * m[4] + m[7] * m[7],
            m[1] * m[2] + m[4] * m[5] + m[7] * m[8],
            m[2] * m[2] + m[5] * m[5] + m[8] * m[8],
        ];
        [
            pack_half_2x16(4.0 * sigma[0], 4.0 * sigma[1]),
            pack_half_2x16(4.0 * sigma[2], 4.0 * sigma[3]),
            pack_half_2x16(4.0 * sigma[4], 4.0 * sigma[5]),
        ]
    }

    pub fn render(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        camera: &Camera,
        render_data: &RenderData,
    ) {
        let scene_data = render_data.cur_scene_data.as_ref().unwrap();
        let sort_data = render_data.cur_sort_data.as_ref().unwrap();
        let render_config = &render_data.render_config;

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &render_data.depth_texture.as_ref().unwrap().view,
                depth_ops: Some(wgpu::Operations {
                    load: if render_data.use_proxy {
                        wgpu::LoadOp::Load
                    } else {
                        wgpu::LoadOp::Clear(1.0)
                    },
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        queue.write_buffer(
            &self.camera_uniforms_buffer,
            0,
            bytemuck::bytes_of(&CameraUniforms::from_camera(camera)),
        );

        queue.write_buffer(
            &self.scene_uniforms_buffer,
            0,
            bytemuck::bytes_of(&SceneUniforms::from_data(
                &self.user_data,
                scene_data,
                render_data,
            )),
        );

        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.scene_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));

        let tile_uniforms_block_size =
            usize::max(256, std::mem::size_of::<TileUniforms>() as usize);
        let view_proj = camera.view_proj();
        let mut buffer_offset: BufferAddress = 0;
        for i in 0..sort_data.render_data_vec.len() {
            let tile_instance = &sort_data.tile_instance_vec[i];
            let (render_data_key, option_render_data_value) = &sort_data.render_data_vec[i];
            let tid = tile_instance.tid;

            // viewport culling (only for non-merged tiles)
            if render_data_key.tid.len() == 1 {
                let mut pos2d = vec3(f32::MAX, f32::MAX, -f32::MAX);
                for ci in 0..4 {
                    let corner = view_proj
                        * tile_instance.corner_data.as_ref().unwrap()[ci]
                            .0
                            .extend(1.0);
                    let corner = corner.truncate() / corner.w;
                    if corner.x.abs() < pos2d.x {
                        pos2d.x = corner.x.abs();
                    }
                    if corner.y.abs() < pos2d.y {
                        pos2d.y = corner.y.abs();
                    }
                    if corner.z > pos2d.z {
                        pos2d.z = corner.z;
                    }
                }
                let clip = render_config.culling_dist;
                if pos2d.z < -clip || pos2d.x > clip || pos2d.y > clip {
                    continue;
                }
            }
            if !render_config.lod_enable[tid.0] {
                continue;
            }

            let mut tile_uniforms =
                TileUniforms::from_tile(tile_instance, option_render_data_value);
            if let Some(render_data_value) = option_render_data_value {
                tile_uniforms.single_draw = 1;
                tile_uniforms.single_lod_id = render_data_value.single_lod_id;
            }
            if render_config.debug_log && render_data_key.tid.len() >= 9 {
                log! {"{:?}", tile_instance};
                log! {"{:?}", render_data_key};
                log! {"{:?}", option_render_data_value};
            }

            queue.write_buffer(
                &self.tile_uniforms_buffer,
                (i * tile_uniforms_block_size) as BufferAddress,
                bytemuck::bytes_of(&tile_uniforms),
            );

            let splat_count: u32;
            if let Some(render_data_value) = option_render_data_value {
                splat_count = render_data_value.splat_count as u32;
                let size_byte = splat_count as u64 * 4;

                queue.write_buffer(
                    &self.gs_index_buffer,
                    buffer_offset,
                    bytemuck::cast_slice(render_data_value.gs_index.as_slice()),
                );
                render_pass.set_vertex_buffer(
                    1,
                    self.gs_index_buffer
                        .slice(buffer_offset..buffer_offset + size_byte),
                );

                queue.write_buffer(
                    &self.map_id_buffer,
                    buffer_offset,
                    bytemuck::cast_slice(render_data_value.gs_map_id.as_slice()),
                );
                render_pass.set_vertex_buffer(
                    2,
                    self.map_id_buffer
                        .slice(buffer_offset..buffer_offset + size_byte),
                );

                if render_data_value.single_lod_id == -1 {
                    queue.write_buffer(
                        &self.lod_id_buffer,
                        buffer_offset,
                        bytemuck::cast_slice(
                            render_data_value.gs_lod_id.as_ref().unwrap().as_slice(),
                        ),
                    );
                    render_pass.set_vertex_buffer(
                        3,
                        self.lod_id_buffer
                            .slice(buffer_offset..buffer_offset + size_byte),
                    );
                } else {
                    render_pass.set_vertex_buffer(3, self.lod_id_buffer.slice(..));
                }

                buffer_offset += size_byte;
            } else {
                let base_data: &BufferDataValue;
                if let TileTransitionStatus::Changing(to_lower) = tile_instance.transition_status {
                    if to_lower {
                        base_data = &self.buffer_base_data[tid.0][tid.1][tile_instance.view_id];
                    } else {
                        base_data = &self.buffer_base_data[tid.0 - 1][tid.1][tile_instance.view_id];
                    }
                } else {
                    base_data = &self.buffer_base_data[tid.0][tid.1][tile_instance.view_id];
                }
                splat_count = base_data.splat_count;

                render_pass.set_vertex_buffer(1, base_data.gs_index_buffer.slice(..));
                render_pass.set_vertex_buffer(2, self.map_id_buffer.slice(..));
                render_pass
                    .set_vertex_buffer(3, base_data.lod_id_buffer.as_ref().unwrap().slice(..));
            }

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.scene_bind_group, &[]);
            render_pass.set_bind_group(
                1,
                &self.tile_bind_group,
                &[(i * tile_uniforms_block_size) as u32],
            );
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));

            render_pass.draw(0..6, 0..splat_count);
        }
    }
}

struct BufferDataValue {
    splat_count: u32,
    gs_index_buffer: wgpu::Buffer,
    map_id_buffer: Option<wgpu::Buffer>,
    lod_id_buffer: Option<wgpu::Buffer>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SceneUniforms {
    splat_scale: f32,
    tile_width: f32,
    use_clip: u32,
    clip_height: f32,
    surface_type: u32,
    sphere_radius: f32,
    point_cloud_radius: f32,
    transition_width_ratio: f32,
    num_lod: u32,
    draw_mode: u32,

    map_half_wh: [u32; 2],
    center_coord: [i32; 2],
    _pad0: [u32; 2],
    transition_dist_vec: [f32; 16],
    height_map_scale: [f32; 4],
    scene_scale: [f32; 4],
    motion_params: [f32; 4],
    motion_params2: [f32; 4],
    motion_spline_knots: [[f32; 4]; MOTION_PACKED_KNOT_COUNT],
}
impl SceneUniforms {
    fn expand_to_array<const N: usize, T: Copy>(slice: &[T], pad_value: T) -> [T; N] {
        let mut arr = [pad_value; N];
        let len = slice.len().min(N); // avoid overflow
        arr[..len].copy_from_slice(&slice[..len]);
        arr
    }

    fn from_data(user_data: &UserData, scene_data: &SceneData, render_data: &RenderData) -> Self {
        let render_config = &render_data.render_config;
        Self {
            splat_scale: render_config.splat_scale,
            tile_width: user_data.tile_width,
            use_clip: render_config.use_clip as u32,
            clip_height: render_config.clip_height,
            surface_type: user_data.surface_type as u32,
            sphere_radius: user_data.sphere_radius,
            point_cloud_radius: if render_config.draw_point_cloud {
                render_config.point_cloud_radius
            } else {
                0.0
            },
            transition_width_ratio: user_data.lod_transition_width_ratio,
            num_lod: user_data.n_tiles.1 as u32,
            draw_mode: render_config.draw_mode as u32,

            map_half_wh: [
                user_data.tile_map_half_wh.x as u32,
                user_data.tile_map_half_wh.y as u32,
            ],
            center_coord: [scene_data.center_coord.x, scene_data.center_coord.y],
            _pad0: [0; 2],
            transition_dist_vec: Self::expand_to_array::<16, f32>(
                &user_data.lod_transition_dist,
                0.0,
            ),
            scene_scale: [
                render_config.scene_scale.x,
                render_config.scene_scale.y,
                render_config.scene_scale.z,
                0.0,
            ],
            height_map_scale: [
                user_data.height_map_scale.x,
                user_data.height_map_scale.y,
                user_data.height_map_scale.z * render_config.height_map_scale_v,
                0.0,
            ],
            motion_params: [
                if render_config.motion_edit.enabled {
                    1.0
                } else {
                    0.0
                },
                render_config.motion_edit.amplitude,
                render_config.motion_edit.edge_band,
                render_config.motion_edit.detail_amplitude,
            ],
            motion_params2: [
                render_data.animation_time,
                render_config.motion_edit.wave_phase_span,
                0.0,
                0.0,
            ],
            motion_spline_knots: pack_motion_spline_knots(&render_config.motion_edit),
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TileUniforms {
    single_draw: u32,
    map_index: u32,
    single_lod_id: i32,
    valid_lod_id: i32,
    changing: u32,
    changing_to_lower: i32,
    _pad0: [u32; 2],

    tile_id: [u32; 4],
    offset: [f32; 4],
    map_coord: [u32; 4],
}
impl TileUniforms {
    fn from_tile(tile: &TileInstance, render_data_value: &Option<RenderDataValue>) -> Self {
        let mut uniforms = Self {
            single_draw: 0,
            map_index: tile.map_index as u32,
            single_lod_id: -1,
            valid_lod_id: -1,
            changing: 0,
            changing_to_lower: -1,
            _pad0: [0; 2],

            tile_id: [tile.tid.0 as u32, tile.tid.1 as u32, tile.view_id as u32, 0],
            offset: [
                tile.tile_offset.x,
                tile.tile_offset.y,
                tile.tile_offset.z,
                0.0,
            ],
            map_coord: [tile.map_coord.x as u32, tile.map_coord.y as u32, 0, 0],
        };

        if let Some(data_value) = render_data_value {
            uniforms.single_draw = 1;
            uniforms.single_lod_id = data_value.single_lod_id;
            uniforms.changing = (uniforms.single_lod_id == -1) as u32;
        } else {
            if let TileTransitionStatus::Changing(to_lower) = tile.transition_status {
                uniforms.changing = 1;
                uniforms.changing_to_lower = to_lower as i32;
            } else {
                uniforms.valid_lod_id = tile.tid.0 as i32;
            }
        }

        uniforms
    }
}

fn valid_motion_duration(duration: Option<f32>) -> Option<f32> {
    duration.filter(|value| value.is_finite() && *value > 0.0)
}

fn source_motion_duration_from_frame_count(n_time_frames: usize) -> f32 {
    (n_time_frames as f32 / SOURCE_MOTION_FPS).max(1e-6)
}

fn catmull_rom_motion_duration(
    metadata_duration: Option<f32>,
    network_duration: Option<f32>,
) -> f32 {
    valid_motion_duration(metadata_duration)
        .or_else(|| valid_motion_duration(network_duration))
        .unwrap_or(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::Scene;

    #[test]
    fn covariance_packing_matches_scene_texture_path() {
        let mut scene = Scene::new();
        scene.splat_count = 1;
        scene.buffer = vec![0_u8; 32];

        let scale = [1.2_f32, 0.7_f32, 2.0_f32];
        let q_raw = [0.2_f32, 0.3_f32, 0.4_f32, 0.5_f32];
        let q_len =
            (q_raw[0] * q_raw[0] + q_raw[1] * q_raw[1] + q_raw[2] * q_raw[2] + q_raw[3] * q_raw[3])
                .sqrt();
        let q = [
            q_raw[0] / q_len,
            q_raw[1] / q_len,
            q_raw[2] / q_len,
            q_raw[3] / q_len,
        ];

        {
            let fbuf: &mut [f32] = transmute_slice_mut(scene.buffer.as_mut_slice());
            fbuf[0] = 0.0;
            fbuf[1] = 0.0;
            fbuf[2] = 0.0;
            fbuf[3] = scale[0];
            fbuf[4] = scale[1];
            fbuf[5] = scale[2];
        }
        {
            let ubuf: &mut [u8] = transmute_slice_mut(scene.buffer.as_mut_slice());
            ubuf[24] = 128;
            ubuf[25] = 128;
            ubuf[26] = 128;
            ubuf[27] = 255;
            ubuf[28] = (((q[0] + 1.0) * 0.5 * 255.0) as u8).clamp(0, 255);
            ubuf[29] = (((q[1] + 1.0) * 0.5 * 255.0) as u8).clamp(0, 255);
            ubuf[30] = (((q[2] + 1.0) * 0.5 * 255.0) as u8).clamp(0, 255);
            ubuf[31] = (((q[3] + 1.0) * 0.5 * 255.0) as u8).clamp(0, 255);
        }

        scene.generate_texture();
        let tex_cov = [scene.tex_data[4], scene.tex_data[5], scene.tex_data[6]];
        let ubuf: &[u8] = transmute_slice(scene.buffer.as_slice());
        let q_dec = [
            (ubuf[28] as f32 / 255.0) * 2.0 - 1.0,
            (ubuf[29] as f32 / 255.0) * 2.0 - 1.0,
            (ubuf[30] as f32 / 255.0) * 2.0 - 1.0,
            (ubuf[31] as f32 / 255.0) * 2.0 - 1.0,
        ];
        let pack_cov = GSWTRenderer::pack_covariance(scale, q_dec);
        assert_eq!(tex_cov, pack_cov);
    }

    #[test]
    fn source_motion_duration_matches_volume_frame_count() {
        assert!((source_motion_duration_from_frame_count(75) - 2.5).abs() < 1e-6);
    }

    #[test]
    fn catmull_rom_duration_prefers_metadata_then_network_then_default() {
        assert!((catmull_rom_motion_duration(Some(3.0), Some(2.5)) - 3.0).abs() < 1e-6);
        assert!((catmull_rom_motion_duration(None, Some(2.5)) - 2.5).abs() < 1e-6);
        assert!((catmull_rom_motion_duration(Some(-1.0), Some(2.5)) - 2.5).abs() < 1e-6);
        assert!((catmull_rom_motion_duration(None, None) - 1.0).abs() < 1e-6);
    }
}
