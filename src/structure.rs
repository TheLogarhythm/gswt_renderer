use std::{
    collections::VecDeque,
    sync::{
        Arc,
        mpsc::{Receiver, Sender},
    },
};
use winit::keyboard::KeyCode;

use crate::basis_bank_edit::{BasisEditOverride, BasisKnotEditState, resize_basis_edit_overrides};
use crate::basis_bank_motion::BasisBankMotionSet;
use crate::basis_graph_playback::{BasisGraphPlaybackConfig, BasisGraphPlaybackState};
use crate::catmull_rom_motion::MotionMode;
use crate::control::{CameraControl, FlyPathControl};
use crate::deformation::DeformationNetwork;
use crate::motion::MotionEditConfig;
use crate::scene::Scene;
use crate::skybox::SkyboxTexture;
use crate::texture::Texture;
use crate::utils::*;

/// All user data from Config stage
#[derive(Clone)]
pub struct UserData {
    /// ID for this config
    pub config_id: u32,
    /// Half of width/height for tile map (in number of tiles)
    ///
    /// Actual width/height computed based on surface type (2n or 2n+1)
    pub tile_map_half_wh: Vector2<usize>,
    /// Number of center options for each tile (less equal than that provided during upload)
    pub center_option: usize,
    /// The distance (squared) that the camera needs to travel until a tile map update is triggered
    pub update_distance2: f32,
    /// Width of a tile
    pub tile_width: f32,

    pub tile_sort_type: TileSortType,

    // Surface
    pub surface_type: SurfaceType,
    pub height_map_wh: Vector2<usize>,
    pub height_map_type: HeightMapType,
    pub height_map_scale: Vec3,
    pub height_tex: Option<(Vec<f32>, Vector2<usize>)>,
    pub sphere_radius: f32,

    // LOD
    pub lod_max_dist: f32,
    pub lod_blending: bool,
    pub lod_transition_width_ratio: f32,
    pub lod_bbox_check: bool,
    pub lod_dist_tolerance: f32,

    // Selective merging
    pub merge_type: SelectiveMergeType,
    pub merge_tile_dist: (i32, i32),
    pub merge_dot_threshold: f32,
    pub merge_topk: usize,
    pub use_cache: bool,
    pub cache_size: usize,

    pub reset_rng: bool,
    pub always_sort: bool,

    // From wang thread
    /// Actual width/height for the tile map (in number of tiles)
    pub tile_map_wh: Vector2<usize>,
    pub height_map: Vec<f32>,
    /// A list of transition distances of each lod
    pub lod_transition_dist: Vec<f32>,
    /// n_lod, n_tile, n_view
    pub n_tiles: (usize, usize, usize),
}
impl UserData {
    pub fn new() -> Self {
        Self {
            config_id: 0,
            tile_map_half_wh: Vector2::new(48, 48),
            center_option: 1,
            update_distance2: 1.0,
            tile_width: 4.0,
            tile_sort_type: TileSortType::Graph,
            surface_type: SurfaceType::HeightMap,
            height_map_wh: Vector2::new(0, 0),
            height_map_type: HeightMapType::Random,
            height_map_scale: vec3(1.0, 1.0, 0.0),
            height_tex: None,
            sphere_radius: 0.0,
            lod_max_dist: 0.0,
            lod_blending: true,
            lod_transition_width_ratio: 0.0,
            lod_bbox_check: true,
            lod_dist_tolerance: 0.0,
            merge_type: SelectiveMergeType::Edge,
            merge_tile_dist: (-1, -1),
            merge_dot_threshold: 3.0,
            merge_topk: 100,
            use_cache: true,
            cache_size: 1024,
            reset_rng: true,
            always_sort: false,

            tile_map_wh: Vector2::new(0, 0),
            height_map: Vec::new(),
            lod_transition_dist: Vec::new(),
            n_tiles: (0, 0, 0),
        }
    }
}

/// String version of [UserData] to help with config parsing
pub struct UserDataString {
    pub tile_map_half_wh_s: Vector2<String>,
    pub center_option_s: String,
    pub update_dist_s: String,
    pub tile_width_s: String,
    pub height_map_wh_s: Vector2<String>,
    pub height_map_scale_s: Vector2<String>,
    pub sphere_radius_s: String,
    pub merge_tile_dist_s: Vector2<String>,
    pub merge_dot_threshold_s: String,
    pub merge_topk_s: String,
    pub lod_max_dist_s: String,
    pub lod_transition_width_ratio_s: String,
    pub lod_dist_tolerance_s: String,
    pub cache_size_s: String,
}
impl UserDataString {
    pub fn new() -> Self {
        Self {
            tile_map_half_wh_s: vec2(48.to_string(), 48.to_string()),
            center_option_s: 1.to_string(),
            update_dist_s: 1.to_string(),
            tile_width_s: 4.to_string(),
            height_map_wh_s: vec2(10.to_string(), 10.to_string()),
            height_map_scale_s: vec2(1.to_string(), 1.to_string()),
            sphere_radius_s: 20.to_string(),
            merge_tile_dist_s: vec2(3.to_string(), 10.to_string()),
            merge_dot_threshold_s: 0.2.to_string(),
            merge_topk_s: 100.to_string(),
            lod_max_dist_s: 96.to_string(),
            lod_transition_width_ratio_s: String::from("0.05"),
            lod_dist_tolerance_s: 0.to_string(),
            cache_size_s: 1024.to_string(),
        }
    }

    pub fn to_raw(&self, user_data: &mut UserData, err_msg: &mut Option<String>) {
        parse_num(
            &self.tile_map_half_wh_s.x,
            &mut user_data.tile_map_half_wh.x,
            err_msg,
        );
        parse_num(
            &self.tile_map_half_wh_s.y,
            &mut user_data.tile_map_half_wh.y,
            err_msg,
        );
        parse_num(&self.center_option_s, &mut user_data.center_option, err_msg);
        parse_num(
            &self.update_dist_s,
            &mut user_data.update_distance2,
            err_msg,
        );
        user_data.update_distance2 = user_data.update_distance2.powi(2);
        parse_num(&self.tile_width_s, &mut user_data.tile_width, err_msg);
        parse_num(
            &self.height_map_wh_s.x,
            &mut user_data.height_map_wh.x,
            err_msg,
        );
        parse_num(
            &self.height_map_wh_s.y,
            &mut user_data.height_map_wh.y,
            err_msg,
        );
        parse_num(
            &self.height_map_scale_s.x,
            &mut user_data.height_map_scale.x,
            err_msg,
        );
        user_data.height_map_scale.y = user_data.height_map_scale.x;
        parse_num(
            &self.height_map_scale_s.y,
            &mut user_data.height_map_scale.z,
            err_msg,
        );
        parse_num(&self.sphere_radius_s, &mut user_data.sphere_radius, err_msg);
        parse_num(
            &self.merge_tile_dist_s.x,
            &mut user_data.merge_tile_dist.0,
            err_msg,
        );
        parse_num(
            &self.merge_tile_dist_s.y,
            &mut user_data.merge_tile_dist.1,
            err_msg,
        );
        parse_num(
            &self.merge_dot_threshold_s,
            &mut user_data.merge_dot_threshold,
            err_msg,
        );
        parse_num(&self.merge_topk_s, &mut user_data.merge_topk, err_msg);

        parse_num(&self.lod_max_dist_s, &mut user_data.lod_max_dist, err_msg);
        user_data.lod_max_dist *= user_data.tile_width;
        parse_num(
            &self.lod_transition_width_ratio_s,
            &mut user_data.lod_transition_width_ratio,
            err_msg,
        );
        parse_num(
            &self.lod_dist_tolerance_s,
            &mut user_data.lod_dist_tolerance,
            err_msg,
        );
        parse_num(&self.cache_size_s, &mut user_data.cache_size, err_msg);
    }
}

pub struct RenderData {
    pub cur_scene_data: Option<SceneData>,
    pub next_scene_data: Option<SceneData>,
    pub cur_sort_data: Option<SortData>,
    pub next_sort_data: Option<SortData>,
    pub cur_scene_data_id: Option<u32>,
    pub next_scene_data_id: Option<u32>,
    pub cur_sort_data_id: Option<u32>,
    pub next_sort_data_id: Option<u32>,

    pub frame_prev: f64,
    pub time_ma_window: usize,
    pub frame_time_ma: IncrementalMA,
    pub sort_time_ma: IncrementalMA,
    pub build_time_ma: IncrementalMA,
    pub sort_trigger_ma: IncrementalMA,
    pub build_trigger_ma: IncrementalMA,
    pub profiling_enabled: bool,
    pub profiling_interval_frames: u32,
    pub profiling_frame_counter: u64,

    pub show_main_menu: bool,
    pub show_perf_menu: bool,
    pub show_motion_authoring_menu: bool,
    pub show_fly_path_menu: bool,
    pub hide_menu_when_start: bool,

    pub camera_control_type: CameraControl,
    pub set_cam_clicked: bool,
    pub cam_pos_s: Vector3<String>,
    pub cam_dir_s: Vector3<String>,
    pub cam_up_s: Vector3<String>,
    pub set_cam_pos: Vec3,
    pub set_cam_dir: Vec3,
    pub set_cam_up: Vec3,
    pub lockon_center: bool,

    pub lock_tile: bool,
    pub lock_sort: bool,
    pub freeze_frame: bool,
    pub step_frame: bool,
    pub update_worker: bool,

    pub render_config: RenderConfig,
    pub render_gs: bool,
    pub use_skybox: bool,
    pub use_proxy: bool,

    pub max_lod_count: usize,

    pub fly_path_error_msg: Option<String>,
    pub fly_path_benchmark: bool,

    pub skybox_rawtex: Option<(SkyboxTexture, Vector2<usize>)>,
    pub proxy_rawtex: Option<(Vec<Vec<f32>>, Vector2<usize>)>,

    pub depth_texture: Option<Texture>,

    pub has_deformation: bool,
    pub animation_playing: bool,
    pub animation_speed: f32,
    pub animation_phase: f32,
    pub animation_time: f32,
    pub animation_duration: f32,
    pub apply_network_delta_rot: bool,
    pub manual_spline_knot_preview: bool,
    pub selected_spline_knot: u32,
    pub motion_debug_dirty: bool,
    pub active_motion_mode: MotionMode,
    pub catmull_rom_knot_count: Option<u32>,
    pub catmull_rom_uses_volume_key_times: bool,
    pub basis_bank_basis_count: Option<u32>,
    pub basis_bank_top_k: Option<u32>,
    pub basis_bank_preview: Option<Arc<BasisBankMotionSet>>,
    pub basis_preview_enabled: bool,
    pub basis_best_target_preview_enabled: bool,
    pub basis_preview_selected_id: u32,
    pub basis_graph_selected_segment: u32,
    pub basis_preview_projection: BasisPreviewProjection,
    pub basis_preview_heatmap_enabled: bool,
    pub basis_preview_heatmap_normalization: f32,
    pub basis_edit_overrides: Vec<BasisEditOverride>,
    pub basis_edit_dirty: bool,
    pub basis_knot_edits: Option<BasisKnotEditState>,
    pub basis_knot_edit_dirty: bool,
    pub basis_knot_edit_selected_knot: u32,
    pub basis_knot_edit_dragging_knot: Option<u32>,
    pub basis_view3d_yaw: f32,
    pub basis_view3d_pitch: f32,
    pub basis_view3d_zoom: f32,
    pub basis_graph_playback_config: BasisGraphPlaybackConfig,
    pub basis_graph_quality_custom_mode: bool,
    pub basis_graph_playback_reset_requested: bool,
    pub basis_graph_playback_selected_state: Option<BasisGraphPlaybackState>,
    pub motion_compatibility_volume_keys: Option<u32>,
    pub motion_compatibility_scope: MotionCompatibilityScope,
    pub motion_compatibility_requested: bool,
    pub motion_compatibility_running: bool,
    pub motion_compatibility_result: Option<MotionCompatibilityResult>,
    pub motion_compatibility_error: Option<String>,
    pub motion_texture_compare_requested: bool,
    pub motion_texture_compare_running: bool,
    pub motion_texture_compare_result: Option<MotionTextureCompareResult>,
    pub motion_texture_compare_error: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BasisPreviewProjection {
    XY,
    XZ,
    YZ,
}

impl BasisPreviewProjection {
    pub const ALL: [Self; 3] = [Self::XY, Self::XZ, Self::YZ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::XY => "XY",
            Self::XZ => "XZ",
            Self::YZ => "YZ",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MotionCompatibilityScope {
    SelectedKnot,
    AllKnots,
}

impl MotionCompatibilityScope {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SelectedKnot => "selected knot",
            Self::AllKnots => "all knots",
        }
    }
}

#[derive(Clone, Debug)]
pub struct MotionCompatibilityResult {
    pub scope: MotionCompatibilityScope,
    pub actual_compared_count: u64,
    pub mean_error: f32,
    pub rms_error: f32,
    pub max_error: f32,
    pub sampled_p95_error: f32,
    pub mean_spline_delta_magnitude: f32,
    pub max_spline_delta_magnitude: f32,
    pub mean_volume_delta_magnitude: f32,
    pub max_volume_delta_magnitude: f32,
    pub nonzero_error_count: u64,
    pub nonzero_spline_delta_count: u64,
    pub nonzero_volume_delta_count: u64,
    pub splat_count: u32,
    pub knot_count: u32,
    pub compared_knots: u32,
    pub worst_knot: u32,
    pub worst_splat: u32,
    pub sampled_error_count: u32,
}

#[derive(Clone, Debug)]
pub struct MotionTextureCompareResult {
    pub time01: f32,
    pub volume_time01: f32,
    pub actual_compared_count: u64,
    pub mean_error: f32,
    pub rms_error: f32,
    pub max_error: f32,
    pub sampled_p95_error: f32,
    pub mean_catmull_rom_mean_magnitude: f32,
    pub max_catmull_rom_mean_magnitude: f32,
    pub mean_volume_mean_magnitude: f32,
    pub max_volume_mean_magnitude: f32,
    pub nonzero_error_count: u64,
    pub splat_count: u32,
    pub worst_splat: u32,
    pub sampled_error_count: u32,
}

impl RenderData {
    pub fn new(max_lod_count: usize) -> Self {
        let default_ma_window: usize = 200;

        Self {
            cur_scene_data: None,
            next_scene_data: None,
            cur_sort_data: None,
            next_sort_data: None,
            cur_scene_data_id: None,
            next_scene_data_id: None,
            cur_sort_data_id: None,
            next_sort_data_id: None,

            frame_prev: get_time_milliseconds(),
            time_ma_window: default_ma_window,
            frame_time_ma: IncrementalMA::new(default_ma_window),
            sort_time_ma: IncrementalMA::new(default_ma_window),
            build_time_ma: IncrementalMA::new(default_ma_window),
            sort_trigger_ma: IncrementalMA::new(default_ma_window),
            build_trigger_ma: IncrementalMA::new(default_ma_window),
            profiling_enabled: true,
            profiling_interval_frames: 15,
            profiling_frame_counter: 0,

            show_main_menu: true,
            show_perf_menu: false,
            show_motion_authoring_menu: false,
            show_fly_path_menu: false,
            hide_menu_when_start: false,

            camera_control_type: CameraControl::KeyboardFly,
            set_cam_clicked: false,
            cam_pos_s: vec3(0.to_string(), 0.to_string(), 0.to_string()),
            cam_dir_s: vec3(0.to_string(), 1.to_string(), 0.to_string()),
            cam_up_s: vec3(0.to_string(), 0.to_string(), 1.to_string()),
            set_cam_pos: Vec3::zero(),
            set_cam_dir: Vec3::zero(),
            set_cam_up: Vec3::zero(),
            lockon_center: false,

            lock_tile: false,
            lock_sort: false,
            freeze_frame: false,
            step_frame: false,
            update_worker: false,

            render_config: RenderConfig::new(max_lod_count),
            render_gs: true,
            use_skybox: false,
            use_proxy: false,

            max_lod_count,

            fly_path_error_msg: None,
            fly_path_benchmark: false,

            skybox_rawtex: None,
            proxy_rawtex: None,

            depth_texture: None,

            has_deformation: false,
            animation_playing: true,
            animation_speed: 1.0,
            animation_phase: 0.0,
            animation_time: 0.0,
            animation_duration: 1.0,
            apply_network_delta_rot: true,
            manual_spline_knot_preview: false,
            selected_spline_knot: 0,
            motion_debug_dirty: false,
            active_motion_mode: MotionMode::Static,
            catmull_rom_knot_count: None,
            catmull_rom_uses_volume_key_times: false,
            basis_bank_basis_count: None,
            basis_bank_top_k: None,
            basis_bank_preview: None,
            basis_preview_enabled: false,
            basis_best_target_preview_enabled: false,
            basis_preview_selected_id: 0,
            basis_graph_selected_segment: 0,
            basis_preview_projection: BasisPreviewProjection::XY,
            basis_preview_heatmap_enabled: false,
            basis_preview_heatmap_normalization: 1.0,
            basis_edit_overrides: Vec::new(),
            basis_edit_dirty: false,
            basis_knot_edits: None,
            basis_knot_edit_dirty: false,
            basis_knot_edit_selected_knot: 0,
            basis_knot_edit_dragging_knot: None,
            basis_view3d_yaw: 0.65,
            basis_view3d_pitch: 0.35,
            basis_view3d_zoom: 1.0,
            basis_graph_playback_config: BasisGraphPlaybackConfig::default(),
            basis_graph_quality_custom_mode: false,
            basis_graph_playback_reset_requested: false,
            basis_graph_playback_selected_state: None,
            motion_compatibility_volume_keys: None,
            motion_compatibility_scope: MotionCompatibilityScope::SelectedKnot,
            motion_compatibility_requested: false,
            motion_compatibility_running: false,
            motion_compatibility_result: None,
            motion_compatibility_error: None,
            motion_texture_compare_requested: false,
            motion_texture_compare_running: false,
            motion_texture_compare_result: None,
            motion_texture_compare_error: None,
        }
    }

    pub fn parse_camera_config(&mut self) {
        let mut err: Option<String> = None;
        parse_num(&self.cam_pos_s.x, &mut self.set_cam_pos.x, &mut err);
        parse_num(&self.cam_pos_s.y, &mut self.set_cam_pos.y, &mut err);
        parse_num(&self.cam_pos_s.z, &mut self.set_cam_pos.z, &mut err);
        parse_num(&self.cam_dir_s.x, &mut self.set_cam_dir.x, &mut err);
        parse_num(&self.cam_dir_s.y, &mut self.set_cam_dir.y, &mut err);
        parse_num(&self.cam_dir_s.z, &mut self.set_cam_dir.z, &mut err);
        parse_num(&self.cam_up_s.x, &mut self.set_cam_up.x, &mut err);
        parse_num(&self.cam_up_s.y, &mut self.set_cam_up.y, &mut err);
        parse_num(&self.cam_up_s.z, &mut self.set_cam_up.z, &mut err);

        if err.is_none() {
            self.set_cam_clicked = true;
        }
    }

    pub fn toggle_animation_playing(&mut self) -> bool {
        toggle_animation_playing_state(self.has_deformation, &mut self.animation_playing)
    }

    pub fn reset_frame_timing(&mut self, now_ms: f64) {
        self.frame_prev = now_ms;
        self.frame_time_ma.clear();
    }

    pub fn set_motion_debug_backend(
        &mut self,
        active_motion_mode: MotionMode,
        catmull_rom_knot_count: Option<u32>,
        catmull_rom_uses_volume_key_times: bool,
        basis_bank_basis_count: Option<u32>,
        basis_bank_top_k: Option<u32>,
        basis_bank_preview: Option<Arc<BasisBankMotionSet>>,
    ) {
        self.active_motion_mode = active_motion_mode;
        self.catmull_rom_knot_count = catmull_rom_knot_count;
        self.catmull_rom_uses_volume_key_times = catmull_rom_uses_volume_key_times;
        self.basis_bank_basis_count = basis_bank_basis_count;
        self.basis_bank_top_k = basis_bank_top_k;
        let had_basis_bank_preview = self.basis_bank_preview.is_some();
        self.basis_bank_preview = basis_bank_preview;
        if !had_basis_bank_preview && self.basis_bank_preview.is_some() {
            self.basis_preview_enabled = true;
        }
        self.basis_preview_selected_id =
            clamp_basis_preview_id(self.basis_preview_selected_id, self.basis_bank_basis_count);
        self.basis_graph_selected_segment = clamp_basis_graph_segment(
            self.basis_graph_selected_segment,
            self.basis_bank_preview.as_ref(),
        );
        if self
            .basis_bank_preview
            .as_ref()
            .and_then(|motion| motion.motion_graph.as_ref())
            .is_none()
        {
            self.basis_graph_playback_config.enabled = false;
            self.basis_graph_quality_custom_mode = false;
            self.basis_graph_playback_reset_requested = false;
            self.basis_graph_playback_selected_state = None;
        }
        if active_motion_mode == MotionMode::BasisBank {
            if let Some(count) = self.basis_bank_basis_count {
                resize_basis_edit_overrides(&mut self.basis_edit_overrides, count as usize);
            } else {
                self.basis_edit_overrides.clear();
                self.basis_edit_dirty = false;
            }
            if let Some(motion) = self.basis_bank_preview.as_ref() {
                let basis_count = motion.global_basis_count;
                let knot_count = motion.meta.exported_knot_count;
                let needs_init = self
                    .basis_knot_edits
                    .as_ref()
                    .map(|edits| {
                        !edits.matches_shape(basis_count, knot_count)
                            || edits.original_knots() != motion.global_basis_knots.as_slice()
                    })
                    .unwrap_or(true);
                if needs_init {
                    self.basis_knot_edits = Some(BasisKnotEditState::new(
                        motion.global_basis_knots.clone(),
                        basis_count,
                        knot_count,
                        knot_count,
                    ));
                    self.basis_knot_edit_dirty = false;
                }
                self.basis_knot_edit_selected_knot =
                    clamp_basis_knot_edit_selected(self.basis_knot_edit_selected_knot, knot_count);
                if self
                    .basis_knot_edit_dragging_knot
                    .is_some_and(|knot| knot as usize >= knot_count)
                {
                    self.basis_knot_edit_dragging_knot = None;
                }
            } else {
                self.basis_knot_edits = None;
                self.basis_knot_edit_dirty = false;
                self.basis_knot_edit_selected_knot = 0;
                self.basis_knot_edit_dragging_knot = None;
            }
        } else {
            self.basis_edit_overrides.clear();
            self.basis_edit_dirty = false;
            self.basis_knot_edits = None;
            self.basis_knot_edit_dirty = false;
            self.basis_knot_edit_selected_knot = 0;
            self.basis_knot_edit_dragging_knot = None;
        }
        self.selected_spline_knot =
            clamp_spline_knot(self.selected_spline_knot, catmull_rom_knot_count);
        if catmull_rom_knot_count.is_none() {
            self.manual_spline_knot_preview = false;
            self.catmull_rom_uses_volume_key_times = false;
            self.basis_bank_basis_count = None;
            self.basis_bank_top_k = None;
        }
        if self.basis_bank_preview.is_none() {
            self.basis_preview_enabled = false;
            self.basis_best_target_preview_enabled = false;
            self.basis_preview_heatmap_enabled = false;
            self.basis_preview_selected_id = 0;
            self.basis_graph_selected_segment = 0;
            self.basis_graph_playback_config.enabled = false;
            self.basis_graph_quality_custom_mode = false;
            self.basis_graph_playback_reset_requested = false;
            self.basis_graph_playback_selected_state = None;
            self.basis_knot_edits = None;
            self.basis_knot_edit_dirty = false;
            self.basis_knot_edit_selected_knot = 0;
            self.basis_knot_edit_dragging_knot = None;
            if self.render_config.draw_mode == DrawMode::BasisWeight {
                self.render_config.draw_mode = DrawMode::Normal;
            }
        }
    }

    pub fn spline_knot_preview_time(&self) -> Option<f32> {
        spline_knot_preview_time(
            self.selected_spline_knot,
            self.catmull_rom_knot_count,
            self.catmull_rom_uses_volume_key_times,
        )
    }

    pub fn mark_motion_debug_dirty(&mut self) {
        self.motion_debug_dirty = true;
    }

    pub fn clear_motion_debug_dirty(&mut self) {
        self.motion_debug_dirty = false;
    }

    pub fn mark_basis_edit_dirty(&mut self) {
        self.basis_edit_dirty = true;
    }

    pub fn clear_basis_edit_dirty(&mut self) {
        self.basis_edit_dirty = false;
    }

    pub fn mark_basis_knot_edit_dirty(&mut self) {
        self.basis_knot_edit_dirty = true;
    }

    pub fn clear_basis_knot_edit_dirty(&mut self) {
        self.basis_knot_edit_dirty = false;
    }

    pub fn request_basis_graph_playback_reset(&mut self) {
        self.basis_graph_playback_reset_requested = true;
        self.mark_motion_debug_dirty();
    }

    pub fn clear_basis_graph_playback_reset(&mut self) {
        self.basis_graph_playback_reset_requested = false;
    }
}

fn toggle_animation_playing_state(has_deformation: bool, animation_playing: &mut bool) -> bool {
    if has_deformation {
        *animation_playing = !*animation_playing;
    }
    *animation_playing
}

fn rounded_endpoint01(x: f32, width: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    let width = width.clamp(1e-6, 0.5);

    if x < width {
        let inv_width = 1.0 / width;
        let x2 = x * x;
        2.0 * x2 * inv_width - x2 * x * inv_width * inv_width
    } else if x > 1.0 - width {
        1.0 - rounded_endpoint01(1.0 - x, width)
    } else {
        x
    }
}

pub fn smooth_ping_pong01(phase: f32) -> f32 {
    const ENDPOINT_SMOOTH_WIDTH: f32 = 0.08;

    let phase = phase.rem_euclid(1.0);
    let triangle = 1.0 - (2.0 * phase - 1.0).abs();
    rounded_endpoint01(triangle, ENDPOINT_SMOOTH_WIDTH)
}

pub fn animation_delta_seconds(raw_delta_ms: f64) -> f32 {
    const MAX_ANIMATION_DELTA_MS: f64 = 66.0;

    raw_delta_ms.clamp(0.0, MAX_ANIMATION_DELTA_MS) as f32 / 1000.0
}

fn clamp_spline_knot(selected_knot: u32, knot_count: Option<u32>) -> u32 {
    match knot_count {
        Some(count) if count > 0 => selected_knot.min(count - 1),
        _ => 0,
    }
}

fn clamp_basis_preview_id(selected_basis: u32, basis_count: Option<u32>) -> u32 {
    match basis_count {
        Some(count) if count > 0 => selected_basis.min(count - 1),
        _ => 0,
    }
}

fn clamp_basis_graph_segment(
    selected_segment: u32,
    basis_bank_preview: Option<&Arc<BasisBankMotionSet>>,
) -> u32 {
    match basis_bank_preview {
        Some(motion) if motion.meta.exported_knot_count > 0 => {
            selected_segment.min(motion.meta.exported_knot_count as u32 - 1)
        }
        _ => 0,
    }
}

fn clamp_basis_knot_edit_selected(selected_knot: u32, editable_knot_count: usize) -> u32 {
    if editable_knot_count > 0 {
        selected_knot.min(editable_knot_count as u32 - 1)
    } else {
        0
    }
}

fn spline_knot_preview_time(
    selected_knot: u32,
    knot_count: Option<u32>,
    use_volume_key_times: bool,
) -> Option<f32> {
    let count = knot_count?;
    if count == 0 {
        return None;
    }
    let denominator = if use_volume_key_times && count > 1 {
        count - 1
    } else {
        count
    };
    Some(clamp_spline_knot(selected_knot, knot_count) as f32 / denominator as f32)
}

#[derive(Clone)]
pub struct RenderConfig {
    pub draw_mode: DrawMode,
    pub motion_edit: MotionEditConfig,
    pub height_map_scale_v: f32,
    pub scene_scale: Vec3,
    pub use_clip: bool,
    pub clip_height: f32,
    pub draw_point_cloud: bool,
    pub point_cloud_radius: f32,
    pub culling_dist: f32,
    pub proxy_full: bool,
    pub proxy_map: bool,
    pub proxy_height: f32,
    pub proxy_width_scale: f32,
    pub proxy_brightness: f32,
    pub proxy_black_background: bool,
    pub lod_enable: Vec<bool>,
    pub debug_log: bool,
    pub splat_scale: f32,
}
impl RenderConfig {
    pub fn new(max_lod_count: usize) -> Self {
        Self {
            draw_mode: DrawMode::Normal,
            motion_edit: MotionEditConfig::default(),
            height_map_scale_v: 1.0,
            scene_scale: vec3(1.0, 1.0, 1.0),
            use_clip: false,
            clip_height: 0.0,
            draw_point_cloud: false,
            point_cloud_radius: 0.01,
            culling_dist: 1.0,
            proxy_full: false,
            proxy_map: true,
            proxy_height: -0.5,
            proxy_width_scale: 4.0,
            proxy_brightness: 1.0,
            proxy_black_background: false,
            lod_enable: vec![true; max_lod_count],
            debug_log: false,
            splat_scale: 1.0,
        }
    }
}

#[repr(u32)]
#[derive(PartialEq, Clone, Copy)]
pub enum DrawMode {
    Normal = 0,
    TileID = 1,
    TileLOD = 2,
    LOD = 3,
    View = 4,
    BasisWeight = 5,
}

pub struct MainChannels {
    pub tx_vp: Sender<Mat4>,
    pub tx_build_info: Sender<(bool, Vec3)>,
    pub tx_user_data: Sender<UserData>,

    pub rx_user_data: Receiver<UserData>,
    pub rx_sort_data: Receiver<SortData>,
    pub rx_scene_data: Receiver<SceneData>,
    pub rx_sort_time: Receiver<f64>,
    pub rx_build_time: Receiver<f64>,

    pub rx_fly_path_control: Option<Receiver<FlyPathControl>>,
    pub rx_height_tex: Option<Receiver<(Vec<f32>, Vector2<usize>)>>,
    pub rx_skybox_tex: Option<Receiver<(SkyboxTexture, Vector2<usize>)>>,
    pub rx_proxy_tex: Option<Receiver<(Vec<Vec<f32>>, Vector2<usize>)>>,
}

pub struct WorkerChannels {
    pub rx_vp: Receiver<Mat4>,
    pub rx_build_info: Receiver<(bool, Vec3)>,
    pub rx_user_data: Receiver<UserData>,

    pub tx_user_data: Sender<UserData>,
    pub tx_sort_data: Sender<SortData>,
    pub tx_scene_data: Sender<SceneData>,
    pub tx_sort_time: Sender<f64>,
    pub tx_build_time: Sender<f64>,
}

#[derive(Clone, Copy, PartialEq)]
pub enum GUIStatus {
    Config,
    PostConfig,
    Render,
}

#[derive(PartialEq, Clone, Copy)]
pub enum SurfaceType {
    None,
    HeightMap,
    Sphere,
}

#[derive(PartialEq, Clone)]
pub enum HeightMapType {
    Texture,
    Random,
    SlopeX,
    SlopeY,
    DualSlope,
}

#[derive(PartialEq, Clone)]
pub enum TileSortType {
    Distance,
    Viewport,
    Object,
    Graph,
}

#[derive(PartialEq, Clone)]
pub enum SelectiveMergeType {
    None,
    Axis,
    Edge,
}

#[derive(Clone)]
pub struct SceneData {
    pub scene_id: u32,
    pub splat_count: usize,
    pub blending_splat_count: usize,
    pub center_coord: Vector2<i32>,
    pub lod_splat_count: Vec<usize>,
    pub lod_instance_count: Vec<usize>,
}
impl SceneData {
    pub fn new() -> Self {
        Self {
            scene_id: 0,
            splat_count: 0,
            blending_splat_count: 0,
            center_coord: vec2(0, 0),
            lod_splat_count: Vec::new(),
            lod_instance_count: Vec::new(),
        }
    }
}

#[derive(Clone)]
pub struct SortData {
    pub scene_id: u32,
    pub tile_instance_vec: Vec<TileInstance>,
    pub render_data_vec: Vec<(RenderDataKey, Option<RenderDataValue>)>,
}

#[derive(Clone, Debug)]
pub struct TileInstance {
    pub tid: (usize, usize), // (lod_id, tile_id)
    pub view_id: usize,
    pub tile_offset: Vec3,
    pub map_index: usize,
    pub map_coord: Vector2<usize>,
    pub tile_center: Vec3,
    pub merge_status: TileMergeStatus,
    pub transition_status: TileTransitionStatus,
    pub to_local: Mat3,

    pub corner_data: Option<TileCornerData>,
    pub edge_data: Option<TileEdgeData>,
}
impl TileInstance {
    pub fn new() -> Self {
        Self {
            tid: (0, 0),
            view_id: 0,
            tile_offset: Vec3::zero(),
            map_index: 0,
            map_coord: Vector2::zero(),
            tile_center: Vec3::zero(),
            merge_status: TileMergeStatus::None,
            transition_status: TileTransitionStatus::None,
            to_local: Mat3::zero(),

            corner_data: None,
            edge_data: None,
        }
    }

    pub fn from_metadata(tile_inst: &Self) -> Self {
        Self {
            tid: tile_inst.tid.clone(),
            view_id: tile_inst.view_id,
            tile_offset: tile_inst.tile_offset,
            map_index: tile_inst.map_index,
            map_coord: tile_inst.map_coord,
            tile_center: tile_inst.tile_center,
            merge_status: tile_inst.merge_status.clone(),
            transition_status: tile_inst.transition_status.clone(),
            to_local: tile_inst.to_local,
            corner_data: None,
            edge_data: None,
        }
    }
}

#[derive(Clone)]
pub struct TileBaseData {
    pub splat_count: usize,
    pub tile_center: Vec3,
    pub aabb: (Vec3, Vec3),

    pub raw_depth: Vec<i32>,
    pub gs_index: Vec<u32>,
    pub gs_lod_id: Vec<u32>,
}

#[derive(PartialEq, Clone, Debug)]
pub enum TileMergeStatus {
    None,
    MergedFrom(Vec<usize>),
    MergedTo(usize),
}

#[derive(PartialEq, Clone, Debug)]
pub enum TileTransitionStatus {
    None,
    Spawning(f32),  // blending_factor
    Changing(bool), // to_lower
}

#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub enum TileTransitionStatusHash {
    None,
    Spawning,
    Changing(bool),
}
impl TileTransitionStatusHash {
    pub fn from_status(status: &TileTransitionStatus) -> Self {
        match status {
            &TileTransitionStatus::None => Self::None,
            &TileTransitionStatus::Spawning(blend_f) => Self::Spawning,
            &TileTransitionStatus::Changing(to_lower) => Self::Changing(to_lower),
        }
    }
}

#[derive(Clone, Debug)]
pub struct TileCornerData {
    pub southwest: (Vec3, Mat3), // (corner pos, to_world)
    pub northwest: (Vec3, Mat3),
    pub northeast: (Vec3, Mat3),
    pub southeast: (Vec3, Mat3),
}
impl TileCornerData {
    pub fn new() -> Self {
        Self {
            southwest: (Vec3::zero(), Mat3::identity()),
            northwest: (Vec3::zero(), Mat3::identity()),
            northeast: (Vec3::zero(), Mat3::identity()),
            southeast: (Vec3::zero(), Mat3::identity()),
        }
    }
}
impl std::ops::Index<usize> for TileCornerData {
    type Output = (Vec3, Mat3);

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.southwest,
            1 => &self.northwest,
            2 => &self.northeast,
            3 => &self.southeast,
            _ => panic!("Index out of range: {}", index),
        }
    }
}
impl std::ops::IndexMut<usize> for TileCornerData {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.southwest,
            1 => &mut self.northwest,
            2 => &mut self.northeast,
            3 => &mut self.southeast,
            _ => panic!("Index out of range: {}", index),
        }
    }
}

#[derive(Clone, Debug)]
pub struct TileEdgeData {
    pub west: (Vec3, Vec3), // (edge pos, edge normal)
    pub north: (Vec3, Vec3),
    pub east: (Vec3, Vec3),
    pub south: (Vec3, Vec3),
}
impl TileEdgeData {
    pub fn new() -> Self {
        Self {
            west: (Vec3::zero(), Vec3::zero()),
            north: (Vec3::zero(), Vec3::zero()),
            east: (Vec3::zero(), Vec3::zero()),
            south: (Vec3::zero(), Vec3::zero()),
        }
    }
}
impl std::ops::Index<usize> for TileEdgeData {
    type Output = (Vec3, Vec3);

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.west,
            1 => &self.north,
            2 => &self.east,
            3 => &self.south,
            _ => panic!("Index out of range: {}", index),
        }
    }
}
impl std::ops::IndexMut<usize> for TileEdgeData {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.west,
            1 => &mut self.north,
            2 => &mut self.east,
            3 => &mut self.south,
            _ => panic!("Index out of range: {}", index),
        }
    }
}

#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub struct RenderDataKey {
    pub view_id: usize,
    pub tid: Vec<(usize, usize)>,
    pub transition_status: Vec<TileTransitionStatusHash>,
}
impl RenderDataKey {
    pub fn new() -> Self {
        Self {
            view_id: 0,
            tid: Vec::new(),
            transition_status: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RenderDataValue {
    pub splat_count: usize,
    pub gs_index: Vec<u32>,
    pub gs_map_id: Vec<u32>,
    pub merge_from_vec: Vec<usize>,
    pub single_lod_id: i32,
    pub gs_lod_id: Option<Vec<u32>>,
}

#[derive(Clone)]
pub struct MapNeighbor {
    pub west: Option<(Vector2<usize>, usize)>, // (map_coord, which neighbor this is for that)
    pub east: Option<(Vector2<usize>, usize)>,
    pub north: Option<(Vector2<usize>, usize)>,
    pub south: Option<(Vector2<usize>, usize)>,
}
impl MapNeighbor {
    pub fn new() -> Self {
        Self {
            west: None,
            east: None,
            north: None,
            south: None,
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &Option<(Vector2<usize>, usize)>> {
        [&self.west, &self.north, &self.east, &self.south].into_iter()
    }
}
impl std::ops::Index<usize> for MapNeighbor {
    type Output = Option<(Vector2<usize>, usize)>;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.west,
            1 => &self.north,
            2 => &self.east,
            3 => &self.south,
            _ => panic!("Index out of range: {}", index),
        }
    }
}

pub struct PreloadData<'a> {
    pub tile_splats_merged: &'a Scene,
    pub tile_base_data: &'a mut Vec<Vec<Vec<TileBaseData>>>,
    pub deformation_network: Option<DeformationNetwork>,
    pub basis_bank_motion: Option<std::sync::Arc<crate::basis_bank_motion::BasisBankMotionSet>>,
    pub catmull_rom_motion: Option<std::sync::Arc<crate::catmull_rom_motion::CatmullRomMotionSet>>,
    pub merged_orig_means: Option<Vec<[f32; 3]>>,
    pub merged_orig_quats: Option<Vec<[f32; 4]>>,
    // pub tile_spawning_data: &'a mut Vec<Vec<Vec<TileTransitionData>>>,
    // pub tile_changing_data: &'a mut Vec<Vec<Vec<TileTransitionData>>>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex2D {
    pub position: [f32; 2],
}
impl Vertex2D {
    const ATTRIBS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x2];

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;

        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

pub struct InputStatus {
    pub control_left: bool,
    pub shift_left: bool,
}
impl InputStatus {
    pub fn new() -> Self {
        Self {
            control_left: false,
            shift_left: false,
        }
    }

    pub fn update(&mut self, key: KeyCode, pressed: bool) {
        match key {
            KeyCode::ControlLeft => {
                self.control_left = pressed;
            }
            KeyCode::ShiftLeft => {
                self.shift_left = pressed;
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis_bank_motion::{
        BasisBankMotionMeta, BasisBankMotionSet, BasisInfo, BasisUsageStats,
    };

    fn test_basis_bank_preview(
        global_basis_count: usize,
        knot_count: usize,
    ) -> Arc<BasisBankMotionSet> {
        Arc::new(BasisBankMotionSet {
            meta: BasisBankMotionMeta {
                format: "test".to_string(),
                format_version: 1,
                delta_field: "delta_xyz".to_string(),
                basis_scope: "per_lod".to_string(),
                include_lods: vec![0],
                source_knot_count: knot_count,
                exported_knot_count: knot_count,
                loop_closure_knots: 0,
                loop_closure_method: "none".to_string(),
                motion_teacher: "test".to_string(),
                volume_res: None,
                volume_key_count: None,
                basis_count: global_basis_count,
                top_k: 1,
                fit_report_by_lod: serde_json::Value::Null,
            },
            motion_graph: None,
            total_splats: 0,
            global_basis_count,
            basis_infos: (0..global_basis_count)
                .map(|local_basis_id| BasisInfo {
                    lod_id: 0,
                    local_basis_id,
                })
                .collect(),
            usage_stats: vec![BasisUsageStats::default(); global_basis_count],
            global_basis_knots: vec![0.0; global_basis_count * knot_count * 3],
            global_basis_ids: Vec::new(),
            global_weights: Vec::new(),
        })
    }

    #[test]
    fn animation_toggle_only_changes_state_when_deformation_exists() {
        let mut animation_playing = true;

        assert!(toggle_animation_playing_state(
            false,
            &mut animation_playing
        ));
        assert!(animation_playing);

        assert!(!toggle_animation_playing_state(
            true,
            &mut animation_playing
        ));
        assert!(!animation_playing);

        assert!(toggle_animation_playing_state(true, &mut animation_playing));
        assert!(animation_playing);
    }

    #[test]
    fn smooth_ping_pong_maps_phase_to_smooth_deformation_time() {
        const EPS: f32 = 1e-6;

        assert!((smooth_ping_pong01(0.0) - 0.0).abs() < EPS);
        assert!((smooth_ping_pong01(0.125) - 0.25).abs() < EPS);
        assert!((smooth_ping_pong01(0.25) - 0.5).abs() < EPS);
        assert!((smooth_ping_pong01(0.375) - 0.75).abs() < EPS);
        assert!((smooth_ping_pong01(0.5) - 1.0).abs() < EPS);
        assert!((smooth_ping_pong01(0.75) - 0.5).abs() < EPS);
        assert!((smooth_ping_pong01(1.25) - smooth_ping_pong01(0.25)).abs() < EPS);
    }

    #[test]
    fn animation_delta_seconds_clamps_raw_frame_delta() {
        const EPS: f32 = 1e-6;

        assert!((animation_delta_seconds(16.0) - 0.016).abs() < EPS);
        assert!((animation_delta_seconds(1000.0) - 0.066).abs() < EPS);
        assert_eq!(animation_delta_seconds(-10.0), 0.0);
    }

    #[test]
    fn reset_frame_timing_resets_profiling_clock_without_changing_animation() {
        let mut rd = RenderData::new(1);
        rd.frame_time_ma.add(1000.0);
        rd.animation_phase = 0.25;
        rd.animation_time = 0.5;

        rd.reset_frame_timing(42.0);

        assert_eq!(rd.frame_prev, 42.0);
        assert_eq!(rd.frame_time_ma.calc().0, 0.0);
        assert_eq!(rd.animation_phase, 0.25);
        assert_eq!(rd.animation_time, 0.5);
    }

    #[test]
    fn motion_debug_defaults_preserve_current_behavior() {
        let rd = RenderData::new(1);

        assert!(rd.apply_network_delta_rot);
        assert!(!rd.manual_spline_knot_preview);
        assert_eq!(rd.selected_spline_knot, 0);
        assert!(!rd.motion_debug_dirty);
        assert_eq!(rd.active_motion_mode, MotionMode::Static);
        assert_eq!(rd.catmull_rom_knot_count, None);
        assert!(rd.basis_bank_preview.is_none());
        assert!(!rd.basis_preview_enabled);
        assert!(!rd.show_motion_authoring_menu);
        assert!(!rd.basis_best_target_preview_enabled);
        assert_eq!(rd.basis_preview_selected_id, 0);
        assert_eq!(rd.basis_graph_selected_segment, 0);
        assert_eq!(rd.basis_preview_projection, BasisPreviewProjection::XY);
        assert!(!rd.basis_preview_heatmap_enabled);
        assert_eq!(rd.basis_preview_heatmap_normalization, 1.0);
        assert!(rd.basis_knot_edits.is_none());
        assert!(!rd.basis_knot_edit_dirty);
        assert_eq!(rd.basis_knot_edit_selected_knot, 0);
        assert_eq!(rd.basis_knot_edit_dragging_knot, None);
        assert_eq!(rd.basis_view3d_yaw, 0.65);
        assert_eq!(rd.basis_view3d_pitch, 0.35);
        assert_eq!(rd.basis_view3d_zoom, 1.0);
        assert!(!rd.basis_graph_playback_config.enabled);
        assert!(!rd.basis_graph_playback_reset_requested);
        assert!(rd.basis_graph_playback_selected_state.is_none());
    }

    #[test]
    fn motion_debug_selected_knot_maps_to_exact_spline_time() {
        const EPS: f32 = 1e-6;

        assert!((spline_knot_preview_time(0, Some(32), false).unwrap() - 0.0).abs() < EPS);
        assert!((spline_knot_preview_time(31, Some(32), false).unwrap() - 31.0 / 32.0).abs() < EPS);
        assert!((spline_knot_preview_time(24, Some(25), true).unwrap() - 1.0).abs() < EPS);
        assert_eq!(spline_knot_preview_time(0, None, false), None);
        assert_eq!(spline_knot_preview_time(0, Some(0), false), None);
    }

    #[test]
    fn motion_debug_selected_knot_clamps_to_active_knot_count() {
        let mut rd = RenderData::new(1);
        rd.selected_spline_knot = 31;

        rd.set_motion_debug_backend(MotionMode::CatmullRom, Some(12), true, None, None, None);
        assert_eq!(rd.selected_spline_knot, 11);
        assert!(rd.catmull_rom_uses_volume_key_times);

        rd.set_motion_debug_backend(MotionMode::Static, None, false, None, None, None);
        assert_eq!(rd.selected_spline_knot, 0);
        assert!(!rd.manual_spline_knot_preview);
        assert!(!rd.catmull_rom_uses_volume_key_times);
    }

    #[test]
    fn basis_preview_selected_id_clamps_to_active_basis_count() {
        assert_eq!(clamp_basis_preview_id(99, Some(12)), 11);
        assert_eq!(clamp_basis_preview_id(2, Some(12)), 2);
        assert_eq!(clamp_basis_preview_id(99, Some(0)), 0);
        assert_eq!(clamp_basis_preview_id(99, None), 0);
    }

    #[test]
    fn basis_graph_selected_segment_clamps_without_preview_data() {
        assert_eq!(clamp_basis_graph_segment(99, None), 0);
    }

    #[test]
    fn basis_preview_enables_when_basis_bank_preview_becomes_available() {
        let mut rd = RenderData::new(1);

        rd.set_motion_debug_backend(
            MotionMode::BasisBank,
            Some(4),
            false,
            Some(2),
            Some(1),
            Some(test_basis_bank_preview(2, 4)),
        );

        assert!(rd.basis_preview_enabled);
        assert!(!rd.basis_best_target_preview_enabled);
    }

    #[test]
    fn basis_preview_manual_toggle_is_preserved_while_basis_bank_stays_available() {
        let mut rd = RenderData::new(1);
        rd.set_motion_debug_backend(
            MotionMode::BasisBank,
            Some(4),
            false,
            Some(2),
            Some(1),
            Some(test_basis_bank_preview(2, 4)),
        );
        rd.basis_preview_enabled = false;

        rd.set_motion_debug_backend(
            MotionMode::BasisBank,
            Some(4),
            false,
            Some(2),
            Some(1),
            Some(test_basis_bank_preview(2, 4)),
        );

        assert!(!rd.basis_preview_enabled);
    }

    #[test]
    fn basis_edit_overrides_resize_and_preserve_existing_values() {
        let mut rd = RenderData::new(1);
        rd.set_motion_debug_backend(
            MotionMode::BasisBank,
            Some(28),
            false,
            Some(3),
            Some(2),
            None,
        );
        assert_eq!(rd.basis_edit_overrides.len(), 3);
        assert!(
            rd.basis_edit_overrides
                .iter()
                .all(|edit| *edit == crate::basis_bank_edit::BasisEditOverride::default())
        );

        rd.basis_edit_overrides[1].enabled = true;
        rd.basis_edit_overrides[1].amplitude_scale = 2.0;
        rd.set_motion_debug_backend(
            MotionMode::BasisBank,
            Some(28),
            false,
            Some(3),
            Some(2),
            None,
        );
        assert!(rd.basis_edit_overrides[1].enabled);
        assert_eq!(rd.basis_edit_overrides[1].amplitude_scale, 2.0);

        rd.selected_spline_knot = 27;
        rd.basis_preview_selected_id = 2;
        rd.set_motion_debug_backend(
            MotionMode::BasisBank,
            Some(28),
            false,
            Some(1),
            Some(2),
            None,
        );
        assert_eq!(rd.basis_edit_overrides.len(), 1);
        assert_eq!(rd.basis_preview_selected_id, 0);
        assert_eq!(rd.selected_spline_knot, 27);
    }

    #[test]
    fn basis_edit_overrides_clear_when_basis_backend_disappears() {
        let mut rd = RenderData::new(1);
        rd.set_motion_debug_backend(
            MotionMode::BasisBank,
            Some(28),
            false,
            Some(2),
            Some(2),
            None,
        );
        rd.basis_edit_overrides[0].enabled = true;
        rd.mark_basis_edit_dirty();

        rd.set_motion_debug_backend(MotionMode::Static, None, false, None, None, None);

        assert!(rd.basis_edit_overrides.is_empty());
        assert!(!rd.basis_edit_dirty);
    }

    #[test]
    fn basis_edit_dirty_flag_can_be_marked_and_cleared() {
        let mut rd = RenderData::new(1);
        assert!(!rd.basis_edit_dirty);

        rd.mark_basis_edit_dirty();
        assert!(rd.basis_edit_dirty);

        rd.clear_basis_edit_dirty();
        assert!(!rd.basis_edit_dirty);
    }

    #[test]
    fn basis_knot_edits_initialize_from_basis_preview_data() {
        let mut motion = test_basis_bank_preview(2, 4);
        Arc::get_mut(&mut motion).unwrap().global_basis_knots = vec![
            0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, //
            2.0, 0.0, 0.0, //
            3.0, 0.0, 0.0, //
            10.0, 0.0, 0.0, //
            11.0, 0.0, 0.0, //
            12.0, 0.0, 0.0, //
            13.0, 0.0, 0.0, //
        ];
        let mut rd = RenderData::new(1);

        rd.set_motion_debug_backend(
            MotionMode::BasisBank,
            Some(4),
            false,
            Some(2),
            Some(1),
            Some(motion),
        );

        let edits = rd.basis_knot_edits.as_ref().unwrap();
        assert_eq!(edits.basis_count(), 2);
        assert_eq!(edits.knot_count(), 4);
        assert_eq!(edits.editable_knot_count(), 4);
        assert_eq!(edits.knot(1, 2), Some([12.0, 0.0, 0.0]));
        assert!(!rd.basis_knot_edit_dirty);
    }

    #[test]
    fn basis_knot_edits_clear_when_basis_backend_disappears() {
        let mut rd = RenderData::new(1);
        rd.set_motion_debug_backend(
            MotionMode::BasisBank,
            Some(4),
            false,
            Some(2),
            Some(1),
            Some(test_basis_bank_preview(2, 4)),
        );
        rd.mark_basis_knot_edit_dirty();

        rd.set_motion_debug_backend(MotionMode::Static, None, false, None, None, None);

        assert!(rd.basis_knot_edits.is_none());
        assert!(!rd.basis_knot_edit_dirty);
        assert_eq!(rd.basis_knot_edit_selected_knot, 0);
    }

    #[test]
    fn basis_knot_selected_knot_clamps_to_exported_knot_count() {
        let mut motion = test_basis_bank_preview(1, 6);
        Arc::get_mut(&mut motion).unwrap().meta.source_knot_count = 4;
        Arc::get_mut(&mut motion).unwrap().meta.loop_closure_knots = 2;
        let mut rd = RenderData::new(1);
        rd.basis_knot_edit_selected_knot = 99;

        rd.set_motion_debug_backend(
            MotionMode::BasisBank,
            Some(6),
            false,
            Some(1),
            Some(1),
            Some(motion),
        );

        assert_eq!(rd.basis_knot_edit_selected_knot, 5);
        assert_eq!(
            rd.basis_knot_edits
                .as_ref()
                .map(BasisKnotEditState::editable_knot_count),
            Some(6)
        );
    }

    #[test]
    fn basis_knot_edit_dirty_flag_can_be_marked_and_cleared() {
        let mut rd = RenderData::new(1);
        assert!(!rd.basis_knot_edit_dirty);

        rd.mark_basis_knot_edit_dirty();
        assert!(rd.basis_knot_edit_dirty);

        rd.clear_basis_knot_edit_dirty();
        assert!(!rd.basis_knot_edit_dirty);
    }

    #[test]
    fn basis_graph_playback_reset_request_marks_motion_debug_dirty() {
        let mut rd = RenderData::new(1);
        rd.request_basis_graph_playback_reset();

        assert!(rd.basis_graph_playback_reset_requested);
        assert!(rd.motion_debug_dirty);

        rd.clear_basis_graph_playback_reset();
        assert!(!rd.basis_graph_playback_reset_requested);
    }

    #[test]
    fn basis_weight_draw_mode_has_stable_shader_value() {
        assert_eq!(DrawMode::Normal as u32, 0);
        assert_eq!(DrawMode::View as u32, 4);
        assert_eq!(DrawMode::BasisWeight as u32, 5);
    }
}
