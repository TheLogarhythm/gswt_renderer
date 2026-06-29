// https://github.com/kaphula/winit-egui-wgpu-template/blob/master/src/egui_tools.rs

use std::{collections::HashSet, sync::Arc};

use egui::Context;
use egui_wgpu::wgpu::{CommandEncoder, Device, Queue, StoreOp, TextureFormat, TextureView};
use egui_wgpu::{Renderer, RendererOptions, ScreenDescriptor, wgpu};
use egui_winit::State;
use num_format::{Locale, ToFormattedString};
use winit::event::WindowEvent;
use winit::window::Window;

use crate::basis_bank_edit::{
    BasisEditOverride, BasisKnotEditPlane, BasisKnotEditState, apply_basis_knot_plane_delta,
    edited_basis_bank_delta, reset_all_basis_edits, reset_basis_edit,
};
use crate::basis_bank_motion::{
    BasisBankMotionSet, basis_bank_delta, basis_branch_continuity_debug,
};
use crate::basis_branch_regions::{BasisGraphBranchDomain, BasisGraphRegionConfig};
use crate::basis_graph_playback::{
    BasisBranchRejection, BasisGraphLastEdge, BasisGraphPlaybackPolicy, branch_rejection,
};
use crate::camera::Camera;
use crate::control::{CameraControl, FlyPathControl, FlyPathFrame};
use crate::log;
use crate::proxy::upload_proxy_texture;
use crate::skybox::upload_skybox;
use crate::structure::*;
use crate::utils::*;
use crate::wangtile::upload_height_map;

const SHOW_DEVELOPER_MOTION_DIAGNOSTICS: bool = false;
const GSWT_MAIN_WINDOW_ID: &str = "gswt_main_panel_v3";
const PERFORMANCE_WINDOW_ID: &str = "performance_panel_v3";
const MOTION_AUTHORING_WINDOW_ID: &str = "motion_authoring_panel_v4";
const GSWT_MAIN_DEFAULT_POS: [f32; 2] = [16.0, 16.0];
const GSWT_MAIN_DEFAULT_SIZE: [f32; 2] = [520.0, 560.0];
const PANEL_GAP: f32 = 24.0;
const MOTION_AUTHORING_DEFAULT_SIZE: [f32; 2] = [760.0, 820.0];

pub struct GUI {
    state: State,
    renderer: Renderer,
    frame_started: bool,

    pub gui_status: GUIStatus,
    pub config_user_data: UserData,
    config_user_data_string: UserDataString,
    config_confirmed: bool,
    config_lod_count_error_msg: Option<String>,
    config_error_msg: Option<String>,
    config_next_id: u32,
}

impl GUI {
    pub fn context(&self) -> &Context {
        self.state.egui_ctx()
    }

    pub fn new(device: &Device, output_color_format: TextureFormat, window: Arc<Window>) -> Self {
        let egui_context = Context::default();

        let egui_state = egui_winit::State::new(
            egui_context,
            egui::viewport::ViewportId::ROOT,
            window.as_ref(),
            Some(window.scale_factor() as f32),
            None,
            Some(2 * 1024), // default dimension is 2048
        );

        let egui_renderer_options = RendererOptions::default();

        let egui_renderer = Renderer::new(device, output_color_format, egui_renderer_options);

        Self {
            state: egui_state,
            renderer: egui_renderer,
            frame_started: false,

            gui_status: GUIStatus::Config,
            config_user_data: UserData::new(),
            config_user_data_string: UserDataString::new(),
            config_confirmed: false,
            config_lod_count_error_msg: None,
            config_error_msg: None,
            config_next_id: 0,
        }
    }

    pub fn handle_input(&mut self, window: Arc<Window>, event: &WindowEvent) {
        let _ = self.state.on_window_event(window.as_ref(), event);
    }

    pub fn render(
        &mut self,
        channels: &mut MainChannels,
        camera: &Camera,
        rd: &mut RenderData,
        fly_path_control: &mut FlyPathControl,
    ) {
        match self.gui_status {
            GUIStatus::Config => {
                self.config_lod_count_error_msg = None;
                if self.config_lod_count_error_msg.is_none() && self.config_confirmed {
                    self.config_error_msg = None;
                    self.config_user_data.config_id = self.config_next_id;
                    self.config_user_data_string
                        .to_raw(&mut self.config_user_data, &mut self.config_error_msg);

                    if let Some(rx) = &channels.rx_height_tex {
                        if let Ok(height_tex) = rx.try_recv() {
                            self.config_user_data.height_tex = Some(height_tex);
                            channels.rx_height_tex = None;
                        }
                    }

                    if let Some(rx) = &channels.rx_skybox_tex {
                        if let Ok(skybox_tex) = rx.try_recv() {
                            rd.skybox_rawtex = Some(skybox_tex);
                            channels.rx_skybox_tex = None;
                        }
                    }

                    if let Some(rx) = &channels.rx_proxy_tex {
                        if let Ok(proxy_tex) = rx.try_recv() {
                            rd.proxy_rawtex = Some(proxy_tex);
                            channels.rx_proxy_tex = None;
                        }
                    }

                    if self.config_error_msg.is_none() {
                        log!("Config {} confirmed.", self.config_next_id);
                        self.config_confirmed = false;
                        channels
                            .tx_user_data
                            .send(self.config_user_data.clone())
                            .expect("Error sending user data to worker thread.");
                        self.gui_status = GUIStatus::PostConfig;
                        self.config_next_id += 1;
                        return;
                    }
                }

                self.config_confirmed = false;

                egui::Window::new("GSWT")
                    .vscroll(true)
                    .show(&self.context().clone(), |ui| {
                        egui::Grid::new("my_grid")
                            .num_columns(7)
                            .spacing([40.0, 4.0])
                            .striped(true)
                            .show(ui, |ui| {
                                ui.label("Tile map");
                                ui.label("Width (half)");
                                ui.text_edit_singleline(
                                    &mut self.config_user_data_string.tile_map_half_wh_s.x,
                                );
                                ui.label("Height (half)");
                                ui.text_edit_singleline(
                                    &mut self.config_user_data_string.tile_map_half_wh_s.y,
                                );
                                ui.end_row();

                                // ui.separator();
                                // ui.end_row();
                                ui.label("Center option");
                                ui.text_edit_singleline(
                                    &mut self.config_user_data_string.center_option_s,
                                );
                                ui.end_row();

                                ui.label("Update distance tolerance");
                                ui.text_edit_singleline(
                                    &mut self.config_user_data_string.update_dist_s,
                                );
                                ui.end_row();

                                // ui.separator();
                                // ui.end_row();
                                ui.label("Tile width");
                                ui.text_edit_singleline(
                                    &mut self.config_user_data_string.tile_width_s,
                                );
                                ui.end_row();

                                ui.label("Tile sort type");
                                ui.selectable_value(
                                    &mut self.config_user_data.tile_sort_type,
                                    TileSortType::Distance,
                                    "Distance",
                                );
                                ui.selectable_value(
                                    &mut self.config_user_data.tile_sort_type,
                                    TileSortType::Viewport,
                                    "Viewport",
                                );
                                ui.selectable_value(
                                    &mut self.config_user_data.tile_sort_type,
                                    TileSortType::Object,
                                    "Object",
                                );
                                ui.selectable_value(
                                    &mut self.config_user_data.tile_sort_type,
                                    TileSortType::Graph,
                                    "Graph",
                                );
                                ui.end_row();

                                ui.separator();
                                ui.end_row();

                                ui.label("Selective merge");
                                ui.selectable_value(
                                    &mut self.config_user_data.merge_type,
                                    SelectiveMergeType::None,
                                    "None",
                                );
                                ui.selectable_value(
                                    &mut self.config_user_data.merge_type,
                                    SelectiveMergeType::Axis,
                                    "Axis",
                                );
                                ui.selectable_value(
                                    &mut self.config_user_data.merge_type,
                                    SelectiveMergeType::Edge,
                                    "Edge",
                                );
                                ui.end_row();

                                match self.config_user_data.merge_type {
                                    SelectiveMergeType::Axis => {
                                        ui.label("Merge tile distance");
                                        ui.text_edit_singleline(
                                            &mut self.config_user_data_string.merge_tile_dist_s.x,
                                        );
                                        ui.label("to");
                                        ui.text_edit_singleline(
                                            &mut self.config_user_data_string.merge_tile_dist_s.y,
                                        );
                                        ui.end_row();
                                    }
                                    SelectiveMergeType::Edge => {
                                        ui.label("Merge dot threshold");
                                        ui.text_edit_singleline(
                                            &mut self.config_user_data_string.merge_dot_threshold_s,
                                        );
                                        ui.end_row();

                                        ui.label("Merge top-k");
                                        ui.text_edit_singleline(
                                            &mut self.config_user_data_string.merge_topk_s,
                                        );
                                        ui.end_row();
                                    }
                                    SelectiveMergeType::None => {}
                                }
                                if self.config_user_data.merge_type != SelectiveMergeType::None {
                                    ui.label("Use merge cache");
                                    ui.checkbox(&mut self.config_user_data.use_cache, "");
                                    ui.end_row();

                                    if self.config_user_data.use_cache {
                                        ui.label("Cache size");
                                        ui.text_edit_singleline(
                                            &mut self.config_user_data_string.cache_size_s,
                                        );
                                        ui.end_row();
                                    }
                                }

                                ui.separator();
                                ui.end_row();

                                ui.label("Dynamic LOD");
                                ui.end_row();
                                ui.label("Max Distance");
                                ui.text_edit_singleline(
                                    &mut self.config_user_data_string.lod_max_dist_s,
                                );
                                ui.label("x Tile Width");
                                ui.end_row();

                                ui.add(egui::Label::new("LOD blending"));
                                ui.checkbox(&mut self.config_user_data.lod_blending, "");
                                ui.end_row();

                                if self.config_user_data.lod_blending {
                                    ui.label("Blending width ratio");
                                    ui.text_edit_singleline(
                                        &mut self
                                            .config_user_data_string
                                            .lod_transition_width_ratio_s,
                                    );
                                    ui.end_row();

                                    ui.label("Precise bbox check");
                                    ui.checkbox(&mut self.config_user_data.lod_bbox_check, "");
                                    ui.end_row();

                                    ui.label("Blending distance tolerance");
                                    ui.text_edit_singleline(
                                        &mut self.config_user_data_string.lod_dist_tolerance_s,
                                    );
                                    ui.end_row();
                                }

                                ui.separator();
                                ui.end_row();

                                ui.label("Surface mapping");
                                ui.selectable_value(
                                    &mut self.config_user_data.surface_type,
                                    SurfaceType::None,
                                    "None",
                                );
                                ui.selectable_value(
                                    &mut self.config_user_data.surface_type,
                                    SurfaceType::HeightMap,
                                    "HeightMap",
                                );
                                ui.selectable_value(
                                    &mut self.config_user_data.surface_type,
                                    SurfaceType::Sphere,
                                    "Sphere",
                                );
                                ui.end_row();

                                match self.config_user_data.surface_type {
                                    SurfaceType::HeightMap => {
                                        ui.label("Height map");
                                        ui.label("Width");
                                        ui.text_edit_singleline(
                                            &mut self.config_user_data_string.height_map_wh_s.x,
                                        );
                                        ui.label("Height");
                                        ui.text_edit_singleline(
                                            &mut self.config_user_data_string.height_map_wh_s.y,
                                        );
                                        ui.end_row();

                                        ui.label("Type");
                                        ui.selectable_value(
                                            &mut self.config_user_data.height_map_type,
                                            HeightMapType::Texture,
                                            "Texture",
                                        );
                                        ui.selectable_value(
                                            &mut self.config_user_data.height_map_type,
                                            HeightMapType::Random,
                                            "Random",
                                        );
                                        ui.selectable_value(
                                            &mut self.config_user_data.height_map_type,
                                            HeightMapType::SlopeX,
                                            "SlopeX",
                                        );
                                        // ui.selectable_value(&mut height_map_type, HeightMapType::SlopeY, "SlopeY");
                                        // ui.selectable_value(&mut height_map_type, HeightMapType::DualSlope, "Dual Slope");
                                        ui.end_row();

                                        if self.config_user_data.height_map_type
                                            == HeightMapType::Texture
                                        {
                                            ui.label("Height texture");
                                            if ui.button("Upload").clicked() {
                                                channels.rx_height_tex = Some(upload_height_map());
                                            }
                                            ui.end_row();
                                        }

                                        ui.label("Scale (Hori)");
                                        ui.text_edit_singleline(
                                            &mut self.config_user_data_string.height_map_scale_s.x,
                                        );
                                        ui.label("Scale (Vert)");
                                        ui.text_edit_singleline(
                                            &mut self.config_user_data_string.height_map_scale_s.y,
                                        );
                                        ui.end_row();
                                    }
                                    SurfaceType::Sphere => {
                                        ui.label("Sphere radius");
                                        ui.text_edit_singleline(
                                            &mut self.config_user_data_string.sphere_radius_s,
                                        );
                                        ui.end_row();
                                    }
                                    SurfaceType::None => {}
                                }

                                ui.separator();
                                ui.end_row();

                                ui.label("Skybox Texture");
                                if ui.button("Upload").clicked() {
                                    channels.rx_skybox_tex = Some(upload_skybox());
                                }
                                ui.end_row();

                                ui.label("Proxy Texture");
                                if ui.button("Upload").clicked() {
                                    channels.rx_proxy_tex = Some(upload_proxy_texture());
                                }
                                ui.end_row();

                                ui.label("Reset Rng");
                                ui.checkbox(&mut self.config_user_data.reset_rng, "");
                                ui.label("Always Sort");
                                ui.checkbox(&mut self.config_user_data.always_sort, "");

                                if ui.button("Confirm").clicked() {
                                    self.config_confirmed = true;
                                }
                                ui.end_row();
                            });

                        if self.config_lod_count_error_msg.is_some() {
                            ui.label(self.config_lod_count_error_msg.clone().unwrap());
                            ui.end_row();
                        }

                        if self.config_error_msg.is_some() {
                            ui.label(self.config_error_msg.clone().unwrap());
                            ui.end_row();
                        }
                    });
            }
            GUIStatus::PostConfig => {
                // todo!()
            }
            GUIStatus::Render => {
                if rd.show_main_menu {
                    egui::Window::new("GSWT")
                        .id(egui::Id::new(GSWT_MAIN_WINDOW_ID))
                        .default_pos(egui::pos2(
                            GSWT_MAIN_DEFAULT_POS[0],
                            GSWT_MAIN_DEFAULT_POS[1],
                        ))
                        .default_size(egui::vec2(
                            GSWT_MAIN_DEFAULT_SIZE[0],
                            GSWT_MAIN_DEFAULT_SIZE[1],
                        ))
                        .vscroll(true)
                        .show(&self.context().clone(), |ui| {
                            egui::Grid::new("gswt_grid")
                                .num_columns(2)
                                .spacing([40.0, 4.0])
                                .striped(true)
                                .show(ui, |ui| {
                                    let frame_time = rd.frame_time_ma.calc();
                                    ui.add(egui::Label::new("FPS"));
                                    ui.label(format!("{:.2}", 1000.0 / frame_time.0));
                                    ui.end_row();

                                    ui.add(egui::Label::new("Render Time (ms)"));
                                    ui.label(format!("{:.2}±{:.2}", frame_time.0, frame_time.1));
                                    ui.end_row();

                                    let sort_time = rd.sort_time_ma.calc();
                                    let (sort_trigger, _) = rd.sort_trigger_ma.calc();
                                    ui.add(egui::Label::new("Sort Time (ms)"));
                                    ui.label(format!(
                                        "{:.2}±{:.2} ({:.2}%)",
                                        sort_time.0,
                                        sort_time.1,
                                        sort_trigger * 100.0
                                    ));
                                    ui.end_row();

                                    let build_time = rd.build_time_ma.calc();
                                    let (build_trigger, _) = rd.build_trigger_ma.calc();
                                    ui.add(egui::Label::new("Update Time (ms)"));
                                    ui.label(format!(
                                        "{:.2}±{:.2} ({:.2}%)",
                                        build_time.0,
                                        build_time.1,
                                        build_trigger * 100.0
                                    ));
                                    ui.end_row();

                                    ui.label("Timer Avg Window");
                                    ui.add(egui::Slider::new(&mut rd.time_ma_window, 1..=5000));
                                    ui.end_row();

                                    if ui.button("Reset Timer").clicked() {
                                        rd.frame_time_ma = IncrementalMA::new(rd.time_ma_window);
                                        rd.sort_time_ma = IncrementalMA::new(rd.time_ma_window);
                                        rd.build_time_ma = IncrementalMA::new(rd.time_ma_window);
                                        rd.sort_trigger_ma = IncrementalMA::new(rd.time_ma_window);
                                        rd.build_trigger_ma = IncrementalMA::new(rd.time_ma_window);
                                    }
                                    ui.end_row();

                                    let mut splat_count: usize = 0;
                                    let mut blending_splat_count: usize = 0;
                                    if rd.cur_scene_data.is_some() {
                                        splat_count =
                                            rd.cur_scene_data.as_ref().unwrap().splat_count;
                                        blending_splat_count = rd
                                            .cur_scene_data
                                            .as_ref()
                                            .unwrap()
                                            .blending_splat_count;
                                    }
                                    ui.add(egui::Label::new("Splat Count (With Blending)"));
                                    ui.label(format!(
                                        "{} ({})",
                                        splat_count.to_formatted_string(&Locale::en),
                                        blending_splat_count.to_formatted_string(&Locale::en)
                                    ));
                                    ui.end_row();

                                    ui.add(egui::Label::new("Visualization"));
                                    ui.horizontal(|ui| {
                                        ui.selectable_value(
                                            &mut rd.render_config.draw_mode,
                                            DrawMode::Normal,
                                            "Normal",
                                        );
                                        ui.selectable_value(
                                            &mut rd.render_config.draw_mode,
                                            DrawMode::TileID,
                                            "TileID",
                                        );
                                        ui.selectable_value(
                                            &mut rd.render_config.draw_mode,
                                            DrawMode::TileLOD,
                                            "LOD",
                                        );
                                        // ui.selectable_value(
                                        //     &mut rd.render_config.draw_mode,
                                        //     DrawMode::LOD,
                                        //     "LOD",
                                        // );
                                        ui.selectable_value(
                                            &mut rd.render_config.draw_mode,
                                            DrawMode::View,
                                            "View",
                                        );
                                        ui.add_enabled_ui(rd.has_motion, |ui| {
                                            ui.selectable_value(
                                                &mut rd.render_config.draw_mode,
                                                DrawMode::BasisWeight,
                                                "Basis",
                                            );
                                        });
                                    });
                                    ui.end_row();

                                    ui.add(egui::Label::new("Point Cloud"));
                                    ui.checkbox(&mut rd.render_config.draw_point_cloud, "");
                                    ui.end_row();

                                    if rd.render_config.draw_point_cloud {
                                        ui.add(egui::Slider::new(
                                            &mut rd.render_config.point_cloud_radius,
                                            0.001..=1.0,
                                        ));
                                        ui.end_row();
                                    }

                                    ui.add(egui::Label::new("Culling Threshold"));
                                    ui.add(egui::Slider::new(
                                        &mut rd.render_config.culling_dist,
                                        0.0..=10.0,
                                    ));
                                    ui.end_row();

                                    ui.add(egui::Label::new("Render GS"));
                                    ui.checkbox(&mut rd.render_gs, "");
                                    ui.end_row();

                                    if rd.has_motion {
                                        self.draw_dynamics_status_ui(ui, rd);
                                    }

                                    if rd.use_skybox {
                                        ui.add(egui::Label::new("Skybox"));
                                        ui.checkbox(&mut rd.use_skybox, "");
                                        ui.end_row();
                                    }

                                    if rd.use_proxy {
                                        ui.collapsing("Proxy", |ui| {
                                            ui.add(egui::Label::new("Full Proxy"));
                                            ui.checkbox(&mut rd.render_config.proxy_full, "");
                                            ui.end_row();

                                            ui.add(egui::Label::new("Map Proxy"));
                                            ui.checkbox(&mut rd.render_config.proxy_map, "");
                                            ui.end_row();

                                            ui.add(egui::Label::new("Proxy Height"));
                                            ui.add(egui::Slider::new(
                                                &mut rd.render_config.proxy_height,
                                                -20.0..=20.0,
                                            ));
                                            ui.end_row();

                                            ui.add(egui::Label::new("Proxy Width Scale"));
                                            ui.add(egui::Slider::new(
                                                &mut rd.render_config.proxy_width_scale,
                                                0.1..=50.0,
                                            ));
                                            ui.end_row();

                                            ui.add(egui::Label::new("Proxy Brightness"));
                                            ui.add(egui::Slider::new(
                                                &mut rd.render_config.proxy_brightness,
                                                0.0..=3.0,
                                            ));
                                            ui.end_row();

                                            ui.add(egui::Label::new("Proxy black background"));
                                            ui.checkbox(
                                                &mut rd.render_config.proxy_black_background,
                                                "",
                                            );
                                        });
                                        ui.end_row();
                                    }

                                    ui.label("Clip By Z");
                                    ui.checkbox(&mut rd.render_config.use_clip, "");
                                    ui.end_row();

                                    if rd.render_config.use_clip {
                                        ui.label("Clip Height");
                                        ui.add(egui::Slider::new(
                                            &mut rd.render_config.clip_height,
                                            -20.0..=20.0,
                                        ));
                                        ui.end_row();
                                    }

                                    ui.add(egui::Label::new("Lock (Sort)"));
                                    ui.checkbox(&mut rd.lock_sort, "");
                                    ui.end_row();

                                    ui.add(egui::Label::new("Lock (Tile)"));
                                    ui.checkbox(&mut rd.lock_tile, "");
                                    ui.end_row();

                                    ui.collapsing("Scales", |ui| {
                                        ui.add(egui::Label::new("Splat Scale"));
                                        ui.add(egui::Slider::new(
                                            &mut rd.render_config.splat_scale,
                                            0.01..=10.0,
                                        ));
                                        ui.end_row();

                                        ui.label("Height Map Scale V");
                                        ui.add(egui::Slider::new(
                                            &mut rd.render_config.height_map_scale_v,
                                            0.0..=20.0,
                                        ));
                                        ui.end_row();

                                        ui.label("Scene Scale X");
                                        ui.add(egui::Slider::new(
                                            &mut rd.render_config.scene_scale.x,
                                            0.0..=3.0,
                                        ));
                                        ui.end_row();

                                        ui.label("Scene Scale Y");
                                        ui.add(egui::Slider::new(
                                            &mut rd.render_config.scene_scale.y,
                                            0.0..=3.0,
                                        ));
                                        ui.end_row();

                                        ui.label("Scene Scale Z");
                                        ui.add(egui::Slider::new(
                                            &mut rd.render_config.scene_scale.z,
                                            0.0..=3.0,
                                        ));
                                        ui.end_row();
                                    });
                                    ui.end_row();

                                    let cam_pos = camera.position();
                                    ui.add(egui::Label::new("Camera Position"));
                                    ui.label(format!(
                                        "({:.2}, {:.2}, {:.2})",
                                        cam_pos.x, cam_pos.y, cam_pos.z
                                    ));
                                    ui.end_row();

                                    let cam_dir = camera.view_direction();
                                    ui.add(egui::Label::new("Camera Direction"));
                                    ui.label(format!(
                                        "({:.2}, {:.2}, {:.2})",
                                        cam_dir.x, cam_dir.y, cam_dir.z
                                    ));
                                    ui.end_row();

                                    let cam_right = camera.right_direction();
                                    ui.add(egui::Label::new("Camera Right"));
                                    ui.label(format!(
                                        "({:.2}, {:.2}, {:.2})",
                                        cam_right.x, cam_right.y, cam_right.z
                                    ));
                                    ui.end_row();

                                    let cam_up = camera.up();
                                    ui.add(egui::Label::new("Camera Up"));
                                    ui.label(format!(
                                        "({:.2}, {:.2}, {:.2})",
                                        cam_up.x, cam_up.y, cam_up.z
                                    ));
                                    ui.end_row();

                                    ui.add(egui::Label::new("Camera Position"));
                                    ui.horizontal(|ui| {
                                        ui.add(
                                            egui::TextEdit::singleline(&mut rd.cam_pos_s.x)
                                                .desired_width(40.0),
                                        );
                                        ui.add(
                                            egui::TextEdit::singleline(&mut rd.cam_pos_s.y)
                                                .desired_width(40.0),
                                        );
                                        ui.add(
                                            egui::TextEdit::singleline(&mut rd.cam_pos_s.z)
                                                .desired_width(40.0),
                                        );
                                    });
                                    // ui.label(format!("({:.2}, {:.2}, {:.2})", cam_pos.x, cam_pos.y, cam_pos.z));
                                    ui.end_row();

                                    ui.add(egui::Label::new("Camera Direction"));
                                    ui.horizontal(|ui| {
                                        ui.add(
                                            egui::TextEdit::singleline(&mut rd.cam_dir_s.x)
                                                .desired_width(40.0),
                                        );
                                        ui.add(
                                            egui::TextEdit::singleline(&mut rd.cam_dir_s.y)
                                                .desired_width(40.0),
                                        );
                                        ui.add(
                                            egui::TextEdit::singleline(&mut rd.cam_dir_s.z)
                                                .desired_width(40.0),
                                        );
                                    });
                                    // ui.label(format!("({:.2}, {:.2}, {:.2})", cam_dir.x, cam_dir.y, cam_dir.z));
                                    ui.end_row();

                                    ui.add(egui::Label::new("Camera Up"));
                                    ui.horizontal(|ui| {
                                        ui.add(
                                            egui::TextEdit::singleline(&mut rd.cam_up_s.x)
                                                .desired_width(40.0),
                                        );
                                        ui.add(
                                            egui::TextEdit::singleline(&mut rd.cam_up_s.y)
                                                .desired_width(40.0),
                                        );
                                        ui.add(
                                            egui::TextEdit::singleline(&mut rd.cam_up_s.z)
                                                .desired_width(40.0),
                                        );
                                    });
                                    // ui.label(format!("({:.2}, {:.2}, {:.2})", cam_dir.x, cam_dir.y, cam_dir.z));
                                    ui.end_row();

                                    if ui.button("Set camera").clicked() {
                                        rd.parse_camera_config();
                                    }
                                    ui.end_row();

                                    ui.add(egui::Label::new("Camera Control"));
                                    ui.horizontal(|ui| {
                                        ui.selectable_value(
                                            &mut rd.camera_control_type,
                                            CameraControl::KeyboardFly,
                                            "Keyboard Fly",
                                        );
                                        ui.selectable_value(
                                            &mut rd.camera_control_type,
                                            CameraControl::FlyPath,
                                            "Fly Path",
                                        );
                                    });
                                    ui.end_row();

                                    ui.label("Fly Path Menu");
                                    ui.checkbox(&mut rd.show_fly_path_menu, "");
                                    ui.end_row();

                                    // Debug
                                    // if !rd.freeze_frame {
                                    //     if ui.button("Freeze").clicked() {
                                    //         rd.freeze_frame = true;
                                    //     }
                                    // } else {
                                    //     if ui.button("Unfreeze").clicked() {
                                    //         rd.freeze_frame = false;
                                    //         rd.step_frame = false;
                                    //         rd.render_config.debug_log = false;
                                    //     }
                                    //     if ui.button("Step").clicked() {
                                    //         rd.step_frame = true;
                                    //     }
                                    //     ui.end_row();
                                    //     ui.label("Debug");
                                    //     ui.checkbox(&mut rd.render_config.debug_log, "");
                                    // }
                                    // ui.end_row();

                                    if ui.button("Reconfig scene").clicked() {
                                        self.gui_status = GUIStatus::Config;
                                    }
                                    ui.end_row();
                                });
                        });
                }

                if rd.show_perf_menu {
                    egui::Window::new("Performance")
                        .id(egui::Id::new(PERFORMANCE_WINDOW_ID))
                        .default_pos(egui::pos2(
                            performance_default_pos()[0],
                            performance_default_pos()[1],
                        ))
                        .default_width(GSWT_MAIN_DEFAULT_SIZE[0])
                        .show(&self.context().clone(), |ui| {
                            self.draw_performance_ui(ui, rd);
                        });
                }

                self.draw_motion_authoring_window(rd);

                // Fly Path Control Window
                if rd.show_fly_path_menu {
                    egui::Window::new("Fly Path")
                        .show(self.context(), |ui| {
                        egui::Grid::new("fly_path_grid")
                            .num_columns(2)
                            .spacing([40.0, 4.0])
                            .striped(true)
                            .show(ui, |ui| {
                            if let Some(rx) = &channels.rx_fly_path_control {
                                if let Ok(new_control) = rx.try_recv() {
                                    *fly_path_control = new_control;
                                    channels.rx_fly_path_control = None;
                                }
                            }
                            for i in 0..fly_path_control.keyframes.len() {

                                ui.add(egui::Label::new(format!("Keyframe")));
                                ui.add(egui::Label::new(i.to_string()));
                                ui.end_row();

                                ui.add(egui::Label::new("Timestamp (s)"));
                                ui.add(egui::TextEdit::singleline(&mut fly_path_control.keyframes[i].timestamp_s));
                                ui.end_row();

                                ui.add(egui::Label::new("Position"));
                                ui.horizontal(|ui| {
                                    ui.add(egui::TextEdit::singleline(&mut fly_path_control.keyframes[i].position_s.x).desired_width(40.0));
                                    ui.add(egui::TextEdit::singleline(&mut fly_path_control.keyframes[i].position_s.y).desired_width(40.0));
                                    ui.add(egui::TextEdit::singleline(&mut fly_path_control.keyframes[i].position_s.z).desired_width(40.0));
                                });
                                ui.end_row();

                                ui.add(egui::Label::new("Target"));
                                ui.horizontal(|ui| {
                                    ui.add(egui::TextEdit::singleline(&mut fly_path_control.keyframes[i].target_s.x).desired_width(40.0));
                                    ui.add(egui::TextEdit::singleline(&mut fly_path_control.keyframes[i].target_s.y).desired_width(40.0));
                                    ui.add(egui::TextEdit::singleline(&mut fly_path_control.keyframes[i].target_s.z).desired_width(40.0));
                                });
                                ui.end_row();
                            }

                            if ui.button("Add").clicked() {
                                let keyframe = FlyPathFrame::from_camera(&camera);
                                fly_path_control.keyframes.push(keyframe);
                            }
                            if ui.button("Remove").clicked() {
                                fly_path_control.keyframes.pop();
                            }
                            ui.end_row();

                            if ui.button("Upload").clicked() {
                                channels.rx_fly_path_control = Some(FlyPathControl::upload());
                            }
                            if ui.button("Download").clicked() {
                                FlyPathControl::download(&fly_path_control.keyframes);
                            }
                            ui.end_row();

                            ui.label("Hide Menu");
                            ui.checkbox(&mut rd.hide_menu_when_start, "");
                            ui.end_row();

                            ui.add(egui::Label::new("Elapsed Time (s)"));
                            ui.add(egui::Label::new(format!("{:.2}", fly_path_control.timer.elapsed() / 1000.0)));
                            ui.end_row();

                            if ui.button("Reset").clicked() {
                                rd.fly_path_error_msg = fly_path_control.reset_path();
                            }

                            if fly_path_control.ready && !fly_path_control.finished {
                                if fly_path_control.timer.is_paused() {
                                    if ui.button("Start").clicked() {
                                        if rd.hide_menu_when_start {
                                            rd.show_fly_path_menu = false;
                                            rd.show_main_menu = false;
                                        }
                                        fly_path_control.start_path();

                                        // Start benchmark
                                        rd.fly_path_benchmark = true;
                                        rd.frame_time_ma.clear();
                                        rd.sort_time_ma.clear();
                                        rd.build_time_ma.clear();
                                        rd.sort_trigger_ma.clear();
                                        rd.build_trigger_ma.clear();
                                    }
                                } else {
                                    if ui.button("Pause").clicked() {
                                        fly_path_control.pause_path();
                                    }
                                }
                            } else if rd.fly_path_error_msg.is_some() {
                                ui.end_row();
                                ui.label(rd.fly_path_error_msg.clone().unwrap());
                            } else if fly_path_control.finished && rd.fly_path_benchmark {
                                // End benchmark
                                rd.fly_path_benchmark = false;
                                let (f_mean, f_std) = rd.frame_time_ma.calc();
                                let (s_mean, s_std) = rd.sort_time_ma.calc();
                                let (b_mean, b_std) = rd.build_time_ma.calc();
                                let (s_t, _) = rd.sort_trigger_ma.calc();
                                let (b_t, _) = rd.build_trigger_ma.calc();
                                let s_t = s_t * 100.0;
                                let b_t = b_t * 100.0;
                                log!("Render & Sort & Update");
                                log!(r"\( {f_mean:.2} \pm {f_std:.2} \) & \( {s_mean:.2} \pm {s_std:.2} \; ({s_t:.2}\%) \) & \( {b_mean:.2} \pm {b_std:.2} \; ({b_t:.2}\%) \)");
                                rd.frame_time_ma.clear();
                                rd.sort_time_ma.clear();
                                rd.build_time_ma.clear();
                                rd.sort_trigger_ma.clear();
                                rd.build_trigger_ma.clear();
                            }
                            ui.end_row();
                        });
                    });
                }
            }
        }
    }

    fn draw_dynamics_status_ui(&self, ui: &mut egui::Ui, rd: &mut RenderData) {
        ui.add(egui::Label::new("Dynamics"));
        ui.vertical(|ui| {
            ui.label(if rd.animation_playing {
                "Playing"
            } else {
                "Frozen"
            });
            ui.label(basis_bank_status_summary(rd));

            let button_label = if rd.show_motion_authoring_menu {
                "Close Motion Authoring"
            } else {
                "Open Motion Authoring"
            };
            if ui.button(button_label).clicked() {
                rd.show_motion_authoring_menu = !rd.show_motion_authoring_menu;
            }
            ui.label("Shortcut: B");
        });
        ui.end_row();
    }

    fn draw_motion_authoring_window(&self, rd: &mut RenderData) {
        if !rd.show_motion_authoring_menu {
            return;
        }

        let mut open = rd.show_motion_authoring_menu;
        egui::Window::new("Motion Authoring")
            .id(egui::Id::new(MOTION_AUTHORING_WINDOW_ID))
            .default_pos(egui::pos2(
                motion_authoring_default_pos()[0],
                motion_authoring_default_pos()[1],
            ))
            .default_size(egui::vec2(
                MOTION_AUTHORING_DEFAULT_SIZE[0],
                MOTION_AUTHORING_DEFAULT_SIZE[1],
            ))
            .vscroll(true)
            .open(&mut open)
            .show(&self.context().clone(), |ui| {
                self.draw_motion_authoring_contents(ui, rd);
            });
        rd.show_motion_authoring_menu = open;
    }

    fn draw_motion_authoring_contents(&self, ui: &mut egui::Ui, rd: &mut RenderData) {
        ui.horizontal_wrapped(|ui| {
            if ui
                .button(if rd.animation_playing {
                    "Pause"
                } else {
                    "Play"
                })
                .clicked()
            {
                rd.animation_playing = !rd.animation_playing;
            }
            ui.label(format!("Archive duration: {:.2}s", rd.animation_duration));
            ui.add(egui::Slider::new(&mut rd.animation_speed, 0.0..=4.0).text("Playback speed"));
            ui.separator();
            ui.label(basis_bank_status_summary(rd));
        });
        ui.separator();

        self.draw_basis_motion_ui(ui, rd);
        if SHOW_DEVELOPER_MOTION_DIAGNOSTICS {
            self.draw_motion_debug_ui(ui, rd);
        }
    }

    fn draw_performance_ui(&self, ui: &mut egui::Ui, rd: &mut RenderData) {
        egui::Grid::new("performance_grid")
            .num_columns(2)
            .spacing([40.0, 4.0])
            .striped(true)
            .show(ui, |ui| {
                let frame_time = rd.frame_time_ma.calc();
                ui.add(egui::Label::new("FPS"));
                ui.label(format!("{:.2}", 1000.0 / frame_time.0));
                ui.end_row();

                ui.add(egui::Label::new("Render Time (ms)"));
                ui.label(format!("{:.2}±{:.2}", frame_time.0, frame_time.1));
                ui.end_row();

                let sort_time = rd.sort_time_ma.calc();
                let (sort_trigger, _) = rd.sort_trigger_ma.calc();
                ui.add(egui::Label::new("Sort Time (ms)"));
                ui.label(format!(
                    "{:.2}±{:.2} ({:.2}%)",
                    sort_time.0,
                    sort_time.1,
                    sort_trigger * 100.0
                ));
                ui.end_row();

                let build_time = rd.build_time_ma.calc();
                let (build_trigger, _) = rd.build_trigger_ma.calc();
                ui.add(egui::Label::new("Update Time (ms)"));
                ui.label(format!(
                    "{:.2}±{:.2} ({:.2}%)",
                    build_time.0,
                    build_time.1,
                    build_trigger * 100.0
                ));
                ui.end_row();

                if ui.button("Reset Timer").clicked() {
                    rd.frame_time_ma = IncrementalMA::new(rd.time_ma_window);
                    rd.sort_time_ma = IncrementalMA::new(rd.time_ma_window);
                    rd.build_time_ma = IncrementalMA::new(rd.time_ma_window);
                    rd.sort_trigger_ma = IncrementalMA::new(rd.time_ma_window);
                    rd.build_trigger_ma = IncrementalMA::new(rd.time_ma_window);
                }
                ui.end_row();

                let mut splat_count: usize = 0;
                let mut blending_splat_count: usize = 0;
                if rd.cur_scene_data.is_some() {
                    splat_count = rd.cur_scene_data.as_ref().unwrap().splat_count;
                    blending_splat_count = rd.cur_scene_data.as_ref().unwrap().blending_splat_count;
                }
                ui.add(egui::Label::new("Splat Count (With Blending)"));
                ui.label(format!(
                    "{} ({})",
                    splat_count.to_formatted_string(&Locale::en),
                    blending_splat_count.to_formatted_string(&Locale::en)
                ));
                ui.end_row();
            });

        ui.collapsing("LOD", |ui| {
            egui::Grid::new("lod_grid")
                .num_columns(2)
                .spacing([40.0, 4.0])
                .striped(true)
                .show(ui, |ui| {
                    for i in 0..rd.max_lod_count {
                        ui.label(format!("LOD {i}"));
                        ui.end_row();

                        ui.label("Enable");
                        ui.checkbox(&mut rd.render_config.lod_enable[i], "");
                        ui.end_row();

                        ui.label("Splat Count");
                        let splat_count = if rd.cur_scene_data.is_some() {
                            rd.cur_scene_data.as_ref().unwrap().lod_splat_count[i]
                        } else {
                            0
                        };
                        ui.label(splat_count.to_formatted_string(&Locale::en));
                        ui.end_row();

                        ui.label("Tile Instance");
                        let instance_count = if rd.cur_scene_data.is_some() {
                            rd.cur_scene_data.as_ref().unwrap().lod_instance_count[i]
                        } else {
                            0
                        };
                        ui.label(instance_count.to_formatted_string(&Locale::en));
                        ui.end_row();
                    }
                });
        });
    }

    fn draw_basis_motion_ui(&self, ui: &mut egui::Ui, rd: &mut RenderData) {
        if rd.basis_bank_preview.is_none() {
            return;
        }

        egui::CollapsingHeader::new("Basis Motion")
            .id_salt("basis_motion_default_open_v2")
            .default_open(true)
            .show(ui, |ui| {
                let Some(motion) = rd.basis_bank_preview.clone() else {
                    ui.label("Basis preview data is unavailable.");
                    return;
                };
                if motion.basis_count == 0 {
                    ui.label("Basis bank is empty.");
                    return;
                }

                let max_basis = motion.basis_count.saturating_sub(1) as u32;
                rd.basis_preview_selected_id = rd.basis_preview_selected_id.min(max_basis);
                let basis_id = rd.basis_preview_selected_id as usize;
                let mut shared_basis = basis_id;
                let mut selection_changed = false;
                ui.horizontal(|ui| {
                    ui.label("Shared basis");
                    selection_changed |= ui
                        .add(
                            egui::DragValue::new(&mut shared_basis)
                                .speed(1)
                                .range(0..=max_basis as usize),
                        )
                        .changed();
                });
                if selection_changed {
                    rd.basis_preview_selected_id = shared_basis.min(max_basis as usize) as u32;
                }

                let basis_id = rd.basis_preview_selected_id as usize;
                ui.label(artist_basis_label(basis_id));
                if let Some(stats) = motion.usage_stats.get(basis_id) {
                    let affected_prefix = "Affected splats across included LoDs";
                    ui.label(format!(
                        "{}: {}, max |weight|: {:.6}, mean |weight|: {:.6}",
                        affected_prefix,
                        stats.affected_splats.to_formatted_string(&Locale::en),
                        stats.max_abs_weight,
                        stats.mean_abs_weight
                    ));
                }
                ui.label(format!(
                    "Knots: {} source + {} closure = {} exported",
                    motion.meta.source_knot_count,
                    motion.meta.loop_closure_knots,
                    motion.meta.exported_knot_count
                ));
                if let Some(exported_top_k) = rd.basis_bank_top_k.filter(|top_k| *top_k > 0) {
                    let mut active_top_k = rd
                        .basis_bank_active_top_k
                        .unwrap_or(exported_top_k)
                        .clamp(1, exported_top_k);
                    if rd.basis_bank_active_top_k != Some(active_top_k) {
                        rd.basis_bank_active_top_k = Some(active_top_k);
                    }
                    ui.add(
                        egui::Slider::new(&mut active_top_k, 1..=exported_top_k)
                            .text("Basis used per splat"),
                    );
                    ui.label(format!(
                        "Using {} / {} basis coefficients",
                        active_top_k, exported_top_k
                    ));
                    if rd.basis_bank_active_top_k != Some(active_top_k) {
                        rd.basis_bank_active_top_k = Some(active_top_k);
                        rd.mark_motion_debug_dirty();
                    }
                }

                ui.separator();
                let mut selected_edit = rd
                    .basis_edit_overrides
                    .get(basis_id)
                    .copied()
                    .unwrap_or_default();
                let mut edit_changed = false;
                let enabled_response =
                    ui.checkbox(&mut selected_edit.enabled, "Edit selected basis");
                edit_changed |= enabled_response.changed();
                if !selected_edit.enabled {
                    rd.basis_knot_edit_dragging_knot = None;
                }
                ui.add_enabled_ui(selected_edit.enabled, |ui| {
                    edit_changed |= ui
                        .add(
                            egui::Slider::new(&mut selected_edit.amplitude_scale, 0.0..=3.0)
                                .text("Strength"),
                        )
                        .changed();
                    edit_changed |= ui
                        .add(
                            egui::Slider::new(&mut selected_edit.phase_offset, -1.0..=1.0)
                                .text("Timing offset"),
                        )
                        .changed();
                    edit_changed |= ui
                        .add(
                            egui::Slider::new(&mut selected_edit.time_scale, 0.0..=4.0)
                                .text("Speed"),
                        )
                        .changed();
                });
                ui.horizontal(|ui| {
                    if ui.button("Reset selected").clicked() {
                        let (reset_edit, scalar_changed, knot_changed) =
                            reset_selected_basis_authoring_edits(
                                &mut rd.basis_edit_overrides,
                                rd.basis_knot_edits.as_mut(),
                                basis_id,
                            );
                        selected_edit = reset_edit;
                        edit_changed |= scalar_changed;
                        if knot_changed {
                            rd.basis_knot_edit_dragging_knot = None;
                            rd.mark_basis_knot_edit_dirty();
                            rd.request_basis_graph_authoring_refresh();
                            rd.mark_motion_debug_dirty();
                        }
                    }
                    if ui.button("Reset all").clicked() {
                        let (reset_edit, scalar_changed, knot_changed) =
                            reset_all_basis_authoring_edits(
                                &mut rd.basis_edit_overrides,
                                rd.basis_knot_edits.as_mut(),
                                basis_id,
                            );
                        selected_edit = reset_edit;
                        edit_changed |= scalar_changed;
                        if knot_changed {
                            rd.basis_knot_edit_dragging_knot = None;
                            rd.mark_basis_knot_edit_dirty();
                            rd.request_basis_graph_authoring_clear();
                            rd.mark_motion_debug_dirty();
                        }
                    }
                });
                if edit_changed {
                    if let Some(edit) = rd.basis_edit_overrides.get_mut(basis_id) {
                        *edit = selected_edit;
                    }
                    rd.mark_basis_edit_dirty();
                }

                if self.draw_basis_batch_scalar_edit_ui(ui, rd, &motion, basis_id) {
                    selected_edit = rd
                        .basis_edit_overrides
                        .get(basis_id)
                        .copied()
                        .unwrap_or_default();
                }

                ui.separator();
                rd.basis_preview_heatmap_enabled =
                    rd.render_config.draw_mode == DrawMode::BasisWeight;
                ui.horizontal(|ui| {
                    let heatmap_response = ui.checkbox(
                        &mut rd.basis_preview_heatmap_enabled,
                        "Show scene heatmap for selected basis",
                    );
                    if heatmap_response.changed() {
                        if rd.basis_preview_heatmap_enabled {
                            rd.render_config.draw_mode = DrawMode::BasisWeight;
                        } else if rd.render_config.draw_mode == DrawMode::BasisWeight {
                            rd.render_config.draw_mode = DrawMode::Normal;
                        }
                    }
                    ui.add(
                        egui::Slider::new(&mut rd.basis_preview_heatmap_normalization, 0.001..=1.0)
                            .logarithmic(true)
                            .text("Heatmap normalization"),
                    );
                });

                ui.separator();
                ui.horizontal_wrapped(|ui| {
                    ui.checkbox(&mut rd.basis_preview_enabled, "Show selected basis spline");
                    ui.separator();
                    ui.label("Projection");
                    for projection in BasisPreviewProjection::ALL {
                        ui.radio_value(
                            &mut rd.basis_preview_projection,
                            projection,
                            projection.as_str(),
                        );
                    }
                });

                if rd.basis_preview_enabled {
                    let current_edit = rd
                        .basis_edit_overrides
                        .get(basis_id)
                        .copied()
                        .unwrap_or_default();
                    self.draw_basis_curve(
                        ui,
                        rd,
                        &motion,
                        basis_id,
                        rd.basis_preview_projection,
                        current_edit.enabled.then_some(current_edit),
                        current_edit.enabled,
                    );
                }

                if let Some(editable_knot_count) = rd
                    .basis_knot_edits
                    .as_ref()
                    .map(|edits| edits.editable_knot_count())
                {
                    let editable_max = editable_knot_count.saturating_sub(1) as u32;
                    rd.basis_knot_edit_selected_knot =
                        rd.basis_knot_edit_selected_knot.min(editable_max);
                    ui.horizontal(|ui| {
                        ui.label("Selected knot");
                        ui.add(
                            egui::DragValue::new(&mut rd.basis_knot_edit_selected_knot)
                                .speed(1)
                                .range(0..=editable_max),
                        );
                        ui.label(format!(
                            "{}",
                            basis_knot_editor_label(
                                rd.basis_knot_edit_selected_knot as usize,
                                motion.meta.source_knot_count,
                                motion.meta.exported_knot_count,
                            )
                        ));
                    });

                    let selected_knot = rd.basis_knot_edit_selected_knot as usize;
                    if let Some(mut point) = rd
                        .basis_knot_edits
                        .as_ref()
                        .and_then(|edits| edits.knot(basis_id, selected_knot))
                    {
                        let mut knot_changed = false;
                        ui.add_enabled_ui(selected_edit.enabled, |ui| {
                            ui.horizontal(|ui| {
                                ui.label("Position");
                                knot_changed |= ui
                                    .add(
                                        egui::DragValue::new(&mut point[0])
                                            .speed(0.001)
                                            .prefix("X "),
                                    )
                                    .changed();
                                knot_changed |= ui
                                    .add(
                                        egui::DragValue::new(&mut point[1])
                                            .speed(0.001)
                                            .prefix("Y "),
                                    )
                                    .changed();
                                knot_changed |= ui
                                    .add(
                                        egui::DragValue::new(&mut point[2])
                                            .speed(0.001)
                                            .prefix("Z "),
                                    )
                                    .changed();
                            });
                        });
                        let did_set = knot_changed
                            && rd.basis_knot_edits.as_mut().is_some_and(|edits| {
                                edits.set_knot(basis_id, selected_knot, point)
                            });
                        if did_set {
                            rd.mark_basis_knot_edit_dirty();
                            rd.request_basis_graph_authoring_refresh();
                            rd.mark_motion_debug_dirty();
                        }
                    }

                    let mut reset_selected_knot = false;
                    let mut reset_basis_knots = false;
                    let mut reset_all_knots = false;
                    ui.horizontal(|ui| {
                        reset_selected_knot = ui.button("Reset knot").clicked();
                        reset_basis_knots = ui.button("Reset basis knots").clicked();
                        reset_all_knots = ui.button("Reset all knots").clicked();
                    });
                    let reset_changed = if reset_selected_knot {
                        rd.basis_knot_edits.as_mut().is_some_and(|edits| {
                            edits.reset_knot(basis_id, rd.basis_knot_edit_selected_knot as usize)
                        })
                    } else if reset_basis_knots {
                        rd.basis_knot_edits
                            .as_mut()
                            .is_some_and(|edits| edits.reset_basis(basis_id))
                    } else if reset_all_knots {
                        if let Some(edits) = rd.basis_knot_edits.as_mut() {
                            edits.reset_all();
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    };
                    if reset_changed {
                        rd.mark_basis_knot_edit_dirty();
                        if reset_all_knots {
                            rd.request_basis_graph_authoring_clear();
                        } else {
                            rd.request_basis_graph_authoring_refresh();
                        }
                        rd.mark_motion_debug_dirty();
                    }
                }
                ui.separator();
                self.draw_basis_3d_view(ui, rd, &motion, basis_id);

                if let Some(graph) = motion
                    .motion_graph
                    .as_ref()
                    .filter(|graph| graph.knot_count > 0)
                {
                    ui.separator();
                    self.draw_basis_motion_graph_controls(ui, rd, &motion, graph, basis_id);
                }
            });
    }

    fn draw_basis_batch_scalar_edit_ui(
        &self,
        ui: &mut egui::Ui,
        rd: &mut RenderData,
        motion: &BasisBankMotionSet,
        basis_id: usize,
    ) -> bool {
        if rd.basis_batch_selection_text.trim().is_empty() {
            rd.basis_batch_selection_text = basis_id.to_string();
        }

        let mut changed = false;
        egui::CollapsingHeader::new("Batch scalar edit")
            .id_salt("basis_batch_scalar_edit_v1")
            .default_open(false)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Shared bases");
                    ui.text_edit_singleline(&mut rd.basis_batch_selection_text);
                    if ui.button("All bases").clicked() {
                        rd.basis_batch_selection_text = "all".to_string();
                    }
                    ui.label("e.g. 0,3,8-12");
                });

                ui.add(
                    egui::Slider::new(&mut rd.basis_batch_edit.amplitude_scale, 0.0..=3.0)
                        .text("Strength"),
                );
                ui.add(
                    egui::Slider::new(&mut rd.basis_batch_edit.phase_offset, -1.0..=1.0)
                        .text("Timing offset"),
                );
                ui.add(
                    egui::Slider::new(&mut rd.basis_batch_edit.time_scale, 0.0..=4.0).text("Speed"),
                );

                let parsed =
                    basis_batch_ids_for_motion(motion, rd.basis_batch_selection_text.as_str());
                match parsed.as_ref() {
                    Ok(ids) => {
                        ui.label(format!("{} shared bases selected", ids.len()));
                    }
                    Err(error) => {
                        ui.colored_label(egui::Color32::from_rgb(245, 130, 100), error);
                    }
                }
                ui.horizontal(|ui| {
                    let valid_ids = parsed.as_ref().ok();
                    let apply_clicked = ui
                        .add_enabled(valid_ids.is_some(), egui::Button::new("Apply to selection"))
                        .clicked();
                    if apply_clicked {
                        if let Some(ids) = valid_ids {
                            changed |= apply_basis_batch_scalar_edit(
                                &mut rd.basis_edit_overrides,
                                ids,
                                rd.basis_batch_edit,
                            );
                        }
                    }
                    let reset_clicked = ui
                        .add_enabled(valid_ids.is_some(), egui::Button::new("Reset selection"))
                        .clicked();
                    if reset_clicked {
                        if let Some(ids) = valid_ids {
                            changed |=
                                reset_basis_batch_scalar_edits(&mut rd.basis_edit_overrides, ids);
                        }
                    }
                });
            });
        if changed {
            rd.mark_basis_edit_dirty();
        }
        changed
    }
    fn draw_basis_motion_graph_controls(
        &self,
        ui: &mut egui::Ui,
        rd: &mut RenderData,
        motion: &BasisBankMotionSet,
        graph: &crate::basis_motion_graph::BasisMotionGraph,
        basis_id: usize,
    ) {
        let max_segment = graph.knot_count.saturating_sub(1) as u32;
        rd.basis_graph_selected_segment = rd.basis_graph_selected_segment.min(max_segment);
        ui.add(
            egui::Slider::new(&mut rd.basis_graph_selected_segment, 0..=max_segment)
                .text("Inspect loop segment"),
        );
        let segment = rd.basis_graph_selected_segment as usize;

        let mut playback_config = rd.basis_graph_playback_config;
        let mut region_config = rd.basis_graph_region_config;
        let mut region_changed = false;
        let mut playback_changed = false;
        let mut playback_reset = false;
        ui.horizontal_wrapped(|ui| {
            ui.checkbox(
                &mut rd.basis_graph_authoring_auto_refresh,
                "Update branch suggestions after edits",
            );
            if ui.button("Refresh branches").clicked() {
                rd.request_basis_graph_authoring_refresh();
            }
            let summary = rd.basis_graph_authoring_summary.as_ref();
            let status = if rd.basis_graph_authoring_stale {
                "stale"
            } else if summary.is_some() {
                "fresh"
            } else {
                "baseline"
            };
            let refreshed_nodes = summary
                .map(|summary| summary.refreshed_node_count)
                .unwrap_or(0);
            let changed_targets = summary
                .map(|summary| summary.changed_best_target_count)
                .unwrap_or(0);
            ui.label(format!(
                "Local branches: {}, refreshed nodes {}, changed best targets {}",
                status, refreshed_nodes, changed_targets
            ));
        });
        ui.horizontal(|ui| {
            let enabled_changed = ui
                .checkbox(&mut playback_config.enabled, "Enable graph playback")
                .changed();
            playback_changed |= enabled_changed;
            if enabled_changed {
                playback_reset = playback_config.enabled;
            }
            ui.separator();
            ui.label("Mode");
            for policy in BasisGraphPlaybackPolicy::ALL {
                let response = ui.radio_value(
                    &mut playback_config.policy,
                    policy,
                    basis_graph_policy_ui_label(policy),
                );
                if response.changed() {
                    playback_changed = true;
                    playback_reset = true;
                }
            }
            if ui.button("Reset").clicked() {
                playback_reset = true;
            }
        });

        ui.horizontal_wrapped(|ui| {
            ui.label("Branch domain");
            for domain in BasisGraphBranchDomain::ALL {
                if ui
                    .radio_value(&mut region_config.domain, domain, domain.as_str())
                    .changed()
                {
                    region_changed = true;
                    playback_reset = true;
                }
            }
        });
        if region_config.domain == BasisGraphBranchDomain::FbmRegions {
            ui.horizontal_wrapped(|ui| {
                let mut region_count = region_config.region_count;
                if ui
                    .add(
                        egui::DragValue::new(&mut region_count)
                            .range(1..=BasisGraphRegionConfig::MAX_REGION_COUNT)
                            .prefix("Regions "),
                    )
                    .changed()
                {
                    region_config.region_count = region_count;
                    region_changed = true;
                    playback_reset = true;
                }
                if ui
                    .add(
                        egui::DragValue::new(&mut region_config.region_size_world)
                            .speed(0.1)
                            .range(0.1..=f32::MAX)
                            .prefix("Size "),
                    )
                    .changed()
                {
                    region_changed = true;
                    playback_reset = true;
                }
                if ui
                    .add(
                        egui::DragValue::new(&mut region_config.octaves)
                            .range(1..=4)
                            .prefix("Octaves "),
                    )
                    .changed()
                {
                    region_changed = true;
                    playback_reset = true;
                }
                if ui
                    .add(
                        egui::DragValue::new(&mut region_config.warp_strength)
                            .speed(0.01)
                            .range(0.0..=4.0)
                            .prefix("Warp "),
                    )
                    .changed()
                {
                    region_changed = true;
                    playback_reset = true;
                }
                if ui
                    .add(egui::DragValue::new(&mut region_config.seed).prefix("Seed "))
                    .changed()
                {
                    region_changed = true;
                    playback_reset = true;
                }
            });
            let max_region = region_config.effective_region_count().saturating_sub(1);
            rd.basis_graph_selected_region = rd.basis_graph_selected_region.min(max_region);
            ui.add(
                egui::Slider::new(&mut rd.basis_graph_selected_region, 0..=max_region)
                    .text("Inspect region"),
            );
        } else {
            rd.basis_graph_selected_region = 0;
        }
        ui.add_enabled_ui(playback_config.enabled, |ui| {
            if playback_config.policy == BasisGraphPlaybackPolicy::Stochastic {
                playback_changed |= ui
                    .add(
                        egui::Slider::new(&mut playback_config.branch_probability, 0.0..=1.0)
                            .text("Branch chance"),
                    )
                    .changed();
                playback_changed |= ui
                    .add(
                        egui::Slider::new(&mut playback_config.temperature, 0.01..=2.0)
                            .logarithmic(true)
                            .text("Variety"),
                    )
                    .changed();
                let seed_response =
                    ui.add(egui::DragValue::new(&mut playback_config.seed).prefix("Seed "));
                if seed_response.changed() {
                    playback_changed = true;
                    playback_reset = true;
                }
            }
            playback_changed |= ui
                .add(
                    egui::Slider::new(&mut playback_config.blend_duration, 0.0..=1.0)
                        .text("Smoothness"),
                )
                .changed();
            let min_interval_response = ui.add(
                egui::DragValue::new(&mut playback_config.min_branch_interval_segments)
                    .range(0..=32)
                    .prefix("Min travel "),
            );
            if min_interval_response.changed() {
                playback_changed = true;
                playback_reset = true;
            }

            let detected_quality_preset = basis_graph_quality_preset(playback_config);
            if detected_quality_preset == BasisGraphQualityPreset::Custom {
                rd.basis_graph_quality_custom_mode = true;
            }
            let mut quality_preset = if rd.basis_graph_quality_custom_mode {
                BasisGraphQualityPreset::Custom
            } else {
                detected_quality_preset
            };
            ui.horizontal(|ui| {
                ui.label("Quality");
                for preset in BasisGraphQualityPreset::ALL {
                    if ui
                        .radio_value(
                            &mut quality_preset,
                            preset,
                            basis_graph_quality_preset_label(preset),
                        )
                        .changed()
                    {
                        if preset == BasisGraphQualityPreset::Custom {
                            rd.basis_graph_quality_custom_mode = true;
                        } else {
                            rd.basis_graph_quality_custom_mode = false;
                            apply_basis_graph_quality_preset(&mut playback_config, preset);
                            playback_changed = true;
                            playback_reset = true;
                        }
                    }
                }
            });

            if rd.basis_graph_quality_custom_mode {
                ui.collapsing("Custom quality", |ui| {
                    ui.horizontal(|ui| {
                        playback_changed |= ui
                            .checkbox(&mut playback_config.max_position_cost_enabled, "Position")
                            .changed();
                        ui.add_enabled_ui(playback_config.max_position_cost_enabled, |ui| {
                            playback_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut playback_config.max_position_cost)
                                        .speed(0.01)
                                        .range(0.0..=f32::MAX),
                                )
                                .changed();
                        });
                    });
                    ui.horizontal(|ui| {
                        playback_changed |= ui
                            .checkbox(&mut playback_config.max_velocity_cost_enabled, "Velocity")
                            .changed();
                        ui.add_enabled_ui(playback_config.max_velocity_cost_enabled, |ui| {
                            playback_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut playback_config.max_velocity_cost)
                                        .speed(0.01)
                                        .range(0.0..=f32::MAX),
                                )
                                .changed();
                        });
                    });
                    ui.horizontal(|ui| {
                        playback_changed |= ui
                            .checkbox(
                                &mut playback_config.max_acceleration_cost_enabled,
                                "Acceleration",
                            )
                            .changed();
                        ui.add_enabled_ui(playback_config.max_acceleration_cost_enabled, |ui| {
                            playback_changed |= ui
                                .add(
                                    egui::DragValue::new(
                                        &mut playback_config.max_acceleration_cost,
                                    )
                                    .speed(0.01)
                                    .range(0.0..=f32::MAX),
                                )
                                .changed();
                        });
                    });
                });
            }
        });
        if region_changed {
            region_config = region_config.sanitized();
            rd.basis_graph_region_config = region_config;
            rd.basis_graph_selected_region = rd
                .basis_graph_selected_region
                .min(region_config.effective_region_count().saturating_sub(1));
            playback_changed = true;
        }
        if playback_changed
            && (playback_config.max_position_cost_enabled
                != rd.basis_graph_playback_config.max_position_cost_enabled
                || playback_config.max_position_cost
                    != rd.basis_graph_playback_config.max_position_cost
                || playback_config.max_velocity_cost_enabled
                    != rd.basis_graph_playback_config.max_velocity_cost_enabled
                || playback_config.max_velocity_cost
                    != rd.basis_graph_playback_config.max_velocity_cost
                || playback_config.max_acceleration_cost_enabled
                    != rd.basis_graph_playback_config.max_acceleration_cost_enabled
                || playback_config.max_acceleration_cost
                    != rd.basis_graph_playback_config.max_acceleration_cost)
        {
            playback_reset = true;
        }
        if playback_changed {
            rd.basis_graph_playback_config = playback_config;
            rd.mark_motion_debug_dirty();
        }
        if playback_reset {
            rd.request_basis_graph_playback_reset();
        }

        let baseline_branches = graph.branches_for(basis_id, segment);
        let branches: Vec<crate::basis_motion_graph::BasisMotionGraphBranch> = rd
            .basis_graph_authoring_selected_branches
            .as_ref()
            .cloned()
            .unwrap_or_else(|| baseline_branches.into_iter().cloned().collect());
        let usable_branch_count = branches
            .iter()
            .filter(|branch| !branch_rejection(branch, rd.basis_graph_playback_config).rejected())
            .count();

        if rd.basis_graph_playback_config.enabled {
            if let Some(state) = rd.basis_graph_playback_selected_state.as_ref() {
                let min_interval = rd.basis_graph_playback_config.min_branch_interval_segments;
                let playback_label = if rd.basis_graph_region_config.domain
                    == BasisGraphBranchDomain::FbmRegions
                {
                    format!(
                        "Region {} / {} -> {}",
                        state.region_id,
                        artist_basis_label_for_global_in_motion(motion, state.original_basis_id),
                        artist_basis_label_for_global_in_motion(motion, state.active_basis_id)
                    )
                } else {
                    format!(
                        "{} -> {}",
                        artist_basis_label_for_global_in_motion(motion, state.original_basis_id),
                        artist_basis_label_for_global_in_motion(motion, state.active_basis_id)
                    )
                };
                ui.label(playback_label);
                ui.label(format!(
                    "{} / segment {} / phase {:.2}",
                    artist_basis_label(state.active_basis_id),
                    state.segment,
                    state.segment_phase
                ));
                ui.horizontal_wrapped(|ui| {
                    if state.transition_active {
                        ui.label(format!(
                            "Transition {:.1} / {:.1}",
                            state.transition_phase_segments, state.transition_duration_segments
                        ));
                    } else if state.blend_active {
                        ui.label(format!("Transition blend {:.2}", state.blend_weight));
                    } else {
                        ui.label("Transition idle");
                    }
                    ui.separator();
                    ui.label(format!(
                        "Cooldown {} / {}",
                        state.segments_since_branch, min_interval
                    ));
                    ui.separator();
                    ui.label(format!(
                        "Usable branches {} / {}",
                        usable_branch_count,
                        branches.len()
                    ));
                });
            } else {
                ui.label("Waiting for playback state.");
            }
        }

        if !rd.basis_graph_playback_config.enabled
            || rd.basis_graph_playback_selected_state.is_none()
        {
            ui.label(format!(
                "Usable branches {} / {}",
                usable_branch_count,
                branches.len()
            ));
        }
        if branches.is_empty() {
            ui.label("No branch edges for this selection.");
        }

        let has_usable_target = branches
            .iter()
            .any(|branch| !branch_rejection(branch, rd.basis_graph_playback_config).rejected());
        ui.add_enabled(
            has_usable_target,
            egui::Checkbox::new(
                &mut rd.basis_best_target_preview_enabled,
                "Show best target preview",
            ),
        );

        ui.collapsing("Advanced Diagnostics", |ui| {
                ui.collapsing(
                    basis_graph_diagnostics_section_label(BasisGraphDiagnosticsSection::Asset),
                    |ui| {
                        egui::Grid::new("basis_graph_diagnostics_asset")
                            .num_columns(2)
                            .striped(true)
                            .show(ui, |ui| {
                                ui.label("Graph version");
                                ui.label(graph.format_version.to_string());
                                ui.end_row();
                                ui.label("Scope");
                                ui.label(graph.basis_scope.as_str());
                                ui.end_row();
                                ui.label("Source LoD");
                                ui.label(
                                    graph
                                        .basis_source_lod
                                        .map(|lod| lod.to_string())
                                        .unwrap_or_else(|| "n/a".to_string()),
                                );
                                ui.end_row();
                                ui.label("Included LoDs");
                                ui.label(format!("{:?}", graph.include_lods));
                                ui.end_row();
                                ui.label("LODs");
                                ui.label(graph.lods.len().to_string());
                                ui.end_row();
                                ui.label("Local bases");
                                ui.label(graph.basis_count.to_string());
                                ui.end_row();
                                ui.label("Knots");
                                ui.label(graph.knot_count.to_string());
                                ui.end_row();
                                ui.label("Top branches");
                                ui.label(graph.branch_top_k.to_string());
                                ui.end_row();
                                ui.label("Score weights");
                                ui.label(format!(
                                    "pos {:.2}, vel {:.2}, accel {:.2}, usage {:.2}",
                                    graph.score_weights.position,
                                    graph.score_weights.velocity,
                                    graph.score_weights.acceleration,
                                    graph.score_weights.usage
                                ));
                                ui.end_row();
                            });
                    },
                );

                ui.collapsing(
                    basis_graph_diagnostics_section_label(BasisGraphDiagnosticsSection::Selection),
                    |ui| {
                        egui::Grid::new("basis_graph_diagnostics_selection")
                            .num_columns(2)
                            .striped(true)
                            .show(ui, |ui| {
                                ui.label("Selected basis");
                                ui.label(artist_basis_label(basis_id));
                                ui.end_row();
                                ui.label("Selected segment in loop");
                                ui.label(segment.to_string());
                                ui.end_row();
                                ui.label("Default successor");
                                ui.label(artist_branch_target_label(
                                    basis_id,
                                    (segment + 1) % graph.knot_count,
                                ));
                                ui.end_row();
                                ui.label("Branch source");
                                ui.label(if rd.basis_graph_authoring_selected_node_refreshed {
                                    "Local refreshed graph"
                                } else {
                                    "Baseline graph"
                                });
                                ui.end_row();
                            });
                    },
                );

                ui.collapsing(
                    basis_graph_diagnostics_section_label(BasisGraphDiagnosticsSection::Playback),
                    |ui| {
                        if rd.basis_graph_playback_config.enabled {
                            if let Some(state) = rd.basis_graph_playback_selected_state.as_ref() {
                                let min_interval =
                                    rd.basis_graph_playback_config.min_branch_interval_segments;
                                egui::Grid::new("basis_graph_diagnostics_playback")
                                    .num_columns(2)
                                    .striped(true)
                                    .show(ui, |ui| {
                                        ui.label("Branch domain");
                                        ui.label(rd.basis_graph_region_config.domain.as_str());
                                        ui.end_row();
                                        ui.label("Selected region");
                                        ui.label(format!(
                                            "{} / {}",
                                            state.region_id,
                                            rd.basis_graph_region_config
                                                .effective_region_count()
                                                .saturating_sub(1)
                                        ));
                                        ui.end_row();
                                        ui.label("Region field");
                                        ui.label(format!(
                                            "size {:.2}, octaves {}, warp {:.2}, seed {}",
                                            rd.basis_graph_region_config.region_size_world,
                                            rd.basis_graph_region_config.octaves,
                                            rd.basis_graph_region_config.warp_strength,
                                            rd.basis_graph_region_config.seed
                                        ));
                                        ui.end_row();
                                        ui.label("Source -> active");
                                        ui.label(format!(
                                            "{} -> {}",
                                            state.original_basis_id,
                                            state.active_basis_id
                                        ));
                                        ui.end_row();
                                        ui.label("Active node");
                                        ui.label(format!(
                                            "{} / segment {} / phase {:.3}",
                                            artist_basis_label(state.active_basis_id),
                                            state.segment,
                                            state.segment_phase
                                        ));
                                        ui.end_row();
                                        ui.label("Last edge");
                                        ui.label(basis_graph_last_edge_label(state.last_edge));
                                        ui.end_row();
                                        ui.label("Cooldown");
                                        ui.label(format!(
                                            "{} / {}{}",
                                            state.segments_since_branch,
                                            min_interval,
                                            if state.segments_since_branch < min_interval {
                                                " active"
                                            } else {
                                                ""
                                            }
                                        ));
                                        ui.end_row();
                                        ui.label("V1 blend");
                                        ui.label(format!(
                                            "{} | from global {} / segment {} / phase {:.3} / weight {:.3}",
                                            if state.blend_active { "active" } else { "inactive" },
                                            state.blend_from_basis_id,
                                            state.blend_from_segment,
                                            state.blend_phase,
                                            state.blend_weight
                                        ));
                                        ui.end_row();
                                        ui.label("V2 transition");
                                        if state.transition_active {
                                            ui.label(format!(
                                                "active | target global {} / segment {} / phase {:.3} / {:.3}",
                                                state.transition_target_basis_id,
                                                state.transition_target_segment,
                                                state.transition_phase_segments,
                                                state.transition_duration_segments
                                            ));
                                        } else {
                                            ui.label("inactive");
                                        }
                                        ui.end_row();
                                    });
                            } else {
                                ui.label("Playback state will appear after the next motion update.");
                            }
                        } else {
                            ui.label("Graph playback is disabled.");
                        }
                    },
                );

                ui.collapsing(
                    basis_graph_diagnostics_section_label(BasisGraphDiagnosticsSection::Filtering),
                    |ui| {
                        ui.horizontal(|ui| {
                            let mut playback_config = rd.basis_graph_playback_config;
                            let mut score_gate_changed = ui
                                .checkbox(
                                    &mut playback_config.max_branch_score_enabled,
                                    "Limit branch score",
                                )
                                .changed();
                            ui.add_enabled_ui(playback_config.max_branch_score_enabled, |ui| {
                                score_gate_changed |= ui
                                    .add(
                                        egui::DragValue::new(
                                            &mut playback_config.max_branch_score,
                                        )
                                        .speed(0.01)
                                        .range(0.0..=f32::MAX)
                                        .prefix("Max "),
                                    )
                                    .changed();
                            });
                            if score_gate_changed {
                                rd.basis_graph_playback_config = playback_config;
                                rd.mark_motion_debug_dirty();
                                rd.request_basis_graph_playback_reset();
                            }
                        });
                        if let Some(state) = rd.basis_graph_playback_selected_state.as_ref() {
                            egui::Grid::new("basis_graph_diagnostics_filtering")
                                .num_columns(2)
                                .striped(true)
                                .show(ui, |ui| {
                                    ui.label("Rejected total");
                                    ui.label(state.rejected_branch_count.to_string());
                                    ui.end_row();
                                    ui.label("Rejected by score");
                                    ui.label(state.rejected_branch_score_count.to_string());
                                    ui.end_row();
                                    ui.label("Rejected by position");
                                    ui.label(state.rejected_branch_position_count.to_string());
                                    ui.end_row();
                                    ui.label("Rejected by velocity");
                                    ui.label(state.rejected_branch_velocity_count.to_string());
                                    ui.end_row();
                                    ui.label("Rejected by acceleration");
                                    ui.label(state.rejected_branch_acceleration_count.to_string());
                                    ui.end_row();
                                });
                        }
                    },
                );

                ui.collapsing(
                    basis_graph_diagnostics_section_label(BasisGraphDiagnosticsSection::Branches),
                    |ui| {
                        if branches.is_empty() {
                            ui.label("No branch edges for this selection.");
                            return;
                        }
                        ui.label(
                            "Rank is constructor preference order; lower score is better. Quality shows the first active gate that rejected a branch, or Good.",
                        );
                        egui::Grid::new("basis_graph_branch_diagnostics_table")
                            .striped(true)
                            .num_columns(5)
                            .show(ui, |ui| {
                                ui.strong("Rank");
                                ui.strong("Target");
                                ui.strong("Score");
                                ui.strong("Quality");
                                ui.strong("Status");
                                ui.end_row();
                                for branch in &branches {
                                    let rejection =
                                        branch_rejection(branch, rd.basis_graph_playback_config);
                                    ui.label(format!("#{:02}", branch.rank));
                                    ui.label(artist_branch_target_label(
                                        branch.to_basis,
                                        branch.to_segment,
                                    ));
                                    ui.label(format!("{:.3}", branch.score));
                                    ui.label(branch_rejection_quality_label(rejection));
                                    ui.label(branch_rejection_status_label(rejection));
                                    ui.end_row();
                                }
                            });
                        ui.separator();
                        for branch in &branches {
                            let target_global = String::new();
                            let rejection =
                                branch_rejection(branch, rd.basis_graph_playback_config);
                            ui.collapsing(
                                format!(
                                    "#{:02} -> {}{} | score {:.6}{}",
                                    branch.rank,
                                    artist_branch_target_label(
                                        branch.to_basis,
                                        branch.to_segment,
                                    ),
                                    target_global,
                                    branch.score,
                                    branch_rejection_label(rejection)
                                ),
                                |ui| {
                                    egui::Grid::new(format!(
                                        "basis_graph_branch_detail_{}",
                                        branch.rank
                                    ))
                                    .num_columns(2)
                                    .striped(true)
                                    .show(ui, |ui| {
                                        ui.label("Position cost");
                                        ui.label(format!("{:.6}", branch.position_cost));
                                        ui.end_row();
                                        ui.label("Velocity cost");
                                        ui.label(format!("{:.6}", branch.velocity_cost));
                                        ui.end_row();
                                        ui.label("Acceleration cost");
                                        ui.label(format!("{:.6}", branch.acceleration_cost));
                                        ui.end_row();
                                        ui.label("Usage bonus");
                                        ui.label(format!("{:.6}", branch.usage_bonus));
                                        ui.end_row();
                                        if let Some(debug) = basis_branch_continuity_debug(motion, branch) {
                                            ui.label("Endpoints");
                                            ui.label(format!(
                                                "global {} seg {} u=1 -> global {} seg {} u=0",
                                                debug.source_basis_id,
                                                debug.source_segment,
                                                debug.target_basis_id,
                                                debug.target_segment
                                            ));
                                            ui.end_row();
                                            ui.label("Measured position");
                                            ui.label(format!(
                                                "|p| {:.6} {}",
                                                debug.position_norm,
                                                format_vec3(debug.position_delta)
                                            ));
                                            ui.end_row();
                                            ui.label("Measured velocity");
                                            ui.label(format!(
                                                "|v| {:.6} {}",
                                                debug.velocity_norm,
                                                format_vec3(debug.velocity_delta)
                                            ));
                                            ui.end_row();
                                            ui.label("Measured acceleration");
                                            ui.label(format!(
                                                "|a| {:.6} {}",
                                                debug.acceleration_norm,
                                                format_vec3(debug.acceleration_delta)
                                            ));
                                            ui.end_row();
                                        } else {
                                            ui.label("Continuity");
                                            ui.label("unavailable");
                                            ui.end_row();
                                        }
                                        ui.label("Transition");
                                        if let Some(transition) = branch.transition.as_ref() {
                                            ui.label(format!(
                                                "{} | duration {} | knots {}",
                                                transition.kind,
                                                transition.duration_segments,
                                                transition.knots.len()
                                            ));
                                            ui.end_row();
                                            ui.label("Tangents");
                                            ui.label(format!(
                                                "start {} | end {}",
                                                format_vec3(transition.start_tangent),
                                                format_vec3(transition.end_tangent)
                                            ));
                                        } else {
                                            ui.label("V1 branch-only fallback");
                                        }
                                        ui.end_row();
                                    });
                                },
                            );
                        }
                    },
                );
            });

        let top_target_basis = rd
            .basis_best_target_preview_enabled
            .then(|| {
                branches.iter().find_map(|branch| {
                    (!branch_rejection(branch, rd.basis_graph_playback_config).rejected())
                        .then_some(branch.to_basis)
                })
            })
            .flatten();
        if let Some(target_global) = top_target_basis {
            ui.separator();
            ui.label("Best Target Preview");
            self.draw_basis_curve(
                ui,
                rd,
                motion,
                target_global,
                rd.basis_preview_projection,
                None,
                false,
            );
        }
    }

    fn draw_basis_curve(
        &self,
        ui: &mut egui::Ui,
        rd: &mut RenderData,
        motion: &BasisBankMotionSet,
        basis_id: usize,
        projection: BasisPreviewProjection,
        edit: Option<BasisEditOverride>,
        interactive: bool,
    ) {
        let desired_size = egui::vec2(ui.available_width().max(220.0), 220.0);
        let sense = if interactive {
            egui::Sense::click_and_drag()
        } else {
            egui::Sense::hover()
        };
        let (rect, response) = ui.allocate_exact_size(desired_size, sense);
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 8.0, egui::Color32::from_rgb(28, 32, 28));
        painter.rect_stroke(
            rect,
            8.0,
            egui::Stroke::new(1.0, egui::Color32::from_rgb(90, 104, 88)),
            egui::StrokeKind::Inside,
        );

        let knot_count = motion.meta.exported_knot_count;
        let edited_knots_snapshot = rd
            .basis_knot_edits
            .as_ref()
            .map(|edits| edits.edited_knots().to_vec());
        let edited_knots = edited_knots_snapshot
            .as_deref()
            .unwrap_or(motion.basis_knots.as_slice());
        let basis_has_knot_edits = rd
            .basis_knot_edits
            .as_ref()
            .is_some_and(|edits| edits.basis_is_edited(basis_id));
        let show_edited_curve = basis_has_knot_edits || edit.is_some();
        let sample_count = 96_usize.max(knot_count * 4);
        let mut points = Vec::with_capacity(sample_count + knot_count);
        points.push([0.0, 0.0]);
        for i in 0..sample_count {
            let t = i as f32 / sample_count as f32;
            points.push(project_basis_point(
                basis_bank_delta(
                    &motion.basis_knots,
                    motion.basis_count,
                    knot_count,
                    basis_id,
                    t,
                ),
                projection,
            ));
            if show_edited_curve {
                let edited_delta = if let Some(edit) = edit {
                    edited_basis_bank_delta(
                        edited_knots,
                        motion.basis_count,
                        knot_count,
                        basis_id,
                        t,
                        edit,
                    )
                } else {
                    basis_bank_delta(edited_knots, motion.basis_count, knot_count, basis_id, t)
                };
                points.push(project_basis_point(edited_delta, projection));
            }
        }
        let original_knot_points: Vec<[f32; 2]> = (0..knot_count)
            .map(|knot| {
                let base = (basis_id * knot_count + knot) * 3;
                project_basis_point(
                    [
                        motion.basis_knots[base],
                        motion.basis_knots[base + 1],
                        motion.basis_knots[base + 2],
                    ],
                    projection,
                )
            })
            .collect();
        let edited_knot_points: Vec<[f32; 2]> = (0..knot_count)
            .map(|knot| {
                let base = (basis_id * knot_count + knot) * 3;
                project_basis_point(
                    [
                        edited_knots[base],
                        edited_knots[base + 1],
                        edited_knots[base + 2],
                    ],
                    projection,
                )
            })
            .collect();
        points.extend(original_knot_points.iter().copied());
        points.extend(edited_knot_points.iter().copied());

        let (min_xy, max_xy) = basis_plot_bounds(&points);
        let pad = 18.0;
        let plot_width = (rect.width() - 2.0 * pad).max(1.0);
        let plot_height = (rect.height() - 2.0 * pad).max(1.0);
        let world_width = (max_xy[0] - min_xy[0]).abs().max(1e-6);
        let world_height = (max_xy[1] - min_xy[1]).abs().max(1e-6);
        let to_screen = |p: [f32; 2]| -> egui::Pos2 {
            let x = (p[0] - min_xy[0]) / world_width;
            let y = (p[1] - min_xy[1]) / world_height;
            egui::pos2(
                rect.left() + pad + x * plot_width,
                rect.bottom() - pad - y * plot_height,
            )
        };

        let origin = to_screen([0.0, 0.0]);
        painter.line_segment(
            [
                egui::pos2(rect.left() + pad, origin.y),
                egui::pos2(rect.right() - pad, origin.y),
            ],
            egui::Stroke::new(1.0, egui::Color32::from_rgb(88, 96, 92)),
        );
        painter.line_segment(
            [
                egui::pos2(origin.x, rect.top() + pad),
                egui::pos2(origin.x, rect.bottom() - pad),
            ],
            egui::Stroke::new(1.0, egui::Color32::from_rgb(88, 96, 92)),
        );
        painter.circle_filled(origin, 3.0, egui::Color32::from_rgb(230, 230, 220));
        painter.text(
            origin + egui::vec2(5.0, -5.0),
            egui::Align2::LEFT_BOTTOM,
            "0",
            egui::FontId::monospace(10.0),
            egui::Color32::from_rgb(232, 236, 226),
        );

        if interactive {
            if response.drag_started() || response.clicked() {
                if let Some(pointer_pos) = response.interact_pointer_pos() {
                    let nearest = edited_knot_points
                        .iter()
                        .enumerate()
                        .map(|(knot, point)| (knot, to_screen(*point).distance(pointer_pos)))
                        .filter(|(_, distance)| *distance <= 12.0)
                        .min_by(|a, b| a.1.total_cmp(&b.1))
                        .map(|(knot, _)| knot as u32);
                    if let Some(knot) = nearest {
                        rd.basis_knot_edit_selected_knot = knot;
                        if response.drag_started() {
                            rd.basis_knot_edit_dragging_knot = Some(knot);
                        }
                    } else if response.drag_started() {
                        rd.basis_knot_edit_dragging_knot = None;
                    }
                }
            }
            if response.dragged() {
                if let Some(knot) = rd.basis_knot_edit_dragging_knot {
                    let pointer_delta = ui.input(|input| input.pointer.delta());
                    if pointer_delta != egui::Vec2::ZERO {
                        let world_delta = [
                            pointer_delta.x * world_width / plot_width,
                            -pointer_delta.y * world_height / plot_height,
                        ];
                        let did_edit = if let Some(knot_edits) = rd.basis_knot_edits.as_mut() {
                            let knot = knot as usize;
                            if let Some(point) = knot_edits.knot(basis_id, knot) {
                                if let Some(edited_point) = apply_basis_knot_plane_delta(
                                    point,
                                    basis_projection_edit_plane(projection),
                                    world_delta,
                                ) {
                                    knot_edits.set_knot(basis_id, knot, edited_point)
                                } else {
                                    false
                                }
                            } else {
                                false
                            }
                        } else {
                            false
                        };
                        if did_edit {
                            rd.mark_basis_knot_edit_dirty();
                            rd.mark_motion_debug_dirty();
                        }
                    }
                }
            }
            if !ui.input(|input| input.pointer.primary_down()) {
                if rd.basis_knot_edit_dragging_knot.take().is_some() {
                    rd.request_basis_graph_authoring_refresh();
                }
            }
        }

        let curve_points: Vec<egui::Pos2> = (0..sample_count)
            .map(|i| {
                let t = i as f32 / sample_count as f32;
                to_screen(project_basis_point(
                    basis_bank_delta(
                        &motion.basis_knots,
                        motion.basis_count,
                        knot_count,
                        basis_id,
                        t,
                    ),
                    projection,
                ))
            })
            .collect();
        painter.add(egui::Shape::line(
            curve_points,
            egui::Stroke::new(2.5, egui::Color32::from_rgb(255, 142, 85)),
        ));
        if show_edited_curve {
            let edited_curve_points: Vec<egui::Pos2> = (0..sample_count)
                .map(|i| {
                    let t = i as f32 / sample_count as f32;
                    let edited_delta = if let Some(edit) = edit {
                        edited_basis_bank_delta(
                            edited_knots,
                            motion.basis_count,
                            knot_count,
                            basis_id,
                            t,
                            edit,
                        )
                    } else {
                        basis_bank_delta(edited_knots, motion.basis_count, knot_count, basis_id, t)
                    };
                    to_screen(project_basis_point(edited_delta, projection))
                })
                .collect();
            painter.add(egui::Shape::line(
                edited_curve_points,
                egui::Stroke::new(2.0, egui::Color32::from_rgb(104, 176, 255)),
            ));
        }

        for point in &original_knot_points {
            painter.circle_stroke(
                to_screen(*point),
                3.0,
                egui::Stroke::new(1.0, egui::Color32::from_rgb(190, 116, 82)),
            );
        }

        for (knot, point) in edited_knot_points.iter().enumerate() {
            let pos = to_screen(*point);
            let is_closure = knot >= motion.meta.source_knot_count;
            let is_selected = interactive && knot == rd.basis_knot_edit_selected_knot as usize;
            let is_edited = rd
                .basis_knot_edits
                .as_ref()
                .is_some_and(|edits| edits.knot_is_edited(basis_id, knot));
            let color = if is_selected {
                egui::Color32::from_rgb(255, 255, 255)
            } else if is_closure {
                egui::Color32::from_rgb(255, 210, 95)
            } else if is_edited {
                egui::Color32::from_rgb(104, 176, 255)
            } else {
                egui::Color32::from_rgb(126, 186, 132)
            };
            painter.circle_filled(
                pos,
                if is_selected {
                    5.5
                } else if is_closure {
                    4.5
                } else {
                    3.8
                },
                color,
            );
            if is_selected || knot == 0 || knot + 1 == motion.meta.source_knot_count || is_closure {
                let label = if is_closure {
                    format!("C{}", knot - motion.meta.source_knot_count + 1)
                } else {
                    knot.to_string()
                };
                painter.text(
                    pos + egui::vec2(5.0, -5.0),
                    egui::Align2::LEFT_BOTTOM,
                    label,
                    egui::FontId::monospace(10.0),
                    egui::Color32::from_rgb(232, 236, 226),
                );
            }
        }
    }

    fn draw_basis_3d_view(
        &self,
        ui: &mut egui::Ui,
        rd: &mut RenderData,
        motion: &BasisBankMotionSet,
        basis_id: usize,
    ) {
        egui::CollapsingHeader::new("3D Basis View")
            .id_salt("basis_3d_view_v1")
            .default_open(true)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Drag view to rotate");
                    ui.add(
                        egui::Slider::new(&mut rd.basis_view3d_zoom, 0.2..=5.0)
                            .logarithmic(true)
                            .text("Zoom"),
                    );
                    if ui.button("Reset view").clicked() {
                        rd.basis_view3d_yaw = 0.65;
                        rd.basis_view3d_pitch = 0.35;
                        rd.basis_view3d_zoom = 1.0;
                    }
                });

                let desired_size = egui::vec2(ui.available_width().max(240.0), 260.0);
                let (rect, response) =
                    ui.allocate_exact_size(desired_size, egui::Sense::click_and_drag());
                if response.dragged() {
                    let delta = ui.input(|input| input.pointer.delta());
                    rd.basis_view3d_yaw += delta.x * 0.01;
                    rd.basis_view3d_pitch =
                        (rd.basis_view3d_pitch + delta.y * 0.01).clamp(-1.35, 1.35);
                }

                let painter = ui.painter_at(rect);
                painter.rect_filled(rect, 8.0, egui::Color32::from_rgb(25, 27, 31));
                painter.rect_stroke(
                    rect,
                    8.0,
                    egui::Stroke::new(1.0, egui::Color32::from_rgb(82, 88, 98)),
                    egui::StrokeKind::Inside,
                );

                let knot_count = motion.meta.exported_knot_count;
                let edited_knots_snapshot = rd
                    .basis_knot_edits
                    .as_ref()
                    .map(|edits| edits.edited_knots().to_vec());
                let edited_knots = edited_knots_snapshot
                    .as_deref()
                    .unwrap_or(motion.basis_knots.as_slice());
                let basis_has_knot_edits = rd
                    .basis_knot_edits
                    .as_ref()
                    .is_some_and(|edits| edits.basis_is_edited(basis_id));
                let sample_count = 128_usize.max(knot_count * 5);

                let mut all_points = Vec::with_capacity(knot_count * 2);
                for knot in 0..knot_count {
                    let base = (basis_id * knot_count + knot) * 3;
                    all_points.push([
                        motion.basis_knots[base],
                        motion.basis_knots[base + 1],
                        motion.basis_knots[base + 2],
                    ]);
                    all_points.push([
                        edited_knots[base],
                        edited_knots[base + 1],
                        edited_knots[base + 2],
                    ]);
                }
                let frame = basis_view3d_frame(all_points.as_slice());
                let scale = 0.42 * rect.width().min(rect.height()) * rd.basis_view3d_zoom
                    / frame.radius.max(1e-4);
                let center = rect.center();
                let project3 = |p: [f32; 3]| -> egui::Pos2 {
                    let p = subtract3(p, frame.center);
                    let r = rotate_basis_view_point(p, rd.basis_view3d_yaw, rd.basis_view3d_pitch);
                    egui::pos2(center.x + r[0] * scale, center.y - r[2] * scale)
                };

                let axis_len = frame.radius.max(1.0) * 0.35;
                let origin = project3(frame.origin_reference);
                for (label, point, color) in [
                    (
                        "X",
                        [axis_len, 0.0, 0.0],
                        egui::Color32::from_rgb(235, 105, 92),
                    ),
                    (
                        "Y",
                        [0.0, axis_len, 0.0],
                        egui::Color32::from_rgb(112, 190, 118),
                    ),
                    (
                        "Z",
                        [0.0, 0.0, axis_len],
                        egui::Color32::from_rgb(112, 162, 245),
                    ),
                ] {
                    let end = project3(point);
                    painter.line_segment([origin, end], egui::Stroke::new(1.6, color));
                    painter.text(
                        end + egui::vec2(4.0, -4.0),
                        egui::Align2::LEFT_BOTTOM,
                        label,
                        egui::FontId::monospace(11.0),
                        color,
                    );
                }
                painter.circle_filled(origin, 3.2, egui::Color32::from_rgb(235, 235, 226));

                let original_curve: Vec<egui::Pos2> = (0..sample_count)
                    .map(|i| {
                        let t = i as f32 / sample_count as f32;
                        project3(basis_bank_delta(
                            &motion.basis_knots,
                            motion.basis_count,
                            knot_count,
                            basis_id,
                            t,
                        ))
                    })
                    .collect();
                painter.add(egui::Shape::line(
                    original_curve,
                    egui::Stroke::new(2.2, egui::Color32::from_rgb(255, 142, 85)),
                ));
                if basis_has_knot_edits {
                    let edited_curve: Vec<egui::Pos2> = (0..sample_count)
                        .map(|i| {
                            let t = i as f32 / sample_count as f32;
                            project3(basis_bank_delta(
                                edited_knots,
                                motion.basis_count,
                                knot_count,
                                basis_id,
                                t,
                            ))
                        })
                        .collect();
                    painter.add(egui::Shape::line(
                        edited_curve,
                        egui::Stroke::new(2.0, egui::Color32::from_rgb(104, 176, 255)),
                    ));
                }

                for knot in 0..knot_count {
                    let base = (basis_id * knot_count + knot) * 3;
                    let point = [
                        edited_knots[base],
                        edited_knots[base + 1],
                        edited_knots[base + 2],
                    ];
                    let is_closure = knot >= motion.meta.source_knot_count;
                    let is_selected = knot == rd.basis_knot_edit_selected_knot as usize;
                    let color = if is_selected {
                        egui::Color32::WHITE
                    } else if is_closure {
                        egui::Color32::from_rgb(255, 210, 95)
                    } else {
                        egui::Color32::from_rgb(126, 186, 132)
                    };
                    painter.circle_filled(
                        project3(point),
                        if is_selected { 5.0 } else { 3.5 },
                        color,
                    );
                }
            });
    }

    fn draw_motion_debug_ui(&self, ui: &mut egui::Ui, rd: &mut RenderData) {
        ui.collapsing("Motion Debug", |ui| {
            let manual_response = ui.add_enabled(
                rd.has_motion && rd.basis_knot_count.is_some(),
                egui::Checkbox::new(
                    &mut rd.manual_spline_knot_preview,
                    "Manual basis knot preview",
                ),
            );
            if manual_response.changed() {
                rd.mark_motion_debug_dirty();
            }
            if let Some(knot_count) = rd.basis_knot_count.filter(|count| *count > 0) {
                let response = ui.add_enabled(
                    rd.manual_spline_knot_preview,
                    egui::Slider::new(&mut rd.selected_spline_knot, 0..=knot_count - 1)
                        .text("Basis knot"),
                );
                if response.changed() {
                    rd.mark_motion_debug_dirty();
                }
                ui.label(format!(
                    "Preview phase: {:.6}",
                    rd.spline_knot_preview_time().unwrap_or(0.0)
                ));
            }
        });
    }
    pub fn ppp(&mut self, v: f32) {
        self.context().set_pixels_per_point(v);
    }

    pub fn begin_frame(&mut self, window: Arc<Window>) {
        let raw_input = self.state.take_egui_input(window.as_ref());
        self.state.egui_ctx().begin_pass(raw_input);
        self.frame_started = true;
    }

    pub fn end_frame_and_draw(
        &mut self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        window: &Window,
        window_surface_view: &TextureView,
        screen_descriptor: ScreenDescriptor,
    ) {
        if !self.frame_started {
            panic!("begin_frame must be called before end_frame_and_draw can be called!");
        }

        self.ppp(screen_descriptor.pixels_per_point);

        let full_output = self.state.egui_ctx().end_pass();

        self.state
            .handle_platform_output(window, full_output.platform_output);

        let tris = self
            .state
            .egui_ctx()
            .tessellate(full_output.shapes, self.state.egui_ctx().pixels_per_point());
        for (id, image_delta) in &full_output.textures_delta.set {
            self.renderer
                .update_texture(device, queue, *id, image_delta);
        }
        self.renderer
            .update_buffers(device, queue, encoder, &tris, &screen_descriptor);
        let rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: window_surface_view,
                resolve_target: None,
                ops: egui_wgpu::wgpu::Operations {
                    load: egui_wgpu::wgpu::LoadOp::Load,
                    store: StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            label: Some("egui main render pass"),
            occlusion_query_set: None,
        });

        self.renderer
            .render(&mut rpass.forget_lifetime(), &tris, &screen_descriptor);
        for x in &full_output.textures_delta.free {
            self.renderer.free_texture(x)
        }

        self.frame_started = false;
    }
}

fn project_basis_point(point: [f32; 3], projection: BasisPreviewProjection) -> [f32; 2] {
    match projection {
        BasisPreviewProjection::XY => [point[0], point[1]],
        BasisPreviewProjection::XZ => [point[0], point[2]],
        BasisPreviewProjection::YZ => [point[1], point[2]],
    }
}

fn basis_knot_editor_label(
    knot: usize,
    source_knot_count: usize,
    exported_knot_count: usize,
) -> String {
    if knot < source_knot_count {
        format!(
            "source knot {} / {}",
            knot,
            source_knot_count.saturating_sub(1)
        )
    } else {
        let closure_index = knot.saturating_sub(source_knot_count) + 1;
        let closure_count = exported_knot_count.saturating_sub(source_knot_count);
        format!("closure knot C{} / C{}", closure_index, closure_count)
    }
}

fn basis_projection_edit_plane(projection: BasisPreviewProjection) -> BasisKnotEditPlane {
    match projection {
        BasisPreviewProjection::XY => BasisKnotEditPlane::XY,
        BasisPreviewProjection::XZ => BasisKnotEditPlane::XZ,
        BasisPreviewProjection::YZ => BasisKnotEditPlane::YZ,
    }
}

fn rotate_basis_view_point(point: [f32; 3], yaw: f32, pitch: f32) -> [f32; 3] {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let x = cy * point[0] - sy * point[1];
    let y = sy * point[0] + cy * point[1];
    let z = point[2];
    [x, cp * y - sp * z, sp * y + cp * z]
}

const BASIS_GRAPH_QUALITY_BALANCED_COST: f32 = 0.75;
const BASIS_GRAPH_QUALITY_STRICT_COST: f32 = 0.5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BasisGraphQualityPreset {
    Relaxed,
    Balanced,
    Strict,
    Custom,
}

impl BasisGraphQualityPreset {
    const ALL: [Self; 4] = [Self::Relaxed, Self::Balanced, Self::Strict, Self::Custom];
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BasisGraphDiagnosticsSection {
    Asset,
    Selection,
    Playback,
    Filtering,
    Branches,
}

fn basis_graph_diagnostics_section_label(section: BasisGraphDiagnosticsSection) -> &'static str {
    match section {
        BasisGraphDiagnosticsSection::Asset => "Asset",
        BasisGraphDiagnosticsSection::Selection => "Selection",
        BasisGraphDiagnosticsSection::Playback => "Playback",
        BasisGraphDiagnosticsSection::Filtering => "Filtering",
        BasisGraphDiagnosticsSection::Branches => "Branches",
    }
}

fn basis_graph_policy_ui_label(policy: BasisGraphPlaybackPolicy) -> &'static str {
    match policy {
        BasisGraphPlaybackPolicy::Continue => policy.as_str(),
        BasisGraphPlaybackPolicy::Rank0 => "Best",
        BasisGraphPlaybackPolicy::Stochastic => "Random",
    }
}

fn basis_graph_quality_preset_label(preset: BasisGraphQualityPreset) -> &'static str {
    match preset {
        BasisGraphQualityPreset::Relaxed => "Relaxed",
        BasisGraphQualityPreset::Balanced => "Balanced",
        BasisGraphQualityPreset::Strict => "Strict",
        BasisGraphQualityPreset::Custom => "Custom",
    }
}

fn basis_graph_quality_preset(
    config: crate::basis_graph_playback::BasisGraphPlaybackConfig,
) -> BasisGraphQualityPreset {
    if !config.max_position_cost_enabled
        && !config.max_velocity_cost_enabled
        && !config.max_acceleration_cost_enabled
    {
        BasisGraphQualityPreset::Relaxed
    } else if basis_graph_quality_matches(config, BASIS_GRAPH_QUALITY_BALANCED_COST) {
        BasisGraphQualityPreset::Balanced
    } else if basis_graph_quality_matches(config, BASIS_GRAPH_QUALITY_STRICT_COST) {
        BasisGraphQualityPreset::Strict
    } else {
        BasisGraphQualityPreset::Custom
    }
}

fn basis_graph_quality_matches(
    config: crate::basis_graph_playback::BasisGraphPlaybackConfig,
    cost: f32,
) -> bool {
    config.max_position_cost_enabled
        && config.max_velocity_cost_enabled
        && config.max_acceleration_cost_enabled
        && (config.max_position_cost - cost).abs() <= f32::EPSILON
        && (config.max_velocity_cost - cost).abs() <= f32::EPSILON
        && (config.max_acceleration_cost - cost).abs() <= f32::EPSILON
}

fn apply_basis_graph_quality_preset(
    config: &mut crate::basis_graph_playback::BasisGraphPlaybackConfig,
    preset: BasisGraphQualityPreset,
) {
    match preset {
        BasisGraphQualityPreset::Relaxed => {
            config.max_position_cost_enabled = false;
            config.max_velocity_cost_enabled = false;
            config.max_acceleration_cost_enabled = false;
        }
        BasisGraphQualityPreset::Balanced => {
            apply_basis_graph_quality_cost(config, BASIS_GRAPH_QUALITY_BALANCED_COST);
        }
        BasisGraphQualityPreset::Strict => {
            apply_basis_graph_quality_cost(config, BASIS_GRAPH_QUALITY_STRICT_COST);
        }
        BasisGraphQualityPreset::Custom => {}
    }
}

fn apply_basis_graph_quality_cost(
    config: &mut crate::basis_graph_playback::BasisGraphPlaybackConfig,
    cost: f32,
) {
    config.max_position_cost_enabled = true;
    config.max_position_cost = cost;
    config.max_velocity_cost_enabled = true;
    config.max_velocity_cost = cost;
    config.max_acceleration_cost_enabled = true;
    config.max_acceleration_cost = cost;
}

fn basis_graph_last_edge_label(edge: BasisGraphLastEdge) -> String {
    match edge {
        BasisGraphLastEdge::Reset => "reset".to_string(),
        BasisGraphLastEdge::Continue => "default successor".to_string(),
        BasisGraphLastEdge::Branch {
            rank,
            to_basis_id,
            to_segment,
        } => format!(
            "branch rank {} -> global {} / segment {}",
            rank, to_basis_id, to_segment
        ),
    }
}

fn branch_rejection_quality_label(rejection: BasisBranchRejection) -> &'static str {
    if rejection.score {
        "Score"
    } else if rejection.position {
        "Position"
    } else if rejection.velocity {
        "Velocity"
    } else if rejection.acceleration {
        "Acceleration"
    } else {
        "Good"
    }
}

fn branch_rejection_status_label(rejection: BasisBranchRejection) -> &'static str {
    if rejection.rejected() {
        "Rejected"
    } else {
        "Usable"
    }
}

fn branch_rejection_label(rejection: BasisBranchRejection) -> String {
    let mut reasons = Vec::new();
    if rejection.score {
        reasons.push("score");
    }
    if rejection.position {
        reasons.push("position");
    }
    if rejection.velocity {
        reasons.push("velocity");
    }
    if rejection.acceleration {
        reasons.push("acceleration");
    }
    if reasons.is_empty() {
        String::new()
    } else {
        format!(" | rejected by {}", reasons.join(", "))
    }
}

fn parse_basis_batch_selection(text: &str, valid_ids: &[usize]) -> Result<Vec<usize>, String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err("Enter basis IDs or ranges".to_string());
    }
    if trimmed.eq_ignore_ascii_case("all") {
        if valid_ids.is_empty() {
            return Err("No basis IDs available".to_string());
        }
        return Ok(valid_ids.to_vec());
    }
    let valid: HashSet<usize> = valid_ids.iter().copied().collect();
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for token in trimmed.split(',') {
        let token = token.trim();
        if token.is_empty() {
            return Err("Empty basis entry".to_string());
        }
        if let Some((start_s, end_s)) = token.split_once('-') {
            if end_s.contains('-') {
                return Err(format!("Invalid range '{token}'"));
            }
            let start = start_s
                .trim()
                .parse::<usize>()
                .map_err(|_| format!("Invalid basis '{token}'"))?;
            let end = end_s
                .trim()
                .parse::<usize>()
                .map_err(|_| format!("Invalid basis '{token}'"))?;
            if start > end {
                return Err(format!("Range '{token}' is reversed"));
            }
            for id in start..=end {
                if !valid.contains(&id) {
                    return Err(format!("Basis {id} is out of range"));
                }
                if seen.insert(id) {
                    out.push(id);
                }
            }
        } else {
            let id = token
                .parse::<usize>()
                .map_err(|_| format!("Invalid basis '{token}'"))?;
            if !valid.contains(&id) {
                return Err(format!("Basis {id} is out of range"));
            }
            if seen.insert(id) {
                out.push(id);
            }
        }
    }
    if out.is_empty() {
        Err("No basis IDs selected".to_string())
    } else {
        Ok(out)
    }
}

fn basis_batch_ids_for_motion(
    motion: &BasisBankMotionSet,
    selection_text: &str,
) -> Result<Vec<usize>, String> {
    let valid_ids: Vec<_> = (0..motion.basis_count).collect();
    parse_basis_batch_selection(selection_text, &valid_ids)
}
fn apply_basis_batch_scalar_edit(
    basis_edit_overrides: &mut [BasisEditOverride],
    basis_ids: &[usize],
    mut edit: BasisEditOverride,
) -> bool {
    edit.enabled = true;
    let mut changed = false;
    for &basis_id in basis_ids {
        if let Some(slot) = basis_edit_overrides.get_mut(basis_id) {
            if *slot != edit {
                *slot = edit;
                changed = true;
            }
        }
    }
    changed
}

fn reset_basis_batch_scalar_edits(
    basis_edit_overrides: &mut [BasisEditOverride],
    basis_ids: &[usize],
) -> bool {
    let mut changed = false;
    for &basis_id in basis_ids {
        if let Some(slot) = basis_edit_overrides.get_mut(basis_id) {
            if *slot != BasisEditOverride::default() {
                *slot = BasisEditOverride::default();
                changed = true;
            }
        }
    }
    changed
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct BasisView3dFrame {
    center: [f32; 3],
    radius: f32,
    origin_reference: [f32; 3],
}

fn basis_view3d_frame(points: &[[f32; 3]]) -> BasisView3dFrame {
    if points.is_empty() {
        return BasisView3dFrame {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
            origin_reference: [0.0, 0.0, 0.0],
        };
    }
    let mut min_p = points[0];
    let mut max_p = points[0];
    for point in points.iter().copied().skip(1) {
        for axis in 0..3 {
            min_p[axis] = min_p[axis].min(point[axis]);
            max_p[axis] = max_p[axis].max(point[axis]);
        }
    }
    let center = [
        0.5 * (min_p[0] + max_p[0]),
        0.5 * (min_p[1] + max_p[1]),
        0.5 * (min_p[2] + max_p[2]),
    ];
    let radius = points
        .iter()
        .map(|point| {
            let dx = point[0] - center[0];
            let dy = point[1] - center[1];
            let dz = point[2] - center[2];
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .fold(1e-4_f32, f32::max);
    BasisView3dFrame {
        center,
        radius: radius.max(1e-4),
        origin_reference: [0.0, 0.0, 0.0],
    }
}

fn subtract3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn reset_selected_basis_authoring_edits(
    basis_edit_overrides: &mut [BasisEditOverride],
    basis_knot_edits: Option<&mut BasisKnotEditState>,
    basis_id: usize,
) -> (BasisEditOverride, bool, bool) {
    let scalar_changed = basis_edit_overrides
        .get(basis_id)
        .is_some_and(|edit| *edit != BasisEditOverride::default());
    if basis_id < basis_edit_overrides.len() {
        reset_basis_edit(basis_edit_overrides, basis_id);
    }
    let knot_changed = basis_knot_edits.is_some_and(|edits| edits.reset_basis(basis_id));
    (
        basis_edit_overrides
            .get(basis_id)
            .copied()
            .unwrap_or_default(),
        scalar_changed,
        knot_changed,
    )
}

fn reset_all_basis_authoring_edits(
    basis_edit_overrides: &mut [BasisEditOverride],
    basis_knot_edits: Option<&mut BasisKnotEditState>,
    selected_basis_id: usize,
) -> (BasisEditOverride, bool, bool) {
    let scalar_changed = basis_edit_overrides
        .iter()
        .any(|edit| *edit != BasisEditOverride::default());
    reset_all_basis_edits(basis_edit_overrides);
    let knot_changed = if let Some(edits) = basis_knot_edits {
        edits.reset_all();
        true
    } else {
        false
    };
    (
        basis_edit_overrides
            .get(selected_basis_id)
            .copied()
            .unwrap_or_default(),
        scalar_changed,
        knot_changed,
    )
}

fn basis_bank_status_summary(rd: &RenderData) -> String {
    let active_top_k = rd
        .basis_bank_active_top_k
        .or(rd.basis_bank_top_k)
        .map(|v| v.to_string())
        .unwrap_or_else(|| "n/a".to_string());
    let exported_top_k = rd
        .basis_bank_top_k
        .map(|v| v.to_string())
        .unwrap_or_else(|| "n/a".to_string());
    let Some(motion) = rd.basis_bank_preview.as_ref() else {
        let basis_count = rd
            .basis_bank_basis_count
            .map(|v| v.to_string())
            .unwrap_or_else(|| "n/a".to_string());
        return format!(
            "Basis bank: basis={}, basis used per splat {}/{}",
            basis_count, active_top_k, exported_top_k
        );
    };
    format!(
        "Basis bank: shared LoD0, {} shared bases, basis used per splat {}/{}",
        motion.basis_count, active_top_k, exported_top_k
    )
}

fn artist_basis_label(basis_id: usize) -> String {
    format!("Shared LoD0 basis {}", basis_id)
}

fn artist_basis_label_for_global_in_motion(
    _motion: &BasisBankMotionSet,
    basis_id: usize,
) -> String {
    artist_basis_label(basis_id)
}

fn artist_branch_target_label(basis_id: usize, segment: usize) -> String {
    format!("shared basis {} / segment {}", basis_id, segment)
}

fn motion_authoring_default_pos() -> [f32; 2] {
    [
        GSWT_MAIN_DEFAULT_POS[0] + GSWT_MAIN_DEFAULT_SIZE[0] + PANEL_GAP,
        GSWT_MAIN_DEFAULT_POS[1],
    ]
}

fn performance_default_pos() -> [f32; 2] {
    [
        GSWT_MAIN_DEFAULT_POS[0],
        GSWT_MAIN_DEFAULT_POS[1] + GSWT_MAIN_DEFAULT_SIZE[1] + PANEL_GAP / 2.0,
    ]
}

fn format_vec3(v: [f32; 3]) -> String {
    format!("({:.6}, {:.6}, {:.6})", v[0], v[1], v[2])
}

fn basis_plot_bounds(points: &[[f32; 2]]) -> ([f32; 2], [f32; 2]) {
    let mut min_xy = [0.0_f32; 2];
    let mut max_xy = [0.0_f32; 2];
    for point in points {
        min_xy[0] = min_xy[0].min(point[0]);
        min_xy[1] = min_xy[1].min(point[1]);
        max_xy[0] = max_xy[0].max(point[0]);
        max_xy[1] = max_xy[1].max(point[1]);
    }
    if points.is_empty() {
        ([0.0, 0.0], [1.0, 1.0])
    } else {
        (min_xy, max_xy)
    }
}

#[cfg(test)]
mod tests {
    use super::parse_basis_batch_selection;

    #[test]
    fn batch_selection_accepts_all_case_insensitively() {
        let valid = vec![0, 1, 2, 3];
        assert_eq!(parse_basis_batch_selection("all", &valid).unwrap(), valid);
        assert_eq!(parse_basis_batch_selection(" ALL ", &valid).unwrap(), valid);
    }

    #[test]
    fn batch_selection_keeps_ranges_and_rejects_empty_or_invalid_ids() {
        let valid = vec![0, 1, 2, 3];
        assert_eq!(
            parse_basis_batch_selection("0,2-3", &valid).unwrap(),
            vec![0, 2, 3]
        );
        assert!(parse_basis_batch_selection("", &valid).is_err());
        assert!(parse_basis_batch_selection("4", &valid).is_err());
    }
}
