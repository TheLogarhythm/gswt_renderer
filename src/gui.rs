// https://github.com/kaphula/winit-egui-wgpu-template/blob/master/src/egui_tools.rs

use std::sync::Arc;

use egui::Context;
use egui_wgpu::wgpu::{CommandEncoder, Device, Queue, StoreOp, TextureFormat, TextureView};
use egui_wgpu::{Renderer, RendererOptions, ScreenDescriptor, wgpu};
use egui_winit::State;
use num_format::{Locale, ToFormattedString};
use winit::event::WindowEvent;
use winit::window::Window;

use crate::camera::Camera;
use crate::catmull_rom_motion::MotionMode;
use crate::control::{CameraControl, FlyPathControl, FlyPathFrame};
use crate::log;
use crate::proxy::upload_proxy_texture;
use crate::skybox::upload_skybox;
use crate::structure::*;
use crate::utils::*;
use crate::wangtile::upload_height_map;

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

                                    if rd.has_deformation {
                                        self.draw_dynamics_ui(ui, rd);
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
                    egui::Window::new("Performance").show(&self.context().clone(), |ui| {
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
                                    blending_splat_count =
                                        rd.cur_scene_data.as_ref().unwrap().blending_splat_count;
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
                                            rd.cur_scene_data.as_ref().unwrap().lod_instance_count
                                                [i]
                                        } else {
                                            0
                                        };
                                        ui.label(instance_count.to_formatted_string(&Locale::en));
                                        ui.end_row();
                                    }
                                });
                        });
                    });
                }

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

    fn draw_dynamics_ui(&self, ui: &mut egui::Ui, rd: &mut RenderData) {
        ui.add(egui::Label::new("Dynamics"));
        ui.vertical(|ui| {
            ui.label(if rd.animation_playing {
                "Playing"
            } else {
                "Frozen"
            });
            ui.label(format!("Backend: {}", rd.active_motion_mode.as_str()));

            self.draw_motion_debug_ui(ui, rd);
            self.draw_motion_compatibility_ui(ui, rd);
            self.draw_motion_residual_ui(ui, rd);
        });
        ui.end_row();
    }

    fn draw_motion_debug_ui(&self, ui: &mut egui::Ui, rd: &mut RenderData) {
        ui.collapsing("Motion Debug", |ui| {
            let network_active = rd.active_motion_mode == MotionMode::DeformationNetwork;
            let rot_response = ui.add_enabled(
                network_active,
                egui::Checkbox::new(
                    &mut rd.apply_network_delta_rot,
                    "Apply network delta rotation",
                ),
            );
            if rot_response.changed() {
                rd.mark_motion_debug_dirty();
            }
            if !network_active {
                ui.label("Rotation toggle is for deformation-network mode.");
            }

            ui.separator();

            let catmull_rom_active = rd.active_motion_mode == MotionMode::CatmullRom
                && rd.catmull_rom_knot_count.is_some();
            let manual_response = ui.add_enabled(
                catmull_rom_active,
                egui::Checkbox::new(
                    &mut rd.manual_spline_knot_preview,
                    "Manual spline knot preview",
                ),
            );
            if manual_response.changed() {
                rd.mark_motion_debug_dirty();
            }

            if let Some(knot_count) = rd.catmull_rom_knot_count {
                if knot_count > 0 {
                    let before = rd.selected_spline_knot;
                    let slider_response = ui.add_enabled(
                        catmull_rom_active && rd.manual_spline_knot_preview,
                        egui::Slider::new(&mut rd.selected_spline_knot, 0..=knot_count - 1)
                            .text("Spline knot"),
                    );
                    if slider_response.changed() || rd.selected_spline_knot != before {
                        rd.mark_motion_debug_dirty();
                    }
                    let preview_time = rd.spline_knot_preview_time().unwrap_or(0.0);
                    ui.label(format!(
                        "Preview time: {}/{} = {:.6}",
                        rd.selected_spline_knot, knot_count, preview_time
                    ));
                }
            } else {
                ui.label("Spline knot preview is for Catmull-Rom mode.");
            }
        });
    }

    fn draw_motion_compatibility_ui(&self, ui: &mut egui::Ui, rd: &mut RenderData) {
        ui.collapsing("Motion Compatibility", |ui| {
            let catmull_rom_active = rd.active_motion_mode == MotionMode::CatmullRom
                && rd.catmull_rom_knot_count.is_some();
            ui.label(format!(
                "Spline time sampling: {}",
                if rd.catmull_rom_uses_volume_key_times {
                    "volume keys"
                } else {
                    "periodic"
                }
            ));
            ui.label(format!(
                "Spline knots: {}",
                rd.catmull_rom_knot_count
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "n/a".to_string())
            ));
            ui.label(format!(
                "Volume keys: {}",
                rd.motion_compatibility_volume_keys
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "n/a".to_string())
            ));
            ui.label("Target: renderer prebaked xyzt volume, not direct HexPlane/MLP.");
            ui.horizontal(|ui| {
                ui.radio_value(
                    &mut rd.motion_compatibility_scope,
                    MotionCompatibilityScope::SelectedKnot,
                    "Selected knot only",
                );
                ui.radio_value(
                    &mut rd.motion_compatibility_scope,
                    MotionCompatibilityScope::AllKnots,
                    "All knots",
                );
            });

            let compare_response = ui.add_enabled(
                catmull_rom_active && !rd.motion_compatibility_running,
                egui::Button::new("Compare spline knots to volume"),
            );
            if compare_response.clicked() {
                rd.motion_compatibility_requested = true;
            }
            if rd.motion_compatibility_running {
                ui.label("Compatibility comparison: running...");
            }
            if let Some(err) = rd.motion_compatibility_error.as_ref() {
                ui.colored_label(
                    egui::Color32::LIGHT_RED,
                    format!("Compatibility error: {}", err),
                );
            }
            if let Some(result) = rd.motion_compatibility_result.as_ref() {
                self.draw_motion_compatibility_result(ui, result);
            }

            ui.separator();
            ui.label("Texture compare: final GPU Gaussian means after backend dispatch.");
            let texture_compare_response = ui.add_enabled(
                catmull_rom_active && !rd.motion_texture_compare_running,
                egui::Button::new("Compare final means to volume"),
            );
            if texture_compare_response.clicked() {
                rd.motion_texture_compare_requested = true;
            }
            if rd.motion_texture_compare_running {
                ui.label("Texture comparison: running...");
            }
            if let Some(err) = rd.motion_texture_compare_error.as_ref() {
                ui.colored_label(
                    egui::Color32::LIGHT_RED,
                    format!("Texture compare error: {}", err),
                );
            }
            if let Some(result) = rd.motion_texture_compare_result.as_ref() {
                self.draw_motion_texture_compare_result(ui, result);
            }
            if !catmull_rom_active {
                ui.label("Comparisons are for Catmull-Rom mode.");
            }
        });
    }

    fn draw_motion_compatibility_result(
        &self,
        ui: &mut egui::Ui,
        result: &MotionCompatibilityResult,
    ) {
        ui.label(format!("Scope: {}", result.scope.as_str()));
        ui.label(format!(
            "Compared: {} splats, {} / {} knots",
            result.splat_count, result.compared_knots, result.knot_count
        ));
        ui.label(format!(
            "Actual compared items: {}",
            result
                .actual_compared_count
                .to_formatted_string(&Locale::en)
        ));
        ui.label(format!(
            "Mean/RMS/Max: {:.6} / {:.6} / {:.6}",
            result.mean_error, result.rms_error, result.max_error
        ));
        ui.label(format!(
            "Spline |delta| mean/max: {:.6} / {:.6}",
            result.mean_spline_delta_magnitude, result.max_spline_delta_magnitude
        ));
        ui.label(format!(
            "Volume |delta| mean/max: {:.6} / {:.6}",
            result.mean_volume_delta_magnitude, result.max_volume_delta_magnitude
        ));
        ui.label(format!(
            "Nonzero error/spline/volume: {} / {} / {}",
            result.nonzero_error_count.to_formatted_string(&Locale::en),
            result
                .nonzero_spline_delta_count
                .to_formatted_string(&Locale::en),
            result
                .nonzero_volume_delta_count
                .to_formatted_string(&Locale::en)
        ));
        ui.label(format!(
            "Sampled p95: {:.6} ({} samples)",
            result.sampled_p95_error, result.sampled_error_count
        ));
        ui.label(format!(
            "Worst: knot {}, splat {}",
            result.worst_knot, result.worst_splat
        ));
    }

    fn draw_motion_texture_compare_result(
        &self,
        ui: &mut egui::Ui,
        result: &MotionTextureCompareResult,
    ) {
        ui.label(format!(
            "Time: Catmull {:.6}, Volume {:.6}",
            result.time01, result.volume_time01
        ));
        ui.label(format!(
            "Compared final means: {} / {} splats",
            result
                .actual_compared_count
                .to_formatted_string(&Locale::en),
            result.splat_count.to_formatted_string(&Locale::en)
        ));
        ui.label(format!(
            "Mean/RMS/Max: {:.6} / {:.6} / {:.6}",
            result.mean_error, result.rms_error, result.max_error
        ));
        ui.label(format!(
            "Catmull |mean| mean/max: {:.6} / {:.6}",
            result.mean_catmull_rom_mean_magnitude, result.max_catmull_rom_mean_magnitude
        ));
        ui.label(format!(
            "Volume |mean| mean/max: {:.6} / {:.6}",
            result.mean_volume_mean_magnitude, result.max_volume_mean_magnitude
        ));
        ui.label(format!(
            "Nonzero final-mean error: {}",
            result.nonzero_error_count.to_formatted_string(&Locale::en)
        ));
        ui.label(format!(
            "Sampled p95: {:.6} ({} samples)",
            result.sampled_p95_error, result.sampled_error_count
        ));
        ui.label(format!("Worst splat: {}", result.worst_splat));
    }

    fn draw_motion_residual_ui(&self, ui: &mut egui::Ui, rd: &mut RenderData) {
        ui.collapsing("Local edge variation residual", |ui| {
            let motion = &mut rd.render_config.motion_edit;
            ui.checkbox(&mut motion.enabled, "Enable residual");
            ui.add(egui::Slider::new(&mut motion.amplitude, 0.0..=0.25).text("Amplitude"));
            ui.add(egui::Slider::new(&mut motion.edge_band, 0.01..=0.5).text("Edge band"));
            ui.add(
                egui::Slider::new(&mut motion.wave_phase_span, 0.0..=1.0).text("Wave phase span"),
            );
            ui.add(
                egui::Slider::new(&mut motion.detail_amplitude, 0.0..=1.0).text("Detail amplitude"),
            );
            ui.horizontal(|ui| {
                if ui.button("Zero motion").clicked() {
                    motion.zero_motion();
                }
                if ui.button("Load wave preset").clicked() {
                    motion.load_wave_preset();
                }
            });
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
