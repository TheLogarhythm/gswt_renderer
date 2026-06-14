#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
use winit::event_loop::EventLoop;

mod app;
mod basis_bank_edit;
mod basis_bank_motion;
mod basis_bank_motion_gpu;
mod basis_graph_playback;
mod basis_motion_graph;
mod camera;
mod catmull_rom_motion;
mod catmull_rom_motion_gpu;
mod control;
pub mod deformation;
mod deformation_gpu;
mod gui;
pub mod motion;
mod proxy;
mod renderer;
mod scene;
mod skybox;
mod state;
mod structure;
mod texture;
mod utils;
mod wangtile;

use app::App;

pub fn run() -> anyhow::Result<()> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
    }
    #[cfg(target_arch = "wasm32")]
    {
        console_log::init_with_level(log::Level::Info).unwrap_throw();
    }

    let event_loop = EventLoop::with_user_event().build()?;
    let mut app = App::new(
        #[cfg(target_arch = "wasm32")]
        &event_loop,
    );
    event_loop.run_app(&mut app)?;

    Ok(())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn run_web() -> Result<(), wasm_bindgen::JsValue> {
    console_error_panic_hook::set_once();
    run().unwrap_throw();

    Ok(())
}
