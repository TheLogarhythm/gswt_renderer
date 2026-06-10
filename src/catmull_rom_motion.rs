use std::collections::{HashMap, HashSet};
use std::io::{Cursor, Read};
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use regex::Regex;
use zip::ZipArchive;

use crate::log;
use crate::scene::Scene;

pub const CATMULL_ROM_FORMAT: &str = "periodic_catmull_rom_delta_xyz";
pub const CATMULL_ROM_META_FILENAME: &str = "motion_catmull_rom_meta.pt";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CatmullRomTimeSampling {
    Periodic,
    VolumeKeys,
}

impl CatmullRomTimeSampling {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Periodic => "periodic",
            Self::VolumeKeys => "volume_keys",
        }
    }

    pub fn shader_value(self) -> u32 {
        match self {
            Self::Periodic => 0,
            Self::VolumeKeys => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CatmullRomSampleTimeGrid {
    Periodic,
    VolumeKeys,
    Irregular,
}

impl CatmullRomSampleTimeGrid {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Periodic => "periodic",
            Self::VolumeKeys => "volume_keys",
            Self::Irregular => "irregular",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CatmullRomMotionTeacher {
    Legacy,
    Volume,
    DirectNetwork,
}

impl CatmullRomMotionTeacher {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::Volume => "volume",
            Self::DirectNetwork => "direct_network",
        }
    }
}

#[derive(Debug, Clone)]
pub struct CatmullRomMotionMeta {
    pub knot_count: usize,
    pub include_lods: Vec<usize>,
    pub sample_times: Vec<f32>,
    pub time_sampling: CatmullRomTimeSampling,
    pub sample_time_grid: CatmullRomSampleTimeGrid,
    pub has_time_sampling_field: bool,
    pub periodic_flag: bool,
    pub motion_teacher: CatmullRomMotionTeacher,
    pub volume_res: Option<usize>,
    pub volume_key_count: Option<usize>,
    pub source_knot_count: Option<usize>,
    pub exported_knot_count: Option<usize>,
    pub loop_closure_knots: Option<usize>,
    pub loop_closure_method: Option<String>,
    pub source_frame_count: Option<usize>,
    pub source_fps: Option<f32>,
    pub duration_seconds: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct CatmullRomTileMotion {
    pub tile_index: usize,
    pub lod_index: usize,
    pub splat_count: usize,
    /// K-major storage from torch: ((knot * splat_count + splat) * 3 + xyz).
    pub knots: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct CatmullRomMotionSet {
    pub meta: CatmullRomMotionMeta,
    pub total_splats: usize,
    /// Splat-major storage in renderer merge order: ((splat * knot_count + knot) * 3 + xyz).
    pub global_knots: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct CatmullRomMotionZipEntry {
    pub index: usize,
    pub filename: String,
    pub tile_id: usize,
    pub lod_id: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotionMode {
    Auto,
    BasisBank,
    CatmullRom,
    DeformationNetwork,
    Static,
}

impl MotionMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::BasisBank => "basis_bank",
            Self::CatmullRom => "catmull_rom",
            Self::DeformationNetwork => "deformation_network",
            Self::Static => "static",
        }
    }

    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "basis_bank" | "basis-bank" | "basis" => Some(Self::BasisBank),
            "catmull_rom" | "catmull-rom" | "spline" => Some(Self::CatmullRom),
            "deformation_network" | "deformation-network" | "network" | "volume" => {
                Some(Self::DeformationNetwork)
            }
            "static" | "none" => Some(Self::Static),
            _ => None,
        }
    }

    pub fn from_url_query() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Self::Auto
        }
        #[cfg(target_arch = "wasm32")]
        {
            let mut mode = Self::Auto;
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
                if let Some(raw) = part.strip_prefix("motion_mode=") {
                    if let Some(parsed) = Self::parse(raw) {
                        mode = parsed;
                    }
                } else if let Some(raw) = part.strip_prefix("deform_mode=") {
                    match raw {
                        "basis_bank" | "basis-bank" | "basis" => mode = Self::BasisBank,
                        "catmull_rom" | "catmull-rom" | "spline" => mode = Self::CatmullRom,
                        "static" | "none" => mode = Self::Static,
                        "volume" | "hexplane_mlp" | "identity" => mode = Self::DeformationNetwork,
                        _ => {}
                    }
                }
            }
            mode
        }
    }
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn catmull_rom_delta(knots: &[[f32; 3]], t: f32) -> [f32; 3] {
    let k = knots.len();
    assert!(k >= 4);
    let phase = t.rem_euclid(1.0);
    let scaled = phase * k as f32;
    let segment = scaled.floor() as usize;
    let u = scaled - segment as f32;
    let p0 = knots[(segment + k - 1) % k];
    let p1 = knots[segment % k];
    let p2 = knots[(segment + 1) % k];
    let p3 = knots[(segment + 2) % k];
    let u2 = u * u;
    let u3 = u2 * u;
    let mut out = [0.0_f32; 3];
    for c in 0..3 {
        out[c] = 0.5
            * (2.0 * p1[c]
                + (-p0[c] + p2[c]) * u
                + (2.0 * p0[c] - 5.0 * p1[c] + 4.0 * p2[c] - p3[c]) * u2
                + (-p0[c] + 3.0 * p1[c] - 3.0 * p2[c] + p3[c]) * u3);
    }
    out
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn catmull_rom_delta_volume_keys(knots: &[[f32; 3]], t: f32) -> [f32; 3] {
    let k = knots.len();
    assert!(k >= 4);
    let scaled = t.clamp(0.0, 1.0) * (k - 1) as f32;
    let segment = (scaled.floor() as usize).min(k - 1);
    let u = scaled - segment as f32;
    let p0 = knots[if segment > 0 { segment - 1 } else { 0 }];
    let p1 = knots[segment];
    let p2 = knots[(segment + 1).min(k - 1)];
    let p3 = knots[(segment + 2).min(k - 1)];
    let u2 = u * u;
    let u3 = u2 * u;
    let mut out = [0.0_f32; 3];
    for c in 0..3 {
        out[c] = 0.5
            * (2.0 * p1[c]
                + (-p0[c] + p2[c]) * u
                + (2.0 * p0[c] - 5.0 * p1[c] + 4.0 * p2[c] - p3[c]) * u2
                + (-p0[c] + 3.0 * p1[c] - 3.0 * p2[c] + p3[c]) * u3);
    }
    out
}

pub fn detect_motion_file(filename: &str) -> Option<(usize, usize)> {
    let re = Regex::new(r"^tile(\d+)_lod(\d+)_motion_catmull_rom\.pt$").ok()?;
    let caps = re.captures(filename)?;
    let tile_id = caps.get(1)?.as_str().parse().ok()?;
    let lod_id = caps.get(2)?.as_str().parse().ok()?;
    Some((tile_id, lod_id))
}

pub fn load_catmull_rom_motion_from_zip<R: Read + std::io::Seek>(
    archive: &mut ZipArchive<R>,
    meta_index: usize,
    entries: &[CatmullRomMotionZipEntry],
    scene_vec: &[Vec<Scene>],
) -> Result<Option<Arc<CatmullRomMotionSet>>> {
    let meta_bytes = read_zip_entry(archive, meta_index, CATMULL_ROM_META_FILENAME)?;
    let meta =
        parse_catmull_rom_meta_pt(&meta_bytes).context("failed to parse Catmull-Rom meta")?;
    let included_lods: HashSet<usize> = meta.include_lods.iter().copied().collect();
    let mut entry_by_lod_tile: HashMap<(usize, usize), &CatmullRomMotionZipEntry> = HashMap::new();
    for entry in entries {
        entry_by_lod_tile.insert((entry.lod_id, entry.tile_id), entry);
    }

    let mut total_splats = 0_usize;
    let mut global_knots = Vec::new();
    for (lod_id, lod_vec) in scene_vec.iter().enumerate() {
        for (tile_id, scene) in lod_vec.iter().enumerate() {
            total_splats += scene.splat_count;
            if !included_lods.contains(&lod_id) {
                global_knots.resize(
                    global_knots.len() + scene.splat_count * meta.knot_count * 3,
                    0.0,
                );
                continue;
            }
            let Some(entry) = entry_by_lod_tile.get(&(lod_id, tile_id)) else {
                log!(
                    "Catmull-Rom motion missing for tile{}_lod{}; disabling Catmull-Rom backend.",
                    tile_id,
                    lod_id
                );
                return Ok(None);
            };
            let bytes = read_zip_entry(archive, entry.index, &entry.filename)?;
            let tile_motion = parse_catmull_rom_tile_pt(&bytes, meta.knot_count)
                .with_context(|| format!("failed to parse {}", entry.filename))?;
            if tile_motion.tile_index != tile_id || tile_motion.lod_index != lod_id {
                log!(
                    "Catmull-Rom motion index mismatch in {}: file says tile{}_lod{}.",
                    entry.filename,
                    tile_motion.tile_index,
                    tile_motion.lod_index
                );
                return Ok(None);
            }
            if tile_motion.splat_count != scene.splat_count {
                log!(
                    "Catmull-Rom motion splat mismatch for tile{}_lod{}: motion={}, scene={}.",
                    tile_id,
                    lod_id,
                    tile_motion.splat_count,
                    scene.splat_count
                );
                return Ok(None);
            }
            if let Err(err) = append_tile_knots_splat_major(
                &mut global_knots,
                &tile_motion,
                meta.knot_count,
                scene.source_row_indices.as_slice(),
            ) {
                log!(
                    "Catmull-Rom motion reorder failed for tile{}_lod{}: {}; disabling Catmull-Rom backend.",
                    tile_id,
                    lod_id,
                    err
                );
                return Ok(None);
            }
        }
    }

    log!(
        "Catmull-Rom motion loaded: knots={}, time_sampling={} (field_present={}), sample_time_grid={}, periodic={}, motion_teacher={}, volume_res={:?}, volume_key_count={:?}, source_knot_count={:?}, exported_knot_count={:?}, loop_closure_knots={:?}, loop_closure_method={:?}, sample_times={}..{}, included_lods={:?}, source_frames={:?}, source_fps={:?}, duration={:?}, files={}, total_splats={}",
        meta.knot_count,
        meta.time_sampling.as_str(),
        meta.has_time_sampling_field,
        meta.sample_time_grid.as_str(),
        meta.periodic_flag,
        meta.motion_teacher.as_str(),
        meta.volume_res,
        meta.volume_key_count,
        meta.source_knot_count,
        meta.exported_knot_count,
        meta.loop_closure_knots,
        meta.loop_closure_method,
        meta.sample_times.first().copied().unwrap_or(0.0),
        meta.sample_times.last().copied().unwrap_or(0.0),
        meta.include_lods,
        meta.source_frame_count,
        meta.source_fps,
        meta.duration_seconds,
        entries.len(),
        total_splats
    );
    if meta.time_sampling == CatmullRomTimeSampling::Periodic
        && meta.sample_time_grid == CatmullRomSampleTimeGrid::Periodic
    {
        log!(
            "Catmull-Rom motion audit: asset uses periodic knot times k/K; it is not a volume-key export k/(K-1)."
        );
    }
    log!("Catmull-Rom motion reordered with renderer PLY source-row permutations.");
    Ok(Some(Arc::new(CatmullRomMotionSet {
        meta,
        total_splats,
        global_knots,
    })))
}

fn append_tile_knots_splat_major(
    out: &mut Vec<f32>,
    tile_motion: &CatmullRomTileMotion,
    knot_count: usize,
    source_row_indices: &[u32],
) -> Result<()> {
    if source_row_indices.len() != tile_motion.splat_count {
        bail!(
            "source-row permutation length {} != motion splat count {}",
            source_row_indices.len(),
            tile_motion.splat_count
        );
    }
    for splat in 0..tile_motion.splat_count {
        let source_splat = source_row_indices[splat] as usize;
        if source_splat >= tile_motion.splat_count {
            bail!(
                "source-row permutation contains out-of-range row {} for {} splats",
                source_splat,
                tile_motion.splat_count
            );
        }
        for knot in 0..knot_count {
            let src = (knot * tile_motion.splat_count + source_splat) * 3;
            out.extend_from_slice(&tile_motion.knots[src..src + 3]);
        }
    }
    Ok(())
}

pub fn parse_catmull_rom_meta_pt(bytes: &[u8]) -> Result<CatmullRomMotionMeta> {
    let parsed = ParsedPt::from_bytes(bytes)?;
    validate_common_fields(&parsed)?;
    let knot_count = parsed
        .int_field("knot_count")
        .context("missing knot_count")? as usize;
    let include_lods = parsed
        .int_tuple_field("include_lods")
        .context("missing include_lods")?
        .into_iter()
        .map(|v| v as usize)
        .collect::<Vec<_>>();
    let sample_times = parsed.storage_f32(0)?;
    if sample_times.len() != knot_count {
        bail!(
            "sample_times length {} != knot_count {}",
            sample_times.len(),
            knot_count
        );
    }
    let time_sampling_field = parsed.string_field("time_sampling");
    let time_sampling = match time_sampling_field.as_deref() {
        Some("volume_keys") => CatmullRomTimeSampling::VolumeKeys,
        Some("periodic") | None => CatmullRomTimeSampling::Periodic,
        Some(other) => bail!("unsupported Catmull-Rom time_sampling '{}'", other),
    };
    let sample_time_grid = classify_catmull_rom_sample_times(sample_times.as_slice());
    let periodic_flag = parsed.bool_field("periodic").unwrap_or(false);
    let motion_teacher = match parsed.string_field("motion_teacher").as_deref() {
        Some("volume") => CatmullRomMotionTeacher::Volume,
        Some("direct_network") => CatmullRomMotionTeacher::DirectNetwork,
        None => CatmullRomMotionTeacher::Legacy,
        Some(other) => bail!("unsupported Catmull-Rom motion_teacher '{}'", other),
    };
    Ok(CatmullRomMotionMeta {
        knot_count,
        include_lods,
        sample_times,
        time_sampling,
        sample_time_grid,
        has_time_sampling_field: time_sampling_field.is_some(),
        periodic_flag,
        motion_teacher,
        volume_res: parsed.int_field("volume_res").map(|v| v as usize),
        volume_key_count: parsed.int_field("volume_key_count").map(|v| v as usize),
        source_knot_count: parsed.int_field("source_knot_count").map(|v| v as usize),
        exported_knot_count: parsed.int_field("exported_knot_count").map(|v| v as usize),
        loop_closure_knots: parsed.int_field("loop_closure_knots").map(|v| v as usize),
        loop_closure_method: parsed.string_field("loop_closure_method"),
        source_frame_count: parsed.int_field("source_frame_count").map(|v| v as usize),
        source_fps: parsed.float_field("source_fps"),
        duration_seconds: parsed.float_field("duration_seconds"),
    })
}

pub fn classify_catmull_rom_sample_times(sample_times: &[f32]) -> CatmullRomSampleTimeGrid {
    const EPS: f32 = 1.0e-5;
    let knot_count = sample_times.len();
    if knot_count < 2 {
        return CatmullRomSampleTimeGrid::Irregular;
    }
    let periodic = sample_times.iter().enumerate().all(|(i, &t)| {
        let expected = i as f32 / knot_count as f32;
        (t - expected).abs() <= EPS
    });
    if periodic {
        return CatmullRomSampleTimeGrid::Periodic;
    }
    let denom = (knot_count - 1) as f32;
    let volume_keys = sample_times.iter().enumerate().all(|(i, &t)| {
        let expected = i as f32 / denom;
        (t - expected).abs() <= EPS
    });
    if volume_keys {
        CatmullRomSampleTimeGrid::VolumeKeys
    } else {
        CatmullRomSampleTimeGrid::Irregular
    }
}

pub fn parse_catmull_rom_tile_pt(
    bytes: &[u8],
    expected_knot_count: usize,
) -> Result<CatmullRomTileMotion> {
    let parsed = ParsedPt::from_bytes(bytes)?;
    validate_common_fields(&parsed)?;
    let tile_index = parsed
        .int_field("tile_index")
        .context("missing tile_index")? as usize;
    let lod_index = parsed.int_field("lod_index").context("missing lod_index")? as usize;
    let sample_times = parsed.storage_f32(1)?;
    if sample_times.len() != expected_knot_count {
        bail!(
            "sample_times length {} != expected knot_count {}",
            sample_times.len(),
            expected_knot_count
        );
    }
    let knots = parsed.storage_f32(0)?;
    let values_per_splat = expected_knot_count
        .checked_mul(3)
        .ok_or_else(|| anyhow!("knot_count overflow"))?;
    if knots.len() % values_per_splat != 0 {
        bail!(
            "knots value count {} is not divisible by knot_count*3 ({})",
            knots.len(),
            values_per_splat
        );
    }
    let splat_count = knots.len() / values_per_splat;
    Ok(CatmullRomTileMotion {
        tile_index,
        lod_index,
        splat_count,
        knots,
    })
}

fn validate_common_fields(parsed: &ParsedPt) -> Result<()> {
    if parsed.string_field("format").as_deref() != Some(CATMULL_ROM_FORMAT) {
        bail!("unsupported Catmull-Rom motion format");
    }
    if parsed.int_field("format_version") != Some(1) {
        bail!("unsupported Catmull-Rom motion format_version");
    }
    if parsed.string_field("delta_field").as_deref() != Some("delta_xyz") {
        bail!("unsupported Catmull-Rom delta field");
    }
    if parsed.bool_field("periodic").is_none() {
        bail!("Catmull-Rom motion must declare periodic");
    }
    Ok(())
}

fn read_zip_entry<R: Read + std::io::Seek>(
    archive: &mut ZipArchive<R>,
    index: usize,
    label: &str,
) -> Result<Vec<u8>> {
    let mut file = archive
        .by_index(index)
        .with_context(|| format!("failed to open zip entry {}", label))?;
    let mut bytes = vec![0_u8; file.size() as usize];
    file.read_exact(bytes.as_mut_slice())
        .with_context(|| format!("failed to read zip entry {}", label))?;
    Ok(bytes)
}

struct ParsedPt {
    pkl: Vec<u8>,
    storages: Vec<Vec<u8>>,
}

impl ParsedPt {
    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let cursor = Cursor::new(bytes);
        let mut archive = ZipArchive::new(cursor).context("failed to open nested torch .pt zip")?;
        let mut pkl = None;
        let mut storages_by_id: HashMap<usize, Vec<u8>> = HashMap::new();
        let storage_re = Regex::new(r"(?:^|/)data/(\d+)$").unwrap();
        for i in 0..archive.len() {
            let mut file = archive.by_index(i)?;
            let name = file.name().to_string();
            if name.ends_with("data.pkl") {
                let mut bytes = vec![0_u8; file.size() as usize];
                file.read_exact(bytes.as_mut_slice())?;
                pkl = Some(bytes);
            } else if let Some(caps) = storage_re.captures(&name) {
                let id = caps[1].parse::<usize>()?;
                let mut bytes = vec![0_u8; file.size() as usize];
                file.read_exact(bytes.as_mut_slice())?;
                storages_by_id.insert(id, bytes);
            }
        }
        let pkl = pkl.context("torch .pt data.pkl not found")?;
        let mut storages = Vec::new();
        for id in 0..storages_by_id.len() {
            storages.push(
                storages_by_id
                    .remove(&id)
                    .with_context(|| format!("torch .pt storage data/{} not found", id))?,
            );
        }
        Ok(Self { pkl, storages })
    }

    fn storage_f32(&self, id: usize) -> Result<Vec<f32>> {
        let bytes = self
            .storages
            .get(id)
            .with_context(|| format!("storage data/{} missing", id))?;
        if bytes.len() % 4 != 0 {
            bail!("storage data/{} byte count is not divisible by 4", id);
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect())
    }

    fn string_field(&self, key: &str) -> Option<String> {
        let mut pos = self.after_key(key)?;
        skip_binput(&self.pkl, &mut pos);
        read_binunicode(&self.pkl, &mut pos)
    }

    fn int_field(&self, key: &str) -> Option<i64> {
        let mut pos = self.after_key(key)?;
        skip_binput(&self.pkl, &mut pos);
        read_pickle_int(&self.pkl, &mut pos)
    }

    fn bool_field(&self, key: &str) -> Option<bool> {
        let mut pos = self.after_key(key)?;
        skip_binput(&self.pkl, &mut pos);
        read_pickle_bool(&self.pkl, &mut pos).or_else(|| self.memoized_bool_field(key))
    }

    fn memoized_bool_field(&self, key: &str) -> Option<bool> {
        for memo_id in binunicode_memo_ids(&self.pkl, key) {
            let mut pos = 0;
            while pos < self.pkl.len() {
                match self.pkl[pos] {
                    b'h' if self.pkl.get(pos + 1).copied() == Some(memo_id as u8) => {
                        pos += 2;
                        if let Some(value) = read_pickle_bool(&self.pkl, &mut pos) {
                            return Some(value);
                        }
                    }
                    b'j' if self.pkl.get(pos + 1..pos + 5).and_then(|bytes| {
                        let raw: [u8; 4] = bytes.try_into().ok()?;
                        Some(u32::from_le_bytes(raw) as usize == memo_id)
                    }) == Some(true) =>
                    {
                        pos += 5;
                        if let Some(value) = read_pickle_bool(&self.pkl, &mut pos) {
                            return Some(value);
                        }
                    }
                    _ => pos += 1,
                }
            }
        }
        None
    }

    fn float_field(&self, key: &str) -> Option<f32> {
        let mut pos = self.after_key(key)?;
        skip_binput(&self.pkl, &mut pos);
        read_pickle_float(&self.pkl, &mut pos)
    }

    fn int_tuple_field(&self, key: &str) -> Option<Vec<i64>> {
        let mut pos = self.after_key(key)?;
        skip_binput(&self.pkl, &mut pos);
        if self.pkl.get(pos).copied()? != b'(' {
            return None;
        }
        pos += 1;
        let mut values = Vec::new();
        while pos < self.pkl.len() {
            if self.pkl[pos] == b't' {
                return Some(values);
            }
            values.push(read_pickle_int(&self.pkl, &mut pos)?);
        }
        None
    }

    fn after_key(&self, key: &str) -> Option<usize> {
        find_binunicode(&self.pkl, key).map(|pos| pos + 5 + key.len())
    }
}

fn read_pickle_bool(data: &[u8], pos: &mut usize) -> Option<bool> {
    match data.get(*pos).copied()? {
        0x88 => Some(true),
        0x89 => Some(false),
        _ => None,
    }
}

fn find_binunicode(data: &[u8], value: &str) -> Option<usize> {
    let bytes = value.as_bytes();
    if bytes.len() > u32::MAX as usize {
        return None;
    }
    let len = (bytes.len() as u32).to_le_bytes();
    data.windows(5 + bytes.len())
        .position(|window| window[0] == b'X' && window[1..5] == len && window[5..] == *bytes)
}

fn binunicode_memo_ids(data: &[u8], value: &str) -> Vec<usize> {
    let bytes = value.as_bytes();
    if bytes.len() > u32::MAX as usize {
        return Vec::new();
    }
    let len = (bytes.len() as u32).to_le_bytes();
    let mut ids = Vec::new();
    let mut pos = 0;
    while pos + 5 + bytes.len() < data.len() {
        let string_start = pos + 5;
        let string_end = string_start + bytes.len();
        if data[pos] == b'X'
            && data[pos + 1..pos + 5] == len
            && data[string_start..string_end] == *bytes
        {
            match data.get(string_end).copied() {
                Some(b'q') => {
                    if let Some(id) = data.get(string_end + 1).copied() {
                        ids.push(id as usize);
                    }
                }
                Some(b'r') => {
                    if let Some(raw) = data.get(string_end + 1..string_end + 5) {
                        if let Ok(bytes) = raw.try_into() {
                            ids.push(u32::from_le_bytes(bytes) as usize);
                        }
                    }
                }
                _ => {}
            }
        }
        pos += 1;
    }
    ids
}

fn read_binunicode(data: &[u8], pos: &mut usize) -> Option<String> {
    if data.get(*pos).copied()? != b'X' {
        return None;
    }
    *pos += 1;
    let len = u32::from_le_bytes(data.get(*pos..*pos + 4)?.try_into().ok()?) as usize;
    *pos += 4;
    let bytes = data.get(*pos..*pos + len)?;
    *pos += len;
    String::from_utf8(bytes.to_vec()).ok()
}

fn read_pickle_int(data: &[u8], pos: &mut usize) -> Option<i64> {
    match data.get(*pos).copied()? {
        b'K' => {
            *pos += 1;
            let v = *data.get(*pos)? as i64;
            *pos += 1;
            Some(v)
        }
        b'M' => {
            *pos += 1;
            let v = u16::from_le_bytes(data.get(*pos..*pos + 2)?.try_into().ok()?) as i64;
            *pos += 2;
            Some(v)
        }
        b'J' => {
            *pos += 1;
            let v = i32::from_le_bytes(data.get(*pos..*pos + 4)?.try_into().ok()?) as i64;
            *pos += 4;
            Some(v)
        }
        _ => None,
    }
}

fn read_pickle_float(data: &[u8], pos: &mut usize) -> Option<f32> {
    match data.get(*pos).copied()? {
        b'G' => {
            *pos += 1;
            let bytes: [u8; 8] = data.get(*pos..*pos + 8)?.try_into().ok()?;
            *pos += 8;
            Some(f64::from_be_bytes(bytes) as f32)
        }
        b'F' => {
            *pos += 1;
            let start = *pos;
            while *pos < data.len() && data[*pos] != b'\n' {
                *pos += 1;
            }
            let raw = std::str::from_utf8(data.get(start..*pos)?).ok()?;
            if *pos < data.len() {
                *pos += 1;
            }
            raw.parse::<f32>().ok()
        }
        _ => read_pickle_int(data, pos).map(|v| v as f32),
    }
}

fn skip_binput(data: &[u8], pos: &mut usize) {
    if matches!(data.get(*pos), Some(b'q')) {
        *pos += 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deformation::DeformationNetwork;

    #[test]
    fn catmull_rom_evaluator_wraps_and_hits_knots() {
        let knots = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        assert_eq!(
            catmull_rom_delta(&knots, 0.0),
            catmull_rom_delta(&knots, 1.0)
        );
        for i in 0..knots.len() {
            assert_eq!(
                catmull_rom_delta(&knots, i as f32 / knots.len() as f32),
                knots[i]
            );
        }
    }

    #[test]
    fn catmull_rom_volume_key_evaluator_hits_volume_key_times() {
        let knots = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ];
        for i in 0..knots.len() {
            assert_eq!(
                catmull_rom_delta_volume_keys(&knots, i as f32 / (knots.len() - 1) as f32),
                knots[i]
            );
        }
    }

    #[test]
    fn sample_time_classifier_distinguishes_periodic_and_volume_keys() {
        let periodic = (0..25).map(|i| i as f32 / 25.0).collect::<Vec<_>>();
        let volume_keys = (0..25).map(|i| i as f32 / 24.0).collect::<Vec<_>>();
        let irregular = vec![0.0, 0.2, 0.5, 0.9];

        assert_eq!(
            classify_catmull_rom_sample_times(periodic.as_slice()),
            CatmullRomSampleTimeGrid::Periodic
        );
        assert_eq!(
            classify_catmull_rom_sample_times(volume_keys.as_slice()),
            CatmullRomSampleTimeGrid::VolumeKeys
        );
        assert_eq!(
            classify_catmull_rom_sample_times(irregular.as_slice()),
            CatmullRomSampleTimeGrid::Irregular
        );
    }

    #[test]
    fn motion_teacher_labels_are_stable_for_asset_logs() {
        assert_eq!(CatmullRomMotionTeacher::Legacy.as_str(), "legacy");
        assert_eq!(CatmullRomMotionTeacher::Volume.as_str(), "volume");
        assert_eq!(
            CatmullRomMotionTeacher::DirectNetwork.as_str(),
            "direct_network"
        );
    }

    #[test]
    fn detects_motion_filenames() {
        assert_eq!(
            detect_motion_file("tile12_lod3_motion_catmull_rom.pt"),
            Some((12, 3))
        );
        assert_eq!(detect_motion_file("tile12_lod3.ply"), None);
    }

    #[test]
    fn reads_pickle_binfloat() {
        let mut bytes = vec![b'G'];
        bytes.extend_from_slice(&2.5_f64.to_be_bytes());
        let mut pos = 0;
        assert_eq!(read_pickle_float(bytes.as_slice(), &mut pos), Some(2.5));
        assert_eq!(pos, 9);
    }

    #[test]
    fn bool_field_reads_memoized_string_key() {
        let mut pkl = Vec::new();
        pkl.extend_from_slice(b"\x80\x02}q\x00(");
        pkl.extend_from_slice(b"X\r\0\0\0time_samplingq\x01");
        pkl.extend_from_slice(b"X\x08\0\0\0periodicq\x02");
        pkl.extend_from_slice(b"h\x02\x88u.");
        let parsed = ParsedPt {
            pkl,
            storages: Vec::new(),
        };

        assert_eq!(
            parsed.string_field("time_sampling").as_deref(),
            Some("periodic")
        );
        assert_eq!(parsed.bool_field("periodic"), Some(true));
    }

    #[test]
    fn packs_tile_knots_using_renderer_source_row_permutation() {
        let tile_motion = CatmullRomTileMotion {
            tile_index: 0,
            lod_index: 0,
            splat_count: 3,
            knots: vec![
                0.0, 0.1, 0.2, 10.0, 10.1, 10.2, 20.0, 20.1, 20.2, 1.0, 1.1, 1.2, 11.0, 11.1, 11.2,
                21.0, 21.1, 21.2,
            ],
        };
        let mut packed = Vec::new();

        append_tile_knots_splat_major(&mut packed, &tile_motion, 2, &[2, 0, 1]).unwrap();

        assert_eq!(
            packed,
            vec![
                20.0, 20.1, 20.2, 21.0, 21.1, 21.2, 0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 10.0, 10.1, 10.2,
                11.0, 11.1, 11.2,
            ]
        );
    }

    #[test]
    fn packs_tile_knots_identity_permutation_without_change() {
        let tile_motion = CatmullRomTileMotion {
            tile_index: 0,
            lod_index: 0,
            splat_count: 2,
            knots: vec![
                0.0, 0.1, 0.2, 10.0, 10.1, 10.2, 1.0, 1.1, 1.2, 11.0, 11.1, 11.2,
            ],
        };
        let mut packed = Vec::new();

        append_tile_knots_splat_major(&mut packed, &tile_motion, 2, &[0, 1]).unwrap();

        assert_eq!(
            packed,
            vec![0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 10.0, 10.1, 10.2, 11.0, 11.1, 11.2]
        );
    }

    #[test]
    #[ignore = "set GSWT_TEST_CATMULL_ROM_ZIP to a constructor gswt.zip path"]
    fn parses_real_constructor_zip_when_available() {
        let path = std::env::var("GSWT_TEST_CATMULL_ROM_ZIP")
            .expect("GSWT_TEST_CATMULL_ROM_ZIP must point to a constructor gswt.zip");
        let file = std::fs::File::open(path).expect("failed to open constructor gswt.zip");
        let mut archive = ZipArchive::new(file).expect("failed to open zip");
        let mut meta_index = None;
        let mut first_motion_index = None;
        let mut motion_files = 0_usize;
        for i in 0..archive.len() {
            let name = archive.by_index(i).unwrap().name().to_ascii_lowercase();
            let basename = name.rsplit('/').next().unwrap_or(name.as_str());
            if basename == CATMULL_ROM_META_FILENAME {
                meta_index = Some(i);
            } else if detect_motion_file(basename).is_some() {
                first_motion_index.get_or_insert(i);
                motion_files += 1;
            }
        }
        let meta_index = meta_index.expect("motion_catmull_rom_meta.pt not found");
        let meta_bytes = read_zip_entry(&mut archive, meta_index, CATMULL_ROM_META_FILENAME)
            .expect("failed to read meta");
        let meta = parse_catmull_rom_meta_pt(meta_bytes.as_slice()).expect("failed to parse meta");
        println!(
            "Catmull-Rom meta audit: knots={}, time_sampling={}, field_present={}, sample_time_grid={}, periodic={}, motion_teacher={}, volume_res={:?}, volume_key_count={:?}, first_time={}, last_time={}, source_frames={:?}, duration={:?}",
            meta.knot_count,
            meta.time_sampling.as_str(),
            meta.has_time_sampling_field,
            meta.sample_time_grid.as_str(),
            meta.periodic_flag,
            meta.motion_teacher.as_str(),
            meta.volume_res,
            meta.volume_key_count,
            meta.sample_times.first().copied().unwrap_or(0.0),
            meta.sample_times.last().copied().unwrap_or(0.0),
            meta.source_frame_count,
            meta.duration_seconds
        );
        assert!(meta.knot_count >= 4);
        assert_eq!(meta.sample_times.len(), meta.knot_count);
        assert!(motion_files > 0);
        let motion_index = first_motion_index.expect("no tile motion file found");
        let motion_bytes = read_zip_entry(&mut archive, motion_index, "tile motion")
            .expect("failed to read tile motion");
        let tile_motion = parse_catmull_rom_tile_pt(motion_bytes.as_slice(), meta.knot_count)
            .expect("failed to parse tile motion");
        assert_eq!(
            tile_motion.knots.len(),
            meta.knot_count * tile_motion.splat_count * 3
        );
    }

    #[test]
    #[ignore = "set GSWT_TEST_CATMULL_ROM_ZIP to a constructor gswt.zip path"]
    fn audits_real_spline_knots_against_direct_network_when_available() {
        let path = std::env::var("GSWT_TEST_CATMULL_ROM_ZIP")
            .expect("GSWT_TEST_CATMULL_ROM_ZIP must point to a constructor gswt.zip");
        let sample_limit = std::env::var("GSWT_TEST_MOTION_AUDIT_SAMPLES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(2048);
        let (scene_vec, deformation_weights, motion) =
            load_real_constructor_zip_for_motion_audit(path.as_str())
                .expect("failed to load real constructor zip for motion audit");
        let network = DeformationNetwork::from_bytes(deformation_weights.as_slice())
            .expect("failed to parse deformation_weights.bin");
        let scale_factor = network.metadata().scale_factor;
        let mut global_orig_means = Vec::with_capacity(motion.total_splats);
        for lod_vec in &scene_vec {
            for scene in lod_vec {
                let orig_means = scene
                    .orig_means
                    .as_ref()
                    .expect("motion audit requires PLY orig_means");
                global_orig_means.extend_from_slice(orig_means.as_slice());
            }
        }
        assert_eq!(global_orig_means.len(), motion.total_splats);

        let stride = (motion.total_splats / sample_limit.max(1)).max(1);
        let mut compared = 0_u64;
        let mut sum_error = 0.0_f64;
        let mut sum_error_sq = 0.0_f64;
        let mut max_error = 0.0_f32;
        let mut worst_splat = 0_usize;
        let mut worst_knot = 0_usize;
        for splat in (0..motion.total_splats).step_by(stride).take(sample_limit) {
            let orig = global_orig_means[splat];
            for (knot, &time) in motion.meta.sample_times.iter().enumerate() {
                let (dx, _) = network
                    .deform_delta_single(orig, time)
                    .expect("direct network deformation failed");
                let expected = [
                    dx[0] * scale_factor,
                    dx[1] * scale_factor,
                    dx[2] * scale_factor,
                ];
                let base = (splat * motion.meta.knot_count + knot) * 3;
                let actual = [
                    motion.global_knots[base],
                    motion.global_knots[base + 1],
                    motion.global_knots[base + 2],
                ];
                let err = ((actual[0] - expected[0]).powi(2)
                    + (actual[1] - expected[1]).powi(2)
                    + (actual[2] - expected[2]).powi(2))
                .sqrt();
                compared += 1;
                sum_error += err as f64;
                sum_error_sq += (err * err) as f64;
                if err > max_error {
                    max_error = err;
                    worst_splat = splat;
                    worst_knot = knot;
                }
            }
        }
        let denom = compared.max(1) as f64;
        println!(
            "Direct network audit: compared={}, mean_error={:.8}, rms_error={:.8}, max_error={:.8}, worst_knot={}, worst_splat={}, time_sampling={}, sample_time_grid={}",
            compared,
            sum_error / denom,
            (sum_error_sq / denom).sqrt(),
            max_error,
            worst_knot,
            worst_splat,
            motion.meta.time_sampling.as_str(),
            motion.meta.sample_time_grid.as_str()
        );
        assert!(compared > 0);
    }

    fn load_real_constructor_zip_for_motion_audit(
        path: &str,
    ) -> Result<(Vec<Vec<Scene>>, Vec<u8>, Arc<CatmullRomMotionSet>)> {
        struct SceneFileEntry {
            index: usize,
            filename: String,
            lod_id: usize,
            tile_id: usize,
        }

        let file = std::fs::File::open(path)
            .with_context(|| format!("failed to open constructor gswt.zip '{}'", path))?;
        let mut archive = ZipArchive::new(file).context("failed to open constructor gswt.zip")?;
        let re = Regex::new(r"tile(\d+)_lod(\d+)").unwrap();
        let mut file_vec = Vec::new();
        let mut deformation_weights_index = None;
        let mut catmull_rom_meta_index = None;
        let mut catmull_rom_entries = Vec::new();
        for i in 0..archive.len() {
            let file = archive.by_index(i)?;
            let name = file.name().to_string();
            let basename = name
                .rsplit('/')
                .next()
                .unwrap_or(name.as_str())
                .to_ascii_lowercase();
            if basename == "deformation_weights.bin" {
                deformation_weights_index = Some(i);
                continue;
            }
            if basename == CATMULL_ROM_META_FILENAME {
                catmull_rom_meta_index = Some(i);
                continue;
            }
            if let Some((tile_id, lod_id)) = detect_motion_file(basename.as_str()) {
                catmull_rom_entries.push(CatmullRomMotionZipEntry {
                    index: i,
                    filename: name,
                    tile_id,
                    lod_id,
                });
                continue;
            }
            if !(basename.ends_with(".ply") || basename.ends_with(".splat")) {
                continue;
            }
            if let Some(caps) = re.captures(basename.as_str()) {
                file_vec.push(SceneFileEntry {
                    index: i,
                    filename: name,
                    tile_id: caps[1].parse()?,
                    lod_id: caps[2].parse()?,
                });
            }
        }

        let deformation_weights_index =
            deformation_weights_index.context("deformation_weights.bin not found")?;
        let deformation_weights = read_zip_entry(
            &mut archive,
            deformation_weights_index,
            "deformation_weights.bin",
        )?;
        let meta_index = catmull_rom_meta_index.context("motion_catmull_rom_meta.pt not found")?;

        file_vec.sort_by_key(|e| (e.lod_id, e.tile_id));
        let first_entry = file_vec.first().context("no tile files found")?;
        let last_entry = file_vec.last().context("no tile files found")?;
        let n_lod = last_entry.lod_id - first_entry.lod_id + 1;
        let n_tile = last_entry.tile_id + 1;
        let mut scene_vec = Vec::with_capacity(n_lod);
        for lod_id in 0..n_lod {
            let mut lod_vec = Vec::with_capacity(n_tile);
            for tile_id in 0..n_tile {
                let file_entry = file_vec
                    .iter()
                    .find(|entry| entry.lod_id == lod_id && entry.tile_id == tile_id)
                    .with_context(|| format!("missing tile{}_lod{} PLY", tile_id, lod_id))?;
                if !file_entry.filename.to_ascii_lowercase().ends_with(".ply") {
                    bail!("motion audit requires PLY tile files with orig_means");
                }
                let bytes = read_zip_entry(&mut archive, file_entry.index, &file_entry.filename)?;
                let (header, mut cursor) =
                    Scene::parse_file_header(bytes).map_err(anyhow::Error::msg)?;
                let mut scene = Scene::new();
                scene.splat_count = header.splat_count;
                scene
                    .load(&mut cursor, &header)
                    .map_err(anyhow::Error::msg)?;
                lod_vec.push(scene);
            }
            scene_vec.push(lod_vec);
        }
        let motion = load_catmull_rom_motion_from_zip(
            &mut archive,
            meta_index,
            catmull_rom_entries.as_slice(),
            scene_vec.as_slice(),
        )?
        .context("Catmull-Rom motion failed validation")?;
        Ok((scene_vec, deformation_weights, motion))
    }
}
