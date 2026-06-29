use std::collections::HashMap;
use std::io::Read;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use regex::Regex;
use serde::Deserialize;
use zip::ZipArchive;

use crate::basis_motion_graph::{
    BasisMotionGraph, BasisMotionGraphBranch, parse_basis_motion_graph,
};
use crate::log;
use crate::scene::Scene;

pub const BASIS_BANK_META_FILENAME: &str = "motion_basis_meta.bin";
pub const BASIS_BANK_FORMAT: &str = "loop_closed_catmull_rom_basis_bank_delta_xyz";
const BASIS_BANK_VERSION: u32 = 2;
const BASIS_BANK_LEGACY_VERSION: u32 = 1;
const BASIS_LOD_MAGIC: &[u8; 4] = b"MBSB";
const BASIS_COEFFS_MAGIC: &[u8; 4] = b"MBCF";
const DEFORMATION_HEADER_SIZE: usize = 28;
const DEFORMATION_TEMPORAL_RESOLUTION_OFFSET: usize = 24;
const SOURCE_FRAME_RATE: f32 = 30.0;
const FRAMES_PER_TEMPORAL_GRID_SAMPLE: f32 = 2.0;

#[derive(Debug, Clone)]
pub struct BasisBankMotionMeta {
    pub format_version: u32,
    pub include_lods: Vec<usize>,
    pub source_knot_count: usize,
    pub exported_knot_count: usize,
    pub loop_closure_knots: usize,
    pub basis_count: usize,
    pub top_k: usize,
    pub duration_seconds: f32,
}

#[derive(Debug, Deserialize)]
struct BasisBankMotionWireMeta {
    format: String,
    format_version: u32,
    delta_field: String,
    basis_scope: String,
    #[serde(default)]
    basis_source_lod: Option<usize>,
    include_lods: Vec<usize>,
    source_knot_count: usize,
    exported_knot_count: usize,
    loop_closure_knots: usize,
    loop_closure_method: String,
    motion_teacher: String,
    source_time_sampling: String,
    motion_basis_closure_mode: String,
    basis_count: usize,
    top_k: usize,
    #[serde(default)]
    duration_seconds: Option<f32>,
}
#[derive(Debug, Deserialize)]
struct BasisBankVersionWire {
    format_version: u32,
}

#[derive(Debug, Clone)]
pub struct BasisBankLodMotion {
    pub format_version: u32,
    pub lod_index: usize,
    pub basis_count: usize,
    pub knot_count: usize,
    /// Basis-major storage: ((basis * knot_count + knot) * 3 + xyz).
    pub basis_knots: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct BasisBankTileCoefficients {
    pub format_version: u32,
    pub tile_index: usize,
    pub lod_index: usize,
    pub splat_count: usize,
    pub top_k: usize,
    /// Splat-major storage: splat * top_k + slot.
    pub basis_ids: Vec<u32>,
    /// Splat-major storage: splat * top_k + slot.
    pub weights: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct BasisBankMotionSet {
    pub meta: BasisBankMotionMeta,
    pub motion_graph: Option<Arc<BasisMotionGraph>>,
    pub total_splats: usize,
    pub basis_count: usize,
    pub usage_stats: Vec<BasisUsageStats>,
    /// Global basis-major storage: ((basis * knot_count + knot) * 3 + xyz).
    pub basis_knots: Vec<f32>,
    /// Global splat-major sparse IDs: splat * top_k + slot.
    pub basis_ids: Vec<u32>,
    /// Global splat-major sparse weights: splat * top_k + slot.
    pub global_weights: Vec<f32>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BasisUsageStats {
    pub affected_splats: u32,
    pub sum_abs_weight: f32,
    pub max_abs_weight: f32,
    pub mean_abs_weight: f32,
}

pub const BASIS_SCOPE_SHARED_LOD0: &str = "shared_lod0";

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BasisSegmentKinematics {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub acceleration: [f32; 3],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BasisBranchContinuityDebug {
    pub source_basis_id: usize,
    pub target_basis_id: usize,
    pub source_segment: usize,
    pub target_segment: usize,
    pub position_delta: [f32; 3],
    pub velocity_delta: [f32; 3],
    pub acceleration_delta: [f32; 3],
    pub position_norm: f32,
    pub velocity_norm: f32,
    pub acceleration_norm: f32,
}

#[derive(Debug, Clone)]
pub struct BasisBankLodZipEntry {
    pub index: usize,
    pub filename: String,
    pub lod_id: usize,
}

#[derive(Debug, Clone)]
pub struct BasisBankCoeffZipEntry {
    pub index: usize,
    pub filename: String,
    pub tile_id: usize,
    pub lod_id: usize,
}

pub fn detect_basis_lod_file(filename: &str) -> Option<usize> {
    let re = Regex::new(r"^lod(\d+)_motion_basis\.bin$").ok()?;
    let caps = re.captures(filename)?;
    caps.get(1)?.as_str().parse().ok()
}

pub fn detect_basis_coeffs_file(filename: &str) -> Option<(usize, usize)> {
    let re = Regex::new(r"^tile(\d+)_lod(\d+)_motion_basis_coeffs\.bin$").ok()?;
    let caps = re.captures(filename)?;
    let tile_id = caps.get(1)?.as_str().parse().ok()?;
    let lod_id = caps.get(2)?.as_str().parse().ok()?;
    Some((tile_id, lod_id))
}

pub fn parse_basis_bank_meta(bytes: &[u8]) -> Result<BasisBankMotionMeta> {
    parse_basis_bank_meta_with_legacy_duration(bytes, None)
}

fn parse_basis_bank_meta_with_legacy_duration(
    bytes: &[u8],
    legacy_duration_seconds: Option<f32>,
) -> Result<BasisBankMotionMeta> {
    let wire: BasisBankMotionWireMeta =
        serde_json::from_slice(bytes).context("failed to parse basis-bank metadata JSON")?;
    if wire.format != BASIS_BANK_FORMAT {
        bail!("unsupported basis-bank format '{}'", wire.format);
    }
    if !matches!(
        wire.format_version,
        BASIS_BANK_LEGACY_VERSION | BASIS_BANK_VERSION
    ) {
        bail!(
            "unsupported basis-bank version {}; expected {} or {}",
            wire.format_version,
            BASIS_BANK_LEGACY_VERSION,
            BASIS_BANK_VERSION
        );
    }
    if wire.delta_field != "delta_xyz" {
        bail!("unsupported basis-bank delta field '{}'", wire.delta_field);
    }
    if wire.basis_scope != BASIS_SCOPE_SHARED_LOD0 {
        bail!(
            "unsupported basis-bank scope '{}'; expected shared_lod0",
            wire.basis_scope
        );
    }
    if wire.basis_scope == BASIS_SCOPE_SHARED_LOD0 {
        if wire.basis_source_lod != Some(0) {
            bail!(
                "shared_lod0 basis-bank scope requires basis_source_lod=0, got {:?}",
                wire.basis_source_lod
            );
        }
        if !wire.include_lods.contains(&0) {
            bail!("shared_lod0 basis-bank scope requires LoD 0 in include_lods");
        }
    }
    if wire.motion_teacher != "direct_network" {
        bail!(
            "unsupported motion_teacher '{}'; expected 'direct_network'",
            wire.motion_teacher
        );
    }
    if wire.source_time_sampling != "inclusive" {
        bail!(
            "unsupported source_time_sampling '{}'; expected 'inclusive'",
            wire.source_time_sampling
        );
    }
    if wire.loop_closure_method != "cubic_hermite" {
        bail!(
            "unsupported loop_closure_method '{}'; expected 'cubic_hermite'",
            wire.loop_closure_method
        );
    }
    if !matches!(
        wire.motion_basis_closure_mode.as_str(),
        "target_then_basis" | "basis_then_closure"
    ) {
        bail!(
            "unsupported motion_basis_closure_mode '{}'; expected 'target_then_basis' or 'basis_then_closure'",
            wire.motion_basis_closure_mode
        );
    }
    if wire.source_knot_count == 0 {
        bail!("basis-bank source_knot_count must be nonzero");
    }
    let expected_exported_knot_count = wire
        .source_knot_count
        .checked_add(wire.loop_closure_knots)
        .context("basis-bank source_knot_count + loop_closure_knots overflow")?;
    if expected_exported_knot_count != wire.exported_knot_count {
        bail!(
            "basis-bank source_knot_count {} + loop_closure_knots {} must equal exported_knot_count {}, expected {}",
            wire.source_knot_count,
            wire.loop_closure_knots,
            wire.exported_knot_count,
            expected_exported_knot_count
        );
    }
    let duration_seconds = match wire.format_version {
        BASIS_BANK_LEGACY_VERSION => legacy_duration_seconds.context(
            "basis-bank version 1 requires deformation_weights.bin to infer playback duration",
        )?,
        BASIS_BANK_VERSION => wire.duration_seconds.context(
            "basis-bank metadata is missing duration_seconds; rebuild with the current gswt_constructor",
        )?,
        _ => unreachable!(),
    };
    if !duration_seconds.is_finite() || duration_seconds <= 0.0 {
        bail!("basis-bank duration_seconds must be finite and positive, got {duration_seconds}");
    }
    if wire.exported_knot_count < 4 {
        bail!(
            "basis-bank exported knot count must be at least 4, got {}",
            wire.exported_knot_count
        );
    }
    if wire.basis_count == 0 || wire.top_k == 0 || wire.top_k > wire.basis_count {
        bail!(
            "invalid basis-bank basis_count/top_k: basis_count={}, top_k={}",
            wire.basis_count,
            wire.top_k
        );
    }
    Ok(BasisBankMotionMeta {
        format_version: wire.format_version,
        include_lods: wire.include_lods,
        source_knot_count: wire.source_knot_count,
        exported_knot_count: wire.exported_knot_count,
        loop_closure_knots: wire.loop_closure_knots,
        basis_count: wire.basis_count,
        top_k: wire.top_k,
        duration_seconds,
    })
}

pub fn parse_basis_lod_motion(bytes: &[u8]) -> Result<BasisBankLodMotion> {
    const HEADER_SIZE: usize = 20;
    if bytes.len() < HEADER_SIZE {
        bail!("basis LOD payload is too small");
    }
    if &bytes[0..4] != BASIS_LOD_MAGIC {
        bail!("bad basis LOD magic");
    }
    let version = read_u32(bytes, 4)?;
    if !matches!(version, BASIS_BANK_LEGACY_VERSION | BASIS_BANK_VERSION) {
        bail!("unsupported basis LOD version {}", version);
    }
    let lod_index = read_u32(bytes, 8)? as usize;
    let basis_count = read_u32(bytes, 12)? as usize;
    let knot_count = read_u32(bytes, 16)? as usize;
    let value_count = basis_count
        .checked_mul(knot_count)
        .and_then(|v| v.checked_mul(3))
        .context("basis LOD value count overflow")?;
    let payload_bytes = value_count
        .checked_mul(4)
        .context("basis LOD byte length overflow")?;
    let expected = HEADER_SIZE
        .checked_add(payload_bytes)
        .context("basis LOD byte length overflow")?;
    if bytes.len() != expected {
        bail!(
            "basis LOD byte length mismatch: got {}, expected {}",
            bytes.len(),
            expected
        );
    }
    let basis_knots = read_f32_vec(&bytes[HEADER_SIZE..])?;
    if basis_knots.iter().any(|value| !value.is_finite()) {
        bail!("basis LOD payload contains non-finite knot values");
    }
    Ok(BasisBankLodMotion {
        format_version: version,
        lod_index,
        basis_count,
        knot_count,
        basis_knots,
    })
}

pub fn parse_basis_tile_coefficients(bytes: &[u8]) -> Result<BasisBankTileCoefficients> {
    const HEADER_SIZE: usize = 24;
    if bytes.len() < HEADER_SIZE {
        bail!("basis coefficient payload is too small");
    }
    if &bytes[0..4] != BASIS_COEFFS_MAGIC {
        bail!("bad basis coefficient magic");
    }
    let version = read_u32(bytes, 4)?;
    if !matches!(version, BASIS_BANK_LEGACY_VERSION | BASIS_BANK_VERSION) {
        bail!("unsupported basis coefficient version {}", version);
    }
    let tile_index = read_u32(bytes, 8)? as usize;
    let lod_index = read_u32(bytes, 12)? as usize;
    let splat_count = read_u32(bytes, 16)? as usize;
    let top_k = read_u32(bytes, 20)? as usize;
    let count = splat_count
        .checked_mul(top_k)
        .context("basis coefficient count overflow")?;
    let ids_start = HEADER_SIZE;
    let section_bytes = count
        .checked_mul(4)
        .context("basis coefficient byte length overflow")?;
    let weights_start = ids_start
        .checked_add(section_bytes)
        .context("basis coefficient byte length overflow")?;
    let expected = weights_start
        .checked_add(section_bytes)
        .context("basis coefficient byte length overflow")?;
    if bytes.len() != expected {
        bail!(
            "basis coefficient byte length mismatch: got {}, expected {}",
            bytes.len(),
            expected
        );
    }
    let mut basis_ids = Vec::with_capacity(count);
    for chunk in bytes[ids_start..weights_start].chunks_exact(4) {
        basis_ids.push(u32::from_le_bytes(chunk.try_into().unwrap()));
    }
    let weights = read_f32_vec(&bytes[weights_start..])?;
    if weights.iter().any(|value| !value.is_finite()) {
        bail!("basis coefficient payload contains non-finite weights");
    }
    Ok(BasisBankTileCoefficients {
        format_version: version,
        tile_index,
        lod_index,
        splat_count,
        top_k,
        basis_ids,
        weights,
    })
}

pub fn load_basis_bank_motion_from_zip<R: Read + std::io::Seek>(
    archive: &mut ZipArchive<R>,
    meta_index: usize,
    deformation_weights_index: Option<usize>,
    graph_index: Option<usize>,
    lod_entries: &[BasisBankLodZipEntry],
    coeff_entries: &[BasisBankCoeffZipEntry],
    scene_vec: &[Vec<Scene>],
) -> Result<Option<Arc<BasisBankMotionSet>>> {
    let meta_bytes = read_zip_entry(archive, meta_index, BASIS_BANK_META_FILENAME)?;
    let version_wire: BasisBankVersionWire = serde_json::from_slice(&meta_bytes)
        .context("failed to read basis-bank metadata version")?;
    let legacy_duration_seconds = if version_wire.format_version == BASIS_BANK_LEGACY_VERSION {
        let deformation_index = deformation_weights_index.context(
            "basis-bank version 1 requires deformation_weights.bin to infer playback duration",
        )?;
        let header = read_zip_entry_prefix(
            archive,
            deformation_index,
            "deformation_weights.bin",
            DEFORMATION_HEADER_SIZE,
        )?;
        Some(parse_legacy_deformation_duration_seconds(&header)?)
    } else {
        None
    };
    let meta = parse_basis_bank_meta_with_legacy_duration(&meta_bytes, legacy_duration_seconds)?;
    let [basis_entry] = lod_entries else {
        bail!("dynamic archive requires exactly one lod0_motion_basis.bin");
    };
    if basis_entry.lod_id != 0 || basis_entry.filename != "lod0_motion_basis.bin" {
        bail!(
            "unexpected basis payload '{}'; expected exactly lod0_motion_basis.bin",
            basis_entry.filename
        );
    }
    let basis_bytes = read_zip_entry(archive, basis_entry.index, &basis_entry.filename)?;
    let basis = parse_basis_lod_motion(&basis_bytes)
        .with_context(|| format!("failed to parse {}", basis_entry.filename))?;
    if basis.format_version != meta.format_version {
        bail!(
            "{} header version {} does not match metadata version {}",
            basis_entry.filename,
            basis.format_version,
            meta.format_version
        );
    }
    if basis.lod_index != 0 {
        bail!(
            "{} header lod_index={} does not match expected 0",
            basis_entry.filename,
            basis.lod_index
        );
    }
    if basis.basis_count != meta.basis_count {
        bail!(
            "{} header basis_count={} does not match metadata basis_count={}",
            basis_entry.filename,
            basis.basis_count,
            meta.basis_count
        );
    }
    if basis.knot_count != meta.exported_knot_count {
        bail!(
            "{} header knot_count={} does not match metadata exported_knot_count={}",
            basis_entry.filename,
            basis.knot_count,
            meta.exported_knot_count
        );
    }
    let mut coeff_by_lod_tile: HashMap<(usize, usize), &BasisBankCoeffZipEntry> = HashMap::new();
    for entry in coeff_entries {
        if coeff_by_lod_tile
            .insert((entry.lod_id, entry.tile_id), entry)
            .is_some()
        {
            bail!(
                "duplicate basis coefficient payload for tile{}_lod{}",
                entry.tile_id,
                entry.lod_id
            );
        }
    }

    let basis_knots = basis.basis_knots;
    let basis_count = meta.basis_count;
    let mut total_splats = 0_usize;
    let mut basis_ids = Vec::new();
    let mut global_weights = Vec::new();
    for (lod_id, lod_vec) in scene_vec.iter().enumerate() {
        for (tile_id, scene) in lod_vec.iter().enumerate() {
            total_splats += scene.splat_count;
            let Some(entry) = coeff_by_lod_tile.get(&(lod_id, tile_id)) else {
                bail!(
                    "missing tile{}_lod{}_motion_basis_coeffs.bin; rebuild with the current constructor",
                    tile_id,
                    lod_id
                );
            };
            let bytes = read_zip_entry(archive, entry.index, &entry.filename)?;
            let coeffs = parse_basis_tile_coefficients(&bytes)
                .with_context(|| format!("failed to parse {}", entry.filename))?;
            if coeffs.format_version != meta.format_version {
                bail!(
                    "{} header version {} does not match metadata version {}",
                    entry.filename,
                    coeffs.format_version,
                    meta.format_version
                );
            }
            if coeffs.tile_index != tile_id || coeffs.lod_index != lod_id {
                bail!(
                    "{} header identifies tile{}_lod{}; expected tile{}_lod{}",
                    entry.filename,
                    coeffs.tile_index,
                    coeffs.lod_index,
                    tile_id,
                    lod_id
                );
            }
            if coeffs.splat_count != scene.splat_count || coeffs.top_k != meta.top_k {
                bail!(
                    "{} header splat_count/top_k={}/{}; expected {}/{}",
                    entry.filename,
                    coeffs.splat_count,
                    coeffs.top_k,
                    scene.splat_count,
                    meta.top_k
                );
            }
            append_tile_coefficients_splat_major(
                &mut basis_ids,
                &mut global_weights,
                &coeffs,
                scene.source_row_indices.as_slice(),
                basis_count,
            )
            .with_context(|| format!("invalid coefficients in {}", entry.filename))?;
        }
    }
    let usage_stats =
        compute_basis_usage_stats(&basis_ids, &global_weights, basis_count, meta.top_k);
    let motion_graph = if let Some(graph_index) = graph_index {
        match read_zip_entry(
            archive,
            graph_index,
            crate::basis_motion_graph::BASIS_MOTION_GRAPH_FILENAME,
        )
        .and_then(|bytes| parse_basis_motion_graph(&bytes))
        .and_then(|graph| {
            graph.validate_against_basis_bank(&meta)?;
            Ok(graph)
        }) {
            Ok(graph) => {
                let branch_count: usize = graph.lods.iter().map(|lod| lod.branches.len()).sum();
                log!(
                    "Basis motion graph loaded: scope={}, lods={}, basis_count={}, knots={}, branch_top_k={}, branches={}",
                    graph.basis_scope,
                    graph.lods.len(),
                    graph.basis_count,
                    graph.knot_count,
                    graph.branch_top_k,
                    branch_count
                );
                Some(Arc::new(graph))
            }
            Err(err) => {
                log!(
                    "Basis motion graph is invalid; continuing without graph visualization: {}",
                    err
                );
                None
            }
        }
    } else {
        None
    };
    log!(
        "Shared-LoD0 basis motion loaded: version={}, duration_seconds={}, lods={:?}, basis_count={}, top_k={}, knots={}, source_knots={}, closure_knots={}, total_splats={}",
        meta.format_version,
        meta.duration_seconds,
        meta.include_lods,
        meta.basis_count,
        meta.top_k,
        meta.exported_knot_count,
        meta.source_knot_count,
        meta.loop_closure_knots,
        total_splats
    );
    Ok(Some(Arc::new(BasisBankMotionSet {
        meta,
        motion_graph,
        total_splats,
        basis_count,
        usage_stats,
        basis_knots,
        basis_ids,
        global_weights,
    })))
}

pub fn compute_basis_usage_stats(
    basis_ids: &[u32],
    weights: &[f32],
    basis_count: usize,
    top_k: usize,
) -> Vec<BasisUsageStats> {
    let mut stats = vec![BasisUsageStats::default(); basis_count];
    if top_k == 0 || basis_ids.len() != weights.len() {
        return stats;
    }

    for (ids, ws) in basis_ids
        .chunks_exact(top_k)
        .zip(weights.chunks_exact(top_k))
    {
        let mut per_splat_abs = vec![0.0_f32; basis_count];
        for (&basis_id, &weight) in ids.iter().zip(ws.iter()) {
            let basis_id = basis_id as usize;
            if basis_id < basis_count {
                per_splat_abs[basis_id] += weight.abs();
            }
        }
        for (basis_id, abs_weight) in per_splat_abs.into_iter().enumerate() {
            if abs_weight > 0.0 {
                let stat = &mut stats[basis_id];
                stat.affected_splats += 1;
                stat.sum_abs_weight += abs_weight;
                stat.max_abs_weight = stat.max_abs_weight.max(abs_weight);
            }
        }
    }

    for stat in &mut stats {
        if stat.affected_splats > 0 {
            stat.mean_abs_weight = stat.sum_abs_weight / stat.affected_splats as f32;
        }
    }
    stats
}

fn append_tile_coefficients_splat_major(
    ids_out: &mut Vec<u32>,
    weights_out: &mut Vec<f32>,
    coeffs: &BasisBankTileCoefficients,
    source_row_indices: &[u32],
    basis_count: usize,
) -> Result<()> {
    if source_row_indices.len() != coeffs.splat_count {
        bail!(
            "source-row permutation length {} != coefficient splat count {}",
            source_row_indices.len(),
            coeffs.splat_count
        );
    }
    for splat in 0..coeffs.splat_count {
        let source_splat = source_row_indices[splat] as usize;
        if source_splat >= coeffs.splat_count {
            bail!(
                "source-row permutation contains out-of-range row {} for {} splats",
                source_splat,
                coeffs.splat_count
            );
        }
        let src = source_splat * coeffs.top_k;
        for slot in 0..coeffs.top_k {
            let basis_id = coeffs.basis_ids[src + slot];
            if basis_id as usize >= basis_count {
                bail!(
                    "basis coefficient ID {} is out of range for {} shared bases",
                    basis_id,
                    basis_count
                );
            }
            ids_out.push(basis_id);
            weights_out.push(coeffs.weights[src + slot]);
        }
    }
    Ok(())
}

fn parse_legacy_deformation_duration_seconds(bytes: &[u8]) -> Result<f32> {
    if bytes.len() < DEFORMATION_HEADER_SIZE {
        bail!(
            "deformation_weights.bin header is too small: got {} bytes, expected at least {}",
            bytes.len(),
            DEFORMATION_HEADER_SIZE
        );
    }
    if &bytes[0..4] != b"DFWT" {
        bail!("deformation_weights.bin has invalid DFWT magic");
    }
    let version = read_u32(bytes, 4)?;
    if version != 1 {
        bail!(
            "deformation_weights.bin has unsupported header version {}; expected 1",
            version
        );
    }
    let temporal_resolution = read_u32(bytes, DEFORMATION_TEMPORAL_RESOLUTION_OFFSET)?;
    if temporal_resolution == 0 {
        bail!("deformation_weights.bin temporal resolution must be positive");
    }
    let duration_seconds =
        temporal_resolution as f32 * FRAMES_PER_TEMPORAL_GRID_SAMPLE / SOURCE_FRAME_RATE;
    if !duration_seconds.is_finite() || duration_seconds <= 0.0 {
        bail!(
            "deformation_weights.bin produced invalid duration {} from temporal resolution {}",
            duration_seconds,
            temporal_resolution
        );
    }
    Ok(duration_seconds)
}

fn read_zip_entry_prefix<R: Read + std::io::Seek>(
    archive: &mut ZipArchive<R>,
    index: usize,
    label: &str,
    prefix_len: usize,
) -> Result<Vec<u8>> {
    let mut file = archive
        .by_index(index)
        .with_context(|| format!("failed to open {}", label))?;
    if file.size() < prefix_len as u64 {
        bail!(
            "{} is too small: got {} bytes, expected at least {}",
            label,
            file.size(),
            prefix_len
        );
    }
    let mut bytes = vec![0; prefix_len];
    file.read_exact(bytes.as_mut_slice())
        .with_context(|| format!("failed to read {} header", label))?;
    Ok(bytes)
}

fn read_zip_entry<R: Read + std::io::Seek>(
    archive: &mut ZipArchive<R>,
    index: usize,
    label: &str,
) -> Result<Vec<u8>> {
    let mut file = archive
        .by_index(index)
        .with_context(|| format!("failed to open {}", label))?;
    let mut bytes = vec![0_u8; file.size() as usize];
    file.read_exact(bytes.as_mut_slice())
        .with_context(|| format!("failed to read {}", label))?;
    Ok(bytes)
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    let slice = bytes
        .get(offset..offset + 4)
        .with_context(|| format!("missing u32 at byte offset {}", offset))?;
    Ok(u32::from_le_bytes(slice.try_into().unwrap()))
}

fn read_f32_vec(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        bail!("f32 payload byte length is not divisible by 4");
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

pub fn basis_bank_delta(
    basis_knots: &[f32],
    basis_count: usize,
    knot_count: usize,
    basis_id: usize,
    time01: f32,
) -> [f32; 3] {
    debug_assert!(basis_id < basis_count);
    let phase = time01.rem_euclid(1.0);
    let scaled = phase * knot_count as f32;
    let segment = scaled.floor() as usize % knot_count;
    let u = scaled - segment as f32;
    let i0 = (segment + knot_count - 1) % knot_count;
    let i1 = segment;
    let i2 = (segment + 1) % knot_count;
    let i3 = (segment + 2) % knot_count;
    let p0 = basis_knot(basis_knots, knot_count, basis_id, i0);
    let p1 = basis_knot(basis_knots, knot_count, basis_id, i1);
    let p2 = basis_knot(basis_knots, knot_count, basis_id, i2);
    let p3 = basis_knot(basis_knots, knot_count, basis_id, i3);
    let u2 = u * u;
    let u3 = u2 * u;
    let mut out = [0.0; 3];
    for c in 0..3 {
        out[c] = 0.5
            * (2.0 * p1[c]
                + (-p0[c] + p2[c]) * u
                + (2.0 * p0[c] - 5.0 * p1[c] + 4.0 * p2[c] - p3[c]) * u2
                + (-p0[c] + 3.0 * p1[c] - 3.0 * p2[c] + p3[c]) * u3);
    }
    out
}

pub fn basis_bank_segment_kinematics(
    basis_knots: &[f32],
    basis_count: usize,
    knot_count: usize,
    basis_id: usize,
    segment: usize,
    segment_phase: f32,
) -> Option<BasisSegmentKinematics> {
    if basis_count == 0
        || knot_count == 0
        || basis_id >= basis_count
        || basis_knots.len() != basis_count.checked_mul(knot_count)?.checked_mul(3)?
    {
        return None;
    }

    let segment = segment % knot_count;
    let u = segment_phase.clamp(0.0, 1.0);
    let i0 = (segment + knot_count - 1) % knot_count;
    let i1 = segment;
    let i2 = (segment + 1) % knot_count;
    let i3 = (segment + 2) % knot_count;
    let p0 = basis_knot(basis_knots, knot_count, basis_id, i0);
    let p1 = basis_knot(basis_knots, knot_count, basis_id, i1);
    let p2 = basis_knot(basis_knots, knot_count, basis_id, i2);
    let p3 = basis_knot(basis_knots, knot_count, basis_id, i3);
    let u2 = u * u;
    let u3 = u2 * u;
    let mut position = [0.0; 3];
    let mut velocity = [0.0; 3];
    let mut acceleration = [0.0; 3];
    for c in 0..3 {
        let a = -p0[c] + p2[c];
        let b = 2.0 * p0[c] - 5.0 * p1[c] + 4.0 * p2[c] - p3[c];
        let d = -p0[c] + 3.0 * p1[c] - 3.0 * p2[c] + p3[c];
        position[c] = 0.5 * (2.0 * p1[c] + a * u + b * u2 + d * u3);
        velocity[c] = 0.5 * (a + 2.0 * b * u + 3.0 * d * u2);
        acceleration[c] = 0.5 * (2.0 * b + 6.0 * d * u);
    }
    Some(BasisSegmentKinematics {
        position,
        velocity,
        acceleration,
    })
}

pub fn basis_branch_continuity_debug(
    motion: &BasisBankMotionSet,
    branch: &BasisMotionGraphBranch,
) -> Option<BasisBranchContinuityDebug> {
    let source_basis_id = branch.from_basis;
    let target_basis_id = branch.to_basis;
    let source = basis_bank_segment_kinematics(
        motion.basis_knots.as_slice(),
        motion.basis_count,
        motion.meta.exported_knot_count,
        source_basis_id,
        branch.from_segment,
        1.0,
    )?;
    let target = basis_bank_segment_kinematics(
        motion.basis_knots.as_slice(),
        motion.basis_count,
        motion.meta.exported_knot_count,
        target_basis_id,
        branch.to_segment,
        0.0,
    )?;
    let position_delta = sub3(target.position, source.position);
    let velocity_delta = sub3(target.velocity, source.velocity);
    let acceleration_delta = sub3(target.acceleration, source.acceleration);
    Some(BasisBranchContinuityDebug {
        source_basis_id,
        target_basis_id,
        source_segment: branch.from_segment,
        target_segment: branch.to_segment,
        position_delta,
        velocity_delta,
        acceleration_delta,
        position_norm: norm3(position_delta),
        velocity_norm: norm3(velocity_delta),
        acceleration_norm: norm3(acceleration_delta),
    })
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn norm3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn basis_knot(basis_knots: &[f32], knot_count: usize, basis_id: usize, knot: usize) -> [f32; 3] {
    let base = (basis_id * knot_count + knot) * 3;
    [
        basis_knots[base],
        basis_knots[base + 1],
        basis_knots[base + 2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn current_meta() -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "format": BASIS_BANK_FORMAT,
            "format_version": 2,
            "delta_field": "delta_xyz",
            "basis_scope": "shared_lod0",
            "basis_source_lod": 0,
            "include_lods": [0, 1],
            "source_knot_count": 4,
            "exported_knot_count": 4,
            "loop_closure_knots": 0,
            "loop_closure_method": "cubic_hermite",
            "motion_teacher": "direct_network",
            "source_time_sampling": "inclusive",
            "motion_basis_closure_mode": "target_then_basis",
            "basis_count": 2,
            "top_k": 1,
            "duration_seconds": 2.5
        }))
        .unwrap()
    }

    #[test]
    fn accepts_current_constructor_metadata() {
        let meta = parse_basis_bank_meta(&current_meta()).unwrap();
        assert_eq!(meta.include_lods, vec![0, 1]);
        assert_eq!(meta.basis_count, 2);
    }

    #[test]
    fn version_one_metadata_requires_legacy_duration_context() {
        let mut value: serde_json::Value = serde_json::from_slice(&current_meta()).unwrap();
        value["format_version"] = 1.into();
        value.as_object_mut().unwrap().remove("duration_seconds");
        let error = parse_basis_bank_meta(&serde_json::to_vec(&value).unwrap())
            .unwrap_err()
            .to_string();
        assert!(error.contains("deformation_weights.bin"), "{error}");
    }

    #[test]
    fn infers_five_seconds_from_temporal_resolution_75() {
        let mut header = vec![0; DEFORMATION_HEADER_SIZE];
        header[0..4].copy_from_slice(b"DFWT");
        header[4..8].copy_from_slice(&1_u32.to_le_bytes());
        header[DEFORMATION_TEMPORAL_RESOLUTION_OFFSET..DEFORMATION_TEMPORAL_RESOLUTION_OFFSET + 4]
            .copy_from_slice(&75_u32.to_le_bytes());
        assert_eq!(
            parse_legacy_deformation_duration_seconds(&header).unwrap(),
            5.0
        );
    }

    #[test]
    fn rejects_missing_or_invalid_duration() {
        for duration in [serde_json::Value::Null, 0.0.into(), (-1.0).into()] {
            let mut value: serde_json::Value = serde_json::from_slice(&current_meta()).unwrap();
            value["duration_seconds"] = duration;
            assert!(parse_basis_bank_meta(&serde_json::to_vec(&value).unwrap()).is_err());
        }
    }

    #[test]
    fn rejects_inconsistent_knot_counts() {
        let mut value: serde_json::Value = serde_json::from_slice(&current_meta()).unwrap();
        value["loop_closure_knots"] = 1.into();
        let error = parse_basis_bank_meta(&serde_json::to_vec(&value).unwrap())
            .unwrap_err()
            .to_string();
        assert!(error.contains("source_knot_count"), "{error}");
    }

    #[test]
    fn accepts_both_current_closure_modes() {
        let mut value: serde_json::Value = serde_json::from_slice(&current_meta()).unwrap();
        value["motion_basis_closure_mode"] = "basis_then_closure".into();
        assert!(parse_basis_bank_meta(&serde_json::to_vec(&value).unwrap()).is_ok());
    }

    #[test]
    fn rejects_legacy_or_mismatched_metadata_policy() {
        for (key, value, expected) in [
            ("basis_scope", "per_lod", "expected shared_lod0"),
            ("motion_teacher", "volume", "direct_network"),
            ("source_time_sampling", "exclusive", "inclusive"),
            ("loop_closure_method", "none", "cubic_hermite"),
            ("motion_basis_closure_mode", "none", "target_then_basis"),
        ] {
            let mut meta: serde_json::Value = serde_json::from_slice(&current_meta()).unwrap();
            meta[key] = value.into();
            let error = parse_basis_bank_meta(&serde_json::to_vec(&meta).unwrap())
                .unwrap_err()
                .to_string();
            assert!(error.contains(expected), "{key}: {error}");
        }
    }

    #[test]
    fn coefficient_reordering_follows_sorted_scene_source_rows() {
        let coeffs = BasisBankTileCoefficients {
            format_version: BASIS_BANK_VERSION,
            tile_index: 0,
            lod_index: 0,
            splat_count: 3,
            top_k: 1,
            basis_ids: vec![0, 1, 0],
            weights: vec![0.1, 0.2, 0.3],
        };
        let mut ids = Vec::new();
        let mut weights = Vec::new();
        append_tile_coefficients_splat_major(&mut ids, &mut weights, &coeffs, &[2, 0, 1], 2)
            .unwrap();
        assert_eq!(ids, vec![0, 0, 1]);
        assert_eq!(weights, vec![0.3, 0.1, 0.2]);
    }

    #[test]
    fn rejects_non_finite_basis_knots() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(BASIS_LOD_MAGIC);
        for value in [BASIS_BANK_VERSION, 0, 1, 4] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in [f32::NAN; 12] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let error = parse_basis_lod_motion(&bytes).unwrap_err().to_string();
        assert!(error.contains("non-finite"), "{error}");
    }

    #[test]
    fn rejects_non_finite_coefficient_weights() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(BASIS_COEFFS_MAGIC);
        for value in [BASIS_BANK_VERSION, 0, 0, 1, 1] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        bytes.extend_from_slice(&f32::NAN.to_le_bytes());
        let error = parse_basis_tile_coefficients(&bytes)
            .unwrap_err()
            .to_string();
        assert!(error.contains("non-finite"), "{error}");
    }

    #[test]
    fn coefficient_size_overflow_returns_error_instead_of_panicking() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(BASIS_COEFFS_MAGIC);
        for value in [BASIS_BANK_VERSION, 0, 0, u32::MAX, u32::MAX] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let parsed = std::panic::catch_unwind(|| parse_basis_tile_coefficients(&bytes));
        assert!(parsed.is_ok(), "coefficient parser panicked on overflow");
        assert!(parsed.unwrap().is_err());
    }
}
