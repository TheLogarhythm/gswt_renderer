use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use regex::Regex;
use serde::Deserialize;
use zip::ZipArchive;

use crate::log;
use crate::scene::Scene;

pub const BASIS_BANK_META_FILENAME: &str = "motion_basis_meta.bin";
pub const BASIS_BANK_FORMAT: &str = "loop_closed_catmull_rom_basis_bank_delta_xyz";
const BASIS_BANK_VERSION: u32 = 1;
const BASIS_LOD_MAGIC: &[u8; 4] = b"MBSB";
const BASIS_COEFFS_MAGIC: &[u8; 4] = b"MBCF";

#[derive(Debug, Clone, Deserialize)]
pub struct BasisBankMotionMeta {
    pub format: String,
    pub format_version: u32,
    pub delta_field: String,
    pub basis_scope: String,
    pub include_lods: Vec<usize>,
    pub source_knot_count: usize,
    pub exported_knot_count: usize,
    pub loop_closure_knots: usize,
    pub loop_closure_method: String,
    pub motion_teacher: String,
    pub volume_res: Option<usize>,
    pub volume_key_count: Option<usize>,
    pub basis_count: usize,
    pub top_k: usize,
    #[serde(default)]
    pub fit_report_by_lod: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct BasisBankLodMotion {
    pub lod_index: usize,
    pub basis_count: usize,
    pub knot_count: usize,
    /// Basis-major storage: ((basis * knot_count + knot) * 3 + xyz).
    pub basis_knots: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct BasisBankTileCoefficients {
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
    pub total_splats: usize,
    pub global_basis_count: usize,
    pub basis_infos: Vec<BasisInfo>,
    pub usage_stats: Vec<BasisUsageStats>,
    /// Global basis-major storage: ((basis * knot_count + knot) * 3 + xyz).
    pub global_basis_knots: Vec<f32>,
    /// Global splat-major sparse IDs: splat * top_k + slot.
    pub global_basis_ids: Vec<u32>,
    /// Global splat-major sparse weights: splat * top_k + slot.
    pub global_weights: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BasisInfo {
    pub lod_id: usize,
    pub local_basis_id: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BasisUsageStats {
    pub affected_splats: u32,
    pub sum_abs_weight: f32,
    pub max_abs_weight: f32,
    pub mean_abs_weight: f32,
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
    let meta: BasisBankMotionMeta =
        serde_json::from_slice(bytes).context("failed to parse basis-bank metadata JSON")?;
    if meta.format != BASIS_BANK_FORMAT {
        bail!("unsupported basis-bank format '{}'", meta.format);
    }
    if meta.format_version != BASIS_BANK_VERSION {
        bail!("unsupported basis-bank version {}", meta.format_version);
    }
    if meta.delta_field != "delta_xyz" {
        bail!("unsupported basis-bank delta field '{}'", meta.delta_field);
    }
    if meta.basis_scope != "per_lod" {
        bail!("unsupported basis-bank scope '{}'", meta.basis_scope);
    }
    if meta.exported_knot_count < 4 {
        bail!(
            "basis-bank exported knot count must be at least 4, got {}",
            meta.exported_knot_count
        );
    }
    if meta.basis_count == 0 || meta.top_k == 0 || meta.top_k > meta.basis_count {
        bail!(
            "invalid basis-bank basis_count/top_k: basis_count={}, top_k={}",
            meta.basis_count,
            meta.top_k
        );
    }
    Ok(meta)
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
    if version != BASIS_BANK_VERSION {
        bail!("unsupported basis LOD version {}", version);
    }
    let lod_index = read_u32(bytes, 8)? as usize;
    let basis_count = read_u32(bytes, 12)? as usize;
    let knot_count = read_u32(bytes, 16)? as usize;
    let value_count = basis_count
        .checked_mul(knot_count)
        .and_then(|v| v.checked_mul(3))
        .context("basis LOD value count overflow")?;
    let expected = HEADER_SIZE + value_count * 4;
    if bytes.len() != expected {
        bail!(
            "basis LOD byte length mismatch: got {}, expected {}",
            bytes.len(),
            expected
        );
    }
    let basis_knots = read_f32_vec(&bytes[HEADER_SIZE..])?;
    Ok(BasisBankLodMotion {
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
    if version != BASIS_BANK_VERSION {
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
    let weights_start = ids_start + count * 4;
    let expected = weights_start + count * 4;
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
    Ok(BasisBankTileCoefficients {
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
    lod_entries: &[BasisBankLodZipEntry],
    coeff_entries: &[BasisBankCoeffZipEntry],
    scene_vec: &[Vec<Scene>],
) -> Result<Option<Arc<BasisBankMotionSet>>> {
    let meta_bytes = read_zip_entry(archive, meta_index, BASIS_BANK_META_FILENAME)?;
    let meta = parse_basis_bank_meta(&meta_bytes)?;
    let included_lods: HashSet<usize> = meta.include_lods.iter().copied().collect();

    let mut basis_by_lod = HashMap::new();
    for entry in lod_entries {
        if !included_lods.contains(&entry.lod_id) {
            continue;
        }
        let bytes = read_zip_entry(archive, entry.index, &entry.filename)?;
        let basis = parse_basis_lod_motion(&bytes)
            .with_context(|| format!("failed to parse {}", entry.filename))?;
        if basis.lod_index != entry.lod_id
            || basis.basis_count != meta.basis_count
            || basis.knot_count != meta.exported_knot_count
        {
            log!(
                "Basis-bank LOD payload mismatch in {}; disabling basis-bank backend.",
                entry.filename
            );
            return Ok(None);
        }
        basis_by_lod.insert(entry.lod_id, basis);
    }

    for lod_id in &included_lods {
        if !basis_by_lod.contains_key(lod_id) {
            log!(
                "Basis-bank basis payload missing for lod{}; disabling basis-bank backend.",
                lod_id
            );
            return Ok(None);
        }
    }

    let mut coeff_by_lod_tile: HashMap<(usize, usize), &BasisBankCoeffZipEntry> = HashMap::new();
    for entry in coeff_entries {
        coeff_by_lod_tile.insert((entry.lod_id, entry.tile_id), entry);
    }

    let mut lod_basis_offset = HashMap::new();
    let mut global_basis_knots = Vec::new();
    for lod_id in &meta.include_lods {
        let basis = basis_by_lod
            .get(lod_id)
            .with_context(|| format!("missing basis payload for lod{}", lod_id))?;
        let offset = global_basis_knots.len() / (meta.exported_knot_count * 3);
        lod_basis_offset.insert(*lod_id, offset);
        global_basis_knots.extend_from_slice(&basis.basis_knots);
    }
    if global_basis_knots.is_empty() {
        log!("Basis-bank has no basis knots; disabling basis-bank backend.");
        return Ok(None);
    }

    let mut total_splats = 0_usize;
    let mut global_basis_ids = Vec::new();
    let mut global_weights = Vec::new();
    for (lod_id, lod_vec) in scene_vec.iter().enumerate() {
        for (tile_id, scene) in lod_vec.iter().enumerate() {
            total_splats += scene.splat_count;
            if !included_lods.contains(&lod_id) {
                append_zero_coefficients(
                    &mut global_basis_ids,
                    &mut global_weights,
                    scene.splat_count,
                    meta.top_k,
                );
                continue;
            }
            let Some(entry) = coeff_by_lod_tile.get(&(lod_id, tile_id)) else {
                log!(
                    "Basis-bank coefficients missing for tile{}_lod{}; disabling basis-bank backend.",
                    tile_id,
                    lod_id
                );
                return Ok(None);
            };
            let bytes = read_zip_entry(archive, entry.index, &entry.filename)?;
            let coeffs = parse_basis_tile_coefficients(&bytes)
                .with_context(|| format!("failed to parse {}", entry.filename))?;
            if coeffs.tile_index != tile_id
                || coeffs.lod_index != lod_id
                || coeffs.splat_count != scene.splat_count
                || coeffs.top_k != meta.top_k
            {
                log!(
                    "Basis-bank coefficient mismatch for tile{}_lod{}; disabling basis-bank backend.",
                    tile_id,
                    lod_id
                );
                return Ok(None);
            }
            let offset = *lod_basis_offset.get(&lod_id).unwrap() as u32;
            if let Err(err) = append_tile_coefficients_splat_major(
                &mut global_basis_ids,
                &mut global_weights,
                &coeffs,
                scene.source_row_indices.as_slice(),
                offset,
            ) {
                log!(
                    "Basis-bank coefficient reorder failed for tile{}_lod{}: {}; disabling basis-bank backend.",
                    tile_id,
                    lod_id,
                    err
                );
                return Ok(None);
            }
        }
    }

    let global_basis_count = global_basis_knots.len() / (meta.exported_knot_count * 3);
    let basis_infos = build_basis_infos(&meta.include_lods, meta.basis_count);
    let usage_stats = compute_basis_usage_stats(
        &global_basis_ids,
        &global_weights,
        global_basis_count,
        meta.top_k,
    );
    let fit_report_lod_count = meta
        .fit_report_by_lod
        .as_object()
        .map_or(0, |lods| lods.len());
    log!(
        "Basis-bank motion loaded: lods={:?}, basis_per_lod={}, global_basis={}, top_k={}, knots={}, source_knots={}, closure_knots={}, closure_method={}, teacher={}, volume_res={:?}, volume_key_count={:?}, fit_report_lods={}, total_splats={}",
        meta.include_lods,
        meta.basis_count,
        global_basis_count,
        meta.top_k,
        meta.exported_knot_count,
        meta.source_knot_count,
        meta.loop_closure_knots,
        meta.loop_closure_method,
        meta.motion_teacher,
        meta.volume_res,
        meta.volume_key_count,
        fit_report_lod_count,
        total_splats
    );
    Ok(Some(Arc::new(BasisBankMotionSet {
        meta,
        total_splats,
        global_basis_count,
        basis_infos,
        usage_stats,
        global_basis_knots,
        global_basis_ids,
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

pub fn build_basis_infos(include_lods: &[usize], basis_count: usize) -> Vec<BasisInfo> {
    let mut infos = Vec::with_capacity(include_lods.len() * basis_count);
    for &lod_id in include_lods {
        for local_basis_id in 0..basis_count {
            infos.push(BasisInfo {
                lod_id,
                local_basis_id,
            });
        }
    }
    infos
}

fn append_zero_coefficients(
    ids: &mut Vec<u32>,
    weights: &mut Vec<f32>,
    splat_count: usize,
    top_k: usize,
) {
    ids.resize(ids.len() + splat_count * top_k, 0);
    weights.resize(weights.len() + splat_count * top_k, 0.0);
}

fn append_tile_coefficients_splat_major(
    ids_out: &mut Vec<u32>,
    weights_out: &mut Vec<f32>,
    coeffs: &BasisBankTileCoefficients,
    source_row_indices: &[u32],
    basis_offset: u32,
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
            ids_out.push(coeffs.basis_ids[src + slot] + basis_offset);
            weights_out.push(coeffs.weights[src + slot]);
        }
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

    #[test]
    fn detects_basis_bank_asset_names() {
        assert_eq!(detect_basis_lod_file("lod3_motion_basis.bin"), Some(3));
        assert_eq!(
            detect_basis_coeffs_file("tile12_lod4_motion_basis_coeffs.bin"),
            Some((12, 4))
        );
        assert_eq!(
            detect_basis_lod_file("tile0_lod0_motion_basis_coeffs.bin"),
            None
        );
    }

    #[test]
    fn parses_basis_bank_meta_json() {
        let bytes = br#"{
            "format": "loop_closed_catmull_rom_basis_bank_delta_xyz",
            "format_version": 1,
            "delta_field": "delta_xyz",
            "basis_scope": "per_lod",
            "include_lods": [0, 2],
            "source_knot_count": 25,
            "exported_knot_count": 28,
            "loop_closure_knots": 3,
            "loop_closure_method": "cubic_hermite",
            "motion_teacher": "volume",
            "volume_res": 64,
            "volume_key_count": 25,
            "basis_count": 64,
            "top_k": 8
        }"#;
        let meta = parse_basis_bank_meta(bytes).unwrap();
        assert_eq!(meta.exported_knot_count, 28);
        assert_eq!(meta.top_k, 8);
    }

    #[test]
    fn basis_bank_evaluator_wraps_and_hits_knots() {
        let knots = vec![
            0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, //
            2.0, 0.0, 0.0, //
            3.0, 0.0, 0.0, //
        ];
        assert_eq!(basis_bank_delta(&knots, 1, 4, 0, 0.0), [0.0, 0.0, 0.0]);
        assert_eq!(basis_bank_delta(&knots, 1, 4, 0, 1.0), [0.0, 0.0, 0.0]);
        assert_eq!(basis_bank_delta(&knots, 1, 4, 0, 0.25), [1.0, 0.0, 0.0]);
    }

    #[test]
    fn basis_infos_map_global_id_to_lod_and_local_basis() {
        let infos = build_basis_infos(&[0, 2], 3);
        assert_eq!(infos.len(), 6);
        assert_eq!(
            infos[0],
            BasisInfo {
                lod_id: 0,
                local_basis_id: 0
            }
        );
        assert_eq!(
            infos[2],
            BasisInfo {
                lod_id: 0,
                local_basis_id: 2
            }
        );
        assert_eq!(
            infos[3],
            BasisInfo {
                lod_id: 2,
                local_basis_id: 0
            }
        );
        assert_eq!(
            infos[5],
            BasisInfo {
                lod_id: 2,
                local_basis_id: 2
            }
        );
    }

    #[test]
    fn basis_usage_stats_count_splats_once_per_basis() {
        let basis_ids = vec![
            0, 1, 0, 2, //
            1, 1, 2, 0, //
        ];
        let weights = vec![
            0.5, -0.25, 0.5, 0.0, //
            0.1, 0.2, -0.4, 0.0, //
        ];
        let stats = compute_basis_usage_stats(&basis_ids, &weights, 3, 4);

        assert_eq!(stats[0].affected_splats, 1);
        assert!((stats[0].sum_abs_weight - 1.0).abs() < 1e-6);
        assert!((stats[0].max_abs_weight - 1.0).abs() < 1e-6);
        assert!((stats[0].mean_abs_weight - 1.0).abs() < 1e-6);

        assert_eq!(stats[1].affected_splats, 2);
        assert!((stats[1].sum_abs_weight - 0.55).abs() < 1e-6);
        assert!((stats[1].max_abs_weight - 0.3).abs() < 1e-6);
        assert!((stats[1].mean_abs_weight - 0.275).abs() < 1e-6);

        assert_eq!(stats[2].affected_splats, 1);
        assert!((stats[2].sum_abs_weight - 0.4).abs() < 1e-6);
    }
}
