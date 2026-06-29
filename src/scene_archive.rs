use anyhow::{Context, Result, bail};
use regex::Regex;
use std::{
    io::{Cursor, Read},
    sync::Arc,
};

use crate::basis_bank_motion::{
    BASIS_BANK_META_FILENAME, BasisBankCoeffZipEntry, BasisBankLodZipEntry, BasisBankMotionSet,
    detect_basis_coeffs_file, detect_basis_lod_file, load_basis_bank_motion_from_zip,
};
use crate::basis_motion_graph::BASIS_MOTION_GRAPH_FILENAME;
use crate::scene::Scene;

pub struct SceneZipData {
    pub scene_vec: Vec<Vec<Scene>>,
    pub basis_bank_motion: Option<Arc<BasisBankMotionSet>>,
}

pub async fn load_scene_zip() -> Result<SceneZipData> {
    let file = rfd::AsyncFileDialog::new()
        .set_title("Upload Tiles (.zip)")
        .add_filter("Tiles", &["zip"])
        .pick_file()
        .await
        .context("No GSWT archive was selected")?;
    parse_scene_zip_bytes(file.read().await)
}

pub fn parse_scene_zip_bytes(file_zip: Vec<u8>) -> Result<SceneZipData> {
    struct SceneFileEntry {
        index: usize,
        filename: String,
        lod_id: usize,
        tile_id: usize,
    }

    let mut archive =
        zip::ZipArchive::new(Cursor::new(file_zip)).context("failed to open GSWT ZIP archive")?;
    let scene_name = Regex::new(r"^tile(\d+)_lod(\d+)\.(ply|splat)$").unwrap();
    let mut scene_entries = Vec::new();
    let mut deformation_weights_index = None;
    let mut legacy_dense_motion_present = false;
    let mut basis_bank_meta_index = None;
    let mut basis_motion_graph_index = None;
    let mut basis_bank_lod_entries = Vec::new();
    let mut basis_bank_coeff_entries = Vec::new();

    for index in 0..archive.len() {
        let file = archive.by_index(index)?;
        let path = file
            .enclosed_name()
            .with_context(|| format!("ZIP entry {index} has an unsafe path"))?;
        let filename = path
            .file_name()
            .and_then(|name| name.to_str())
            .with_context(|| format!("ZIP entry {index} has a non-UTF-8 filename"))?
            .to_string();
        let lower = filename.to_ascii_lowercase();

        if lower == "deformation_weights.bin" {
            if deformation_weights_index.replace(index).is_some() {
                bail!("duplicate deformation_weights.bin entries");
            }
            continue;
        }
        if lower == "motion_catmull_rom_meta.pt" || lower.ends_with("_motion_catmull_rom.pt") {
            legacy_dense_motion_present = true;
            continue;
        }
        if lower == BASIS_BANK_META_FILENAME {
            if basis_bank_meta_index.replace(index).is_some() {
                bail!("duplicate {BASIS_BANK_META_FILENAME} entries");
            }
            continue;
        }
        if lower == BASIS_MOTION_GRAPH_FILENAME {
            if basis_motion_graph_index.replace(index).is_some() {
                bail!("duplicate {BASIS_MOTION_GRAPH_FILENAME} entries");
            }
            continue;
        }
        if let Some(lod_id) = detect_basis_lod_file(&lower) {
            basis_bank_lod_entries.push(BasisBankLodZipEntry {
                index,
                filename,
                lod_id,
            });
            continue;
        }
        if let Some((tile_id, lod_id)) = detect_basis_coeffs_file(&lower) {
            basis_bank_coeff_entries.push(BasisBankCoeffZipEntry {
                index,
                filename,
                tile_id,
                lod_id,
            });
            continue;
        }
        if let Some(captures) = scene_name.captures(&lower) {
            scene_entries.push(SceneFileEntry {
                index,
                filename,
                tile_id: captures[1].parse()?,
                lod_id: captures[2].parse()?,
            });
        }
    }

    if legacy_dense_motion_present {
        bail!(
            "legacy dense Catmull-Rom motion assets are unsupported; rebuild the archive with the current gswt_constructor"
        );
    }
    if scene_entries.is_empty() {
        bail!("archive contains no tile{{id}}_lod{{id}}.ply or .splat files");
    }

    scene_entries.sort_by_key(|entry| (entry.lod_id, entry.tile_id));
    let max_lod = scene_entries.last().unwrap().lod_id;
    let max_tile = scene_entries
        .iter()
        .map(|entry| entry.tile_id)
        .max()
        .unwrap();
    let lod_count = max_lod + 1;
    let tile_count = max_tile + 1;
    if scene_entries.len() != lod_count * tile_count {
        bail!(
            "incomplete tile grid: found {} scene files, expected {} LoDs x {} tiles = {} files",
            scene_entries.len(),
            lod_count,
            tile_count,
            lod_count * tile_count
        );
    }
    for (index, entry) in scene_entries.iter().enumerate() {
        let expected = (index / tile_count, index % tile_count);
        if (entry.lod_id, entry.tile_id) != expected {
            bail!(
                "incomplete tile grid: expected tile{}_lod{}, found {}",
                expected.1,
                expected.0,
                entry.filename
            );
        }
    }

    let mut scene_vec = Vec::with_capacity(lod_count);
    for lod_id in 0..lod_count {
        let mut lod = Vec::with_capacity(tile_count);
        for tile_id in 0..tile_count {
            let entry = &scene_entries[lod_id * tile_count + tile_id];
            let mut file = archive.by_index(entry.index)?;
            let mut bytes = vec![0; file.size() as usize];
            file.read_exact(&mut bytes)
                .with_context(|| format!("failed to read {}", entry.filename))?;
            let mut scene = Scene::new();
            if entry.filename.to_ascii_lowercase().ends_with(".ply") {
                let (header, mut cursor) = Scene::parse_file_header(bytes)
                    .map_err(anyhow::Error::msg)
                    .with_context(|| format!("failed to parse {}", entry.filename))?;
                scene.splat_count = header.splat_count;
                scene
                    .load(&mut cursor, &header)
                    .map_err(anyhow::Error::msg)
                    .with_context(|| format!("failed to load {}", entry.filename))?;
            } else {
                if bytes.len() % 32 != 0 {
                    bail!("{} byte length is not a multiple of 32", entry.filename);
                }
                scene.splat_count = bytes.len() / 32;
                scene.source_row_indices = (0..scene.splat_count as u32).collect();
                scene.buffer = bytes;
            }
            lod.push(scene);
        }
        scene_vec.push(lod);
    }

    let basis_files_present = !basis_bank_lod_entries.is_empty()
        || !basis_bank_coeff_entries.is_empty()
        || basis_motion_graph_index.is_some();
    let basis_bank_motion = if let Some(meta_index) = basis_bank_meta_index {
        if basis_bank_lod_entries.len() != 1 || basis_bank_lod_entries[0].lod_id != 0 {
            bail!(
                "shared-LoD0 motion requires exactly lod0_motion_basis.bin; rebuild with the current gswt_constructor"
            );
        }
        let expected_coefficients = lod_count * tile_count;
        if basis_bank_coeff_entries.len() != expected_coefficients {
            bail!(
                "shared-LoD0 motion has {} coefficient files, expected {}",
                basis_bank_coeff_entries.len(),
                expected_coefficients
            );
        }
        let mut seen = std::collections::HashSet::new();
        for entry in &basis_bank_coeff_entries {
            if entry.lod_id >= lod_count || entry.tile_id >= tile_count {
                bail!("{} addresses a tile outside the scene grid", entry.filename);
            }
            if !seen.insert((entry.lod_id, entry.tile_id)) {
                bail!(
                    "duplicate basis coefficients for tile{}_lod{}",
                    entry.tile_id,
                    entry.lod_id
                );
            }
        }
        let motion = load_basis_bank_motion_from_zip(
            &mut archive,
            meta_index,
            deformation_weights_index,
            basis_motion_graph_index,
            &basis_bank_lod_entries,
            &basis_bank_coeff_entries,
            &scene_vec,
        )?
        .context("basis-bank assets were present but could not be loaded")?;
        let expected_lods: Vec<_> = (0..lod_count).collect();
        if motion.meta.include_lods != expected_lods {
            bail!(
                "basis metadata include_lods {:?} does not match scene LoDs {:?}",
                motion.meta.include_lods,
                expected_lods
            );
        }
        Some(motion)
    } else {
        if basis_files_present {
            bail!(
                "basis-bank files are present but {BASIS_BANK_META_FILENAME} is missing; rebuild with the current gswt_constructor"
            );
        }
        if deformation_weights_index.is_some() {
            bail!(
                "deformation_weights.bin is present without required shared-LoD0 basis assets; rebuild with the current gswt_constructor"
            );
        }
        None
    };

    Ok(SceneZipData {
        scene_vec,
        basis_bank_motion,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use zip::{ZipWriter, write::SimpleFileOptions};

    fn zip_bytes(entries: Vec<(&str, Vec<u8>)>) -> Vec<u8> {
        let mut writer = ZipWriter::new(Cursor::new(Vec::new()));
        for (name, bytes) in entries {
            writer
                .start_file(name, SimpleFileOptions::default())
                .unwrap();
            writer.write_all(&bytes).unwrap();
        }
        writer.finish().unwrap().into_inner()
    }

    fn splat(count: usize) -> Vec<u8> {
        vec![0; count * 32]
    }

    fn meta(scope: &str, source_lod: usize, include_lods: &[usize]) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "format": "loop_closed_catmull_rom_basis_bank_delta_xyz",
            "format_version": 2,
            "delta_field": "delta_xyz",
            "basis_scope": scope,
            "basis_source_lod": source_lod,
            "include_lods": include_lods,
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

    fn version_one_meta() -> Vec<u8> {
        let mut value: serde_json::Value =
            serde_json::from_slice(&meta("shared_lod0", 0, &[0])).unwrap();
        value["format_version"] = 1.into();
        value.as_object_mut().unwrap().remove("duration_seconds");
        serde_json::to_vec(&value).unwrap()
    }

    fn deformation_header(temporal_resolution: u32) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"DFWT");
        for value in [1_u32, 3, 6, 32, 64, temporal_resolution] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn basis_payload() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"MBSB");
        for value in [2_u32, 0, 2, 4] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in 0..24 {
            bytes.extend_from_slice(&(value as f32).to_le_bytes());
        }
        bytes
    }

    fn coeff_payload(tile: u32, lod: u32, ids: &[u32]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"MBCF");
        for value in [2_u32, tile, lod, ids.len() as u32, 1] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for id in ids {
            bytes.extend_from_slice(&id.to_le_bytes());
        }
        for _ in ids {
            bytes.extend_from_slice(&1.0_f32.to_le_bytes());
        }
        bytes
    }

    fn set_payload_version(bytes: &mut [u8], version: u32) {
        bytes[4..8].copy_from_slice(&version.to_le_bytes());
    }

    fn valid_dynamic_entries() -> Vec<(&'static str, Vec<u8>)> {
        vec![
            ("tile0_lod0.splat", splat(1)),
            (BASIS_BANK_META_FILENAME, meta("shared_lod0", 0, &[0])),
            ("lod0_motion_basis.bin", basis_payload()),
            (
                "tile0_lod0_motion_basis_coeffs.bin",
                coeff_payload(0, 0, &[1]),
            ),
        ]
    }

    fn valid_version_one_dynamic_entries() -> Vec<(&'static str, Vec<u8>)> {
        let mut basis = basis_payload();
        let mut coeffs = coeff_payload(0, 0, &[1]);
        set_payload_version(&mut basis, 1);
        set_payload_version(&mut coeffs, 1);
        vec![
            ("tile0_lod0.splat", splat(1)),
            (BASIS_BANK_META_FILENAME, version_one_meta()),
            ("lod0_motion_basis.bin", basis),
            ("tile0_lod0_motion_basis_coeffs.bin", coeffs),
            ("deformation_weights.bin", deformation_header(75)),
        ]
    }

    fn error_for(entries: Vec<(&str, Vec<u8>)>) -> String {
        match parse_scene_zip_bytes(zip_bytes(entries)) {
            Ok(_) => panic!("archive unexpectedly accepted"),
            Err(error) => format!("{error:#}"),
        }
    }

    #[test]
    fn accepts_static_archive() {
        let archive =
            parse_scene_zip_bytes(zip_bytes(vec![("tile0_lod0.splat", splat(1))])).unwrap();
        assert!(archive.basis_bank_motion.is_none());
        assert_eq!(archive.scene_vec.len(), 1);
    }

    #[test]
    fn accepts_valid_shared_lod0_archive() {
        let archive = parse_scene_zip_bytes(zip_bytes(valid_dynamic_entries())).unwrap();
        let motion = archive.basis_bank_motion.unwrap();
        assert_eq!(motion.basis_count, 2);
        assert_eq!(motion.basis_ids, vec![1]);
        assert_eq!(motion.meta.duration_seconds, 2.5);
    }

    #[test]
    fn accepts_version_one_archive_and_infers_duration_from_temporal_resolution() {
        let archive =
            parse_scene_zip_bytes(zip_bytes(valid_version_one_dynamic_entries())).unwrap();
        let motion = archive.basis_bank_motion.unwrap();
        assert_eq!(motion.meta.duration_seconds, 5.0);
    }

    #[test]
    fn rejects_version_one_archive_without_deformation_header() {
        let mut entries = valid_version_one_dynamic_entries();
        entries.pop();
        let error = error_for(entries);
        assert!(error.contains("deformation_weights.bin"), "{error}");
    }

    #[test]
    fn rejects_version_one_archive_with_invalid_temporal_resolution() {
        let mut entries = valid_version_one_dynamic_entries();
        entries[4].1 = deformation_header(0);
        let error = error_for(entries);
        assert!(error.contains("temporal resolution"), "{error}");
    }

    #[test]
    fn rejects_mixed_basis_asset_versions() {
        let mut entries = valid_version_one_dynamic_entries();
        set_payload_version(&mut entries[2].1, 2);
        let error = error_for(entries);
        assert!(error.contains("version 2"), "{error}");
        assert!(error.contains("metadata version 1"), "{error}");
    }

    #[test]
    fn rejects_per_lod_scope() {
        let mut entries = valid_dynamic_entries();
        entries[1].1 = meta("per_lod", 0, &[0]);
        assert!(error_for(entries).contains("expected shared_lod0"));
    }

    #[test]
    fn rejects_wrong_source_lod() {
        let mut entries = valid_dynamic_entries();
        entries[1].1 = meta("shared_lod0", 1, &[0]);
        assert!(error_for(entries).contains("basis_source_lod=0"));
    }

    #[test]
    fn rejects_incomplete_lod_coverage() {
        let mut entries = valid_dynamic_entries();
        entries.insert(1, ("tile0_lod1.splat", splat(1)));
        entries.push((
            "tile0_lod1_motion_basis_coeffs.bin",
            coeff_payload(0, 1, &[0]),
        ));
        assert!(error_for(entries).contains("include_lods"));
    }

    #[test]
    fn rejects_missing_coefficient_payload() {
        let mut entries = valid_dynamic_entries();
        entries.pop();
        assert!(error_for(entries).contains("coefficient files"));
    }

    #[test]
    fn rejects_duplicate_coefficient_payload() {
        let mut entries = valid_dynamic_entries();
        entries.push((
            "Tile0_Lod0_Motion_Basis_Coeffs.bin",
            coeff_payload(0, 0, &[0]),
        ));
        let error = error_for(entries);
        assert!(error.contains("coefficient files") || error.contains("duplicate"));
    }

    #[test]
    fn rejects_bad_coefficient_header() {
        let mut entries = valid_dynamic_entries();
        entries[3].1 = coeff_payload(9, 0, &[0]);
        assert!(error_for(entries).contains("expected tile0_lod0"));
    }

    #[test]
    fn rejects_out_of_range_basis_id() {
        let mut entries = valid_dynamic_entries();
        entries[3].1 = coeff_payload(0, 0, &[2]);
        let error = error_for(entries);
        assert!(
            error.contains("basis coefficient ID 2 is out of range"),
            "{error}"
        );
    }

    #[test]
    fn rejects_deformation_only_archive() {
        let error = error_for(vec![
            ("tile0_lod0.splat", splat(1)),
            ("deformation_weights.bin", vec![1, 2, 3]),
        ]);
        assert!(error.contains("deformation_weights.bin"));
        assert!(error.contains("rebuild"));
    }

    #[test]
    fn rejects_dense_catmull_rom_archive() {
        let error = error_for(vec![
            ("tile0_lod0.splat", splat(1)),
            ("motion_catmull_rom_meta.pt", vec![1]),
        ]);
        assert!(error.contains("legacy dense Catmull-Rom"));
    }

    #[test]
    fn invalid_optional_graph_keeps_basis_playback() {
        let mut entries = valid_dynamic_entries();
        entries.push((BASIS_MOTION_GRAPH_FILENAME, b"{}".to_vec()));
        let archive = parse_scene_zip_bytes(zip_bytes(entries)).unwrap();
        let motion = archive.basis_bank_motion.unwrap();
        assert!(motion.motion_graph.is_none());
    }
}
