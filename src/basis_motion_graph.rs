use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::basis_bank_motion::{BasisBankMotionMeta, BasisInfo};

pub const BASIS_MOTION_GRAPH_FILENAME: &str = "motion_graph_basis.json";
const BASIS_MOTION_GRAPH_FORMAT: &str = "basis_motion_graph";
const BASIS_MOTION_GRAPH_MIN_VERSION: u32 = 1;
const BASIS_MOTION_GRAPH_MAX_VERSION: u32 = 2;

#[derive(Debug, Clone, Deserialize)]
pub struct BasisMotionGraph {
    pub format: String,
    pub format_version: u32,
    pub basis_scope: String,
    pub node_unit: String,
    pub include_lods: Vec<usize>,
    pub basis_count: usize,
    pub knot_count: usize,
    pub branch_top_k: usize,
    pub score_weights: BasisMotionGraphScoreWeights,
    pub lods: Vec<BasisMotionGraphLod>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct BasisMotionGraphScoreWeights {
    pub position: f32,
    pub velocity: f32,
    pub acceleration: f32,
    pub usage: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BasisMotionGraphLod {
    pub lod_id: usize,
    pub branches: Vec<BasisMotionGraphBranch>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BasisMotionGraphBranch {
    pub from_basis: usize,
    pub from_segment: usize,
    pub to_basis: usize,
    pub to_segment: usize,
    pub rank: usize,
    pub score: f32,
    pub position_cost: f32,
    pub velocity_cost: f32,
    pub acceleration_cost: f32,
    pub usage_bonus: f32,
    #[serde(default)]
    pub transition: Option<BasisMotionGraphTransition>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct BasisMotionGraphTransition {
    pub kind: String,
    pub duration_segments: usize,
    pub knots: Vec<[f32; 3]>,
    pub start_tangent: [f32; 3],
    pub end_tangent: [f32; 3],
}

impl BasisMotionGraph {
    pub fn validate_against_basis_bank(
        &self,
        meta: &BasisBankMotionMeta,
        basis_infos: &[BasisInfo],
    ) -> Result<()> {
        if self.include_lods != meta.include_lods {
            bail!(
                "basis motion graph include_lods {:?} != basis bank include_lods {:?}",
                self.include_lods,
                meta.include_lods
            );
        }
        if self.basis_count != meta.basis_count {
            bail!(
                "basis motion graph basis_count {} != basis bank basis_count {}",
                self.basis_count,
                meta.basis_count
            );
        }
        if self.knot_count != meta.exported_knot_count {
            bail!(
                "basis motion graph knot_count {} != basis bank exported_knot_count {}",
                self.knot_count,
                meta.exported_knot_count
            );
        }
        let expected_global_basis_count = self.include_lods.len() * self.basis_count;
        if basis_infos.len() != expected_global_basis_count {
            bail!(
                "basis motion graph expected {} global basis infos, got {}",
                expected_global_basis_count,
                basis_infos.len()
            );
        }
        for lod in &self.lods {
            if !self.include_lods.contains(&lod.lod_id) {
                bail!("basis motion graph references missing LOD {}", lod.lod_id);
            }
            for branch in &lod.branches {
                branch.validate(
                    lod.lod_id,
                    self.format_version,
                    self.basis_count,
                    self.knot_count,
                    self.branch_top_k,
                )?;
                if branch
                    .source_global_basis_id(lod.lod_id, basis_infos)
                    .is_none()
                {
                    bail!(
                        "basis motion graph source LOD {} local basis {} has no global basis",
                        lod.lod_id,
                        branch.from_basis
                    );
                }
                if branch
                    .target_global_basis_id(lod.lod_id, basis_infos)
                    .is_none()
                {
                    bail!(
                        "basis motion graph target LOD {} local basis {} has no global basis",
                        lod.lod_id,
                        branch.to_basis
                    );
                }
            }
        }
        Ok(())
    }

    pub fn branches_for(
        &self,
        lod_id: usize,
        local_basis_id: usize,
        segment: usize,
    ) -> Vec<&BasisMotionGraphBranch> {
        let Some(lod) = self.lods.iter().find(|lod| lod.lod_id == lod_id) else {
            return Vec::new();
        };
        let mut branches: Vec<_> = lod
            .branches
            .iter()
            .filter(|branch| branch.from_basis == local_basis_id && branch.from_segment == segment)
            .collect();
        branches.sort_by_key(|branch| branch.rank);
        branches
    }
}

impl BasisMotionGraphBranch {
    pub fn source_global_basis_id(
        &self,
        lod_id: usize,
        basis_infos: &[BasisInfo],
    ) -> Option<usize> {
        Self::global_basis_id_for(lod_id, basis_infos, self.from_basis)
    }

    pub fn target_global_basis_id(
        &self,
        lod_id: usize,
        basis_infos: &[BasisInfo],
    ) -> Option<usize> {
        Self::global_basis_id_for(lod_id, basis_infos, self.to_basis)
    }

    fn global_basis_id_for(
        lod_id: usize,
        basis_infos: &[BasisInfo],
        local_basis_id: usize,
    ) -> Option<usize> {
        basis_infos
            .iter()
            .position(|info| info.lod_id == lod_id && info.local_basis_id == local_basis_id)
    }

    fn validate(
        &self,
        lod_id: usize,
        format_version: u32,
        basis_count: usize,
        knot_count: usize,
        branch_top_k: usize,
    ) -> Result<()> {
        if self.from_basis >= basis_count || self.to_basis >= basis_count {
            bail!(
                "basis motion graph LOD {} branch basis out of range: {} -> {} with basis_count {}",
                lod_id,
                self.from_basis,
                self.to_basis,
                basis_count
            );
        }
        if self.from_segment >= knot_count || self.to_segment >= knot_count {
            bail!(
                "basis motion graph LOD {} branch segment out of range: {} -> {} with knot_count {}",
                lod_id,
                self.from_segment,
                self.to_segment,
                knot_count
            );
        }
        if self.from_basis == self.to_basis {
            bail!(
                "basis motion graph LOD {} branch targets the same local basis {}",
                lod_id,
                self.from_basis
            );
        }
        if self.rank >= branch_top_k {
            bail!(
                "basis motion graph LOD {} branch rank {} >= branch_top_k {}",
                lod_id,
                self.rank,
                branch_top_k
            );
        }
        if format_version >= 2 {
            let Some(transition) = self.transition.as_ref() else {
                bail!(
                    "basis motion graph LOD {} branch {}:{} -> {}:{} missing transition",
                    lod_id,
                    self.from_basis,
                    self.from_segment,
                    self.to_basis,
                    self.to_segment
                );
            };
            transition.validate(lod_id)?;
        }
        Ok(())
    }
}

impl BasisMotionGraphTransition {
    fn validate(&self, lod_id: usize) -> Result<()> {
        if self.kind != "open_catmull_rom" {
            bail!(
                "basis motion graph LOD {} unsupported transition kind '{}'",
                lod_id,
                self.kind
            );
        }
        if self.duration_segments == 0 {
            bail!(
                "basis motion graph LOD {} transition duration_segments must be positive",
                lod_id
            );
        }
        if self.knots.len() != self.duration_segments + 1 {
            bail!(
                "basis motion graph LOD {} transition knot count {} != duration_segments + 1 ({})",
                lod_id,
                self.knots.len(),
                self.duration_segments + 1
            );
        }
        for value in self
            .knots
            .iter()
            .flatten()
            .chain(self.start_tangent.iter())
            .chain(self.end_tangent.iter())
        {
            if !value.is_finite() {
                bail!(
                    "basis motion graph LOD {} transition contains non-finite value",
                    lod_id
                );
            }
        }
        Ok(())
    }
}

pub fn parse_basis_motion_graph(bytes: &[u8]) -> Result<BasisMotionGraph> {
    let graph: BasisMotionGraph =
        serde_json::from_slice(bytes).context("failed to parse basis motion graph JSON")?;
    if graph.format != BASIS_MOTION_GRAPH_FORMAT {
        bail!("unsupported basis motion graph format '{}'", graph.format);
    }
    if graph.format_version < BASIS_MOTION_GRAPH_MIN_VERSION
        || graph.format_version > BASIS_MOTION_GRAPH_MAX_VERSION
    {
        bail!(
            "unsupported basis motion graph version {}",
            graph.format_version
        );
    }
    if graph.basis_scope != "per_lod" {
        bail!(
            "unsupported basis motion graph scope '{}'",
            graph.basis_scope
        );
    }
    if graph.node_unit != "basis_segment" {
        bail!(
            "unsupported basis motion graph node unit '{}'",
            graph.node_unit
        );
    }
    if graph.basis_count == 0 || graph.knot_count == 0 || graph.branch_top_k == 0 {
        bail!(
            "invalid basis motion graph dimensions: basis_count={}, knot_count={}, branch_top_k={}",
            graph.basis_count,
            graph.knot_count,
            graph.branch_top_k
        );
    }
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis_bank_motion::{BasisBankMotionMeta, BasisInfo};

    fn valid_graph_json() -> &'static [u8] {
        br#"{
            "format": "basis_motion_graph",
            "format_version": 1,
            "basis_scope": "per_lod",
            "node_unit": "basis_segment",
            "include_lods": [0, 1],
            "basis_count": 2,
            "knot_count": 4,
            "branch_top_k": 3,
            "score_weights": {
                "position": 1.0,
                "velocity": 1.0,
                "acceleration": 0.5,
                "usage": 0.25
            },
            "lods": [
                {
                    "lod_id": 0,
                    "branches": [
                        {
                            "from_basis": 0,
                            "from_segment": 1,
                            "to_basis": 1,
                            "to_segment": 2,
                            "rank": 0,
                            "score": 0.125,
                            "position_cost": 0.1,
                            "velocity_cost": 0.02,
                            "acceleration_cost": 0.01,
                            "usage_bonus": 0.5
                        }
                    ]
                },
                {
                    "lod_id": 1,
                    "branches": []
                }
            ]
        }"#
    }

    fn matching_meta() -> BasisBankMotionMeta {
        BasisBankMotionMeta {
            format: crate::basis_bank_motion::BASIS_BANK_FORMAT.to_string(),
            format_version: 1,
            delta_field: "delta_xyz".to_string(),
            basis_scope: "per_lod".to_string(),
            include_lods: vec![0, 1],
            source_knot_count: 4,
            exported_knot_count: 4,
            loop_closure_knots: 0,
            loop_closure_method: "none".to_string(),
            motion_teacher: "volume".to_string(),
            volume_res: None,
            volume_key_count: None,
            basis_count: 2,
            top_k: 1,
            fit_report_by_lod: serde_json::Value::Null,
        }
    }

    fn basis_infos() -> Vec<BasisInfo> {
        vec![
            BasisInfo {
                lod_id: 0,
                local_basis_id: 0,
            },
            BasisInfo {
                lod_id: 0,
                local_basis_id: 1,
            },
            BasisInfo {
                lod_id: 1,
                local_basis_id: 0,
            },
            BasisInfo {
                lod_id: 1,
                local_basis_id: 1,
            },
        ]
    }

    #[test]
    fn parses_valid_basis_motion_graph_json() {
        let graph = parse_basis_motion_graph(valid_graph_json()).unwrap();

        assert_eq!(graph.include_lods, vec![0, 1]);
        assert_eq!(graph.basis_count, 2);
        assert_eq!(graph.knot_count, 4);
        assert_eq!(graph.branch_top_k, 3);
        assert_eq!(graph.lods.len(), 2);
        assert_eq!(graph.lods[0].branches.len(), 1);
        assert_eq!(graph.lods[0].branches[0].to_basis, 1);
    }

    #[test]
    fn rejects_basis_motion_graph_metadata_mismatch() {
        let graph = parse_basis_motion_graph(valid_graph_json()).unwrap();
        let mut meta = matching_meta();
        meta.basis_count = 3;

        let err = graph
            .validate_against_basis_bank(&meta, &basis_infos())
            .unwrap_err();

        assert!(err.to_string().contains("basis_count"));
    }

    #[test]
    fn maps_graph_branch_local_basis_to_global_basis_ids() {
        let graph = parse_basis_motion_graph(valid_graph_json()).unwrap();
        graph
            .validate_against_basis_bank(&matching_meta(), &basis_infos())
            .unwrap();

        let branches = graph.branches_for(0, 0, 1);
        assert_eq!(branches.len(), 1);
        assert_eq!(
            branches[0].target_global_basis_id(0, &basis_infos()),
            Some(1)
        );
        assert_eq!(
            branches[0].source_global_basis_id(0, &basis_infos()),
            Some(0)
        );
    }

    #[test]
    fn rejects_cross_lod_and_same_basis_branches() {
        let graph = parse_basis_motion_graph(
            br#"{
                "format": "basis_motion_graph",
                "format_version": 1,
                "basis_scope": "per_lod",
                "node_unit": "basis_segment",
                "include_lods": [0],
                "basis_count": 2,
                "knot_count": 4,
                "branch_top_k": 3,
                "score_weights": {"position":1.0,"velocity":1.0,"acceleration":0.5,"usage":0.25},
                "lods": [
                    {
                        "lod_id": 0,
                        "branches": [
                            {"from_basis":0,"from_segment":0,"to_basis":0,"to_segment":1,"rank":0,"score":0.1,"position_cost":0.1,"velocity_cost":0.0,"acceleration_cost":0.0,"usage_bonus":0.0}
                        ]
                    }
                ]
            }"#,
        )
        .unwrap();
        let mut meta = matching_meta();
        meta.include_lods = vec![0];

        let err = graph
            .validate_against_basis_bank(&meta, &basis_infos()[..2])
            .unwrap_err();

        assert!(err.to_string().contains("same local basis"));
    }

    #[test]
    fn parses_v2_basis_motion_graph_with_inline_transition() {
        let graph = parse_basis_motion_graph(
            br#"{
                "format": "basis_motion_graph",
                "format_version": 2,
                "basis_scope": "per_lod",
                "node_unit": "basis_segment",
                "include_lods": [0],
                "basis_count": 2,
                "knot_count": 4,
                "branch_top_k": 1,
                "score_weights": {"position":1.0,"velocity":1.0,"acceleration":0.5,"usage":0.25},
                "lods": [
                    {
                        "lod_id": 0,
                        "branches": [
                            {
                                "from_basis":0,
                                "from_segment":0,
                                "to_basis":1,
                                "to_segment":1,
                                "rank":0,
                                "score":0.1,
                                "position_cost":0.1,
                                "velocity_cost":0.0,
                                "acceleration_cost":0.0,
                                "usage_bonus":0.0,
                                "transition": {
                                    "kind": "open_catmull_rom",
                                    "duration_segments": 3,
                                    "knots": [[0.0,0.0,0.0],[0.3,0.0,0.0],[0.7,1.0,0.0],[1.0,1.0,0.0]],
                                    "start_tangent": [1.0,0.0,0.0],
                                    "end_tangent": [1.0,0.0,0.0]
                                }
                            }
                        ]
                    }
                ]
            }"#,
        )
        .unwrap();

        let branch = &graph.lods[0].branches[0];
        let transition = branch.transition.as_ref().unwrap();
        assert_eq!(graph.format_version, 2);
        assert_eq!(transition.duration_segments, 3);
        assert_eq!(transition.knots.len(), 4);
        assert_eq!(transition.start_tangent, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn rejects_v2_branch_missing_transition() {
        let graph = parse_basis_motion_graph(
            br#"{
                "format": "basis_motion_graph",
                "format_version": 2,
                "basis_scope": "per_lod",
                "node_unit": "basis_segment",
                "include_lods": [0],
                "basis_count": 2,
                "knot_count": 4,
                "branch_top_k": 1,
                "score_weights": {"position":1.0,"velocity":1.0,"acceleration":0.5,"usage":0.25},
                "lods": [
                    {
                        "lod_id": 0,
                        "branches": [
                            {"from_basis":0,"from_segment":0,"to_basis":1,"to_segment":1,"rank":0,"score":0.1,"position_cost":0.1,"velocity_cost":0.0,"acceleration_cost":0.0,"usage_bonus":0.0}
                        ]
                    }
                ]
            }"#,
        )
        .unwrap();
        let mut meta = matching_meta();
        meta.include_lods = vec![0];

        let err = graph
            .validate_against_basis_bank(&meta, &basis_infos()[..2])
            .unwrap_err();

        assert!(err.to_string().contains("transition"));
    }
}
