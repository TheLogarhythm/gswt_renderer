use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::basis_bank_motion::{BASIS_SCOPE_SHARED_LOD0, BasisBankMotionMeta};

pub const BASIS_MOTION_GRAPH_FILENAME: &str = "motion_graph_basis.json";
const BASIS_MOTION_GRAPH_FORMAT: &str = "basis_motion_graph";
const BASIS_MOTION_GRAPH_MIN_VERSION: u32 = 1;
const BASIS_MOTION_GRAPH_MAX_VERSION: u32 = 2;

#[derive(Debug, Clone, Deserialize)]
pub struct BasisMotionGraph {
    pub format: String,
    pub format_version: u32,
    pub basis_scope: String,
    #[serde(default)]
    pub basis_source_lod: Option<usize>,
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
    pub fn validate_against_basis_bank(&self, meta: &BasisBankMotionMeta) -> Result<()> {
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
        let [lod0] = self.lods.as_slice() else {
            bail!("shared_lod0 motion graph must contain exactly one LoD0 branch payload");
        };
        if lod0.lod_id != 0 {
            bail!(
                "shared_lod0 motion graph payload has lod_id={}; expected 0",
                lod0.lod_id
            );
        }
        for branch in &lod0.branches {
            branch.validate(
                self.format_version,
                self.basis_count,
                self.knot_count,
                self.branch_top_k,
            )?;
        }
        Ok(())
    }

    pub fn branches_for(&self, basis_id: usize, segment: usize) -> Vec<&BasisMotionGraphBranch> {
        let Some(lod0) = self.lods.first() else {
            return Vec::new();
        };
        let mut branches: Vec<_> = lod0
            .branches
            .iter()
            .filter(|branch| branch.from_basis == basis_id && branch.from_segment == segment)
            .collect();
        branches.sort_by_key(|branch| branch.rank);
        branches
    }
}

impl BasisMotionGraphBranch {
    fn validate(
        &self,
        format_version: u32,
        basis_count: usize,
        knot_count: usize,
        branch_top_k: usize,
    ) -> Result<()> {
        if self.from_basis >= basis_count || self.to_basis >= basis_count {
            bail!(
                "basis motion graph branch basis out of range: {} -> {} with basis_count {}",
                self.from_basis,
                self.to_basis,
                basis_count
            );
        }
        if self.from_segment >= knot_count || self.to_segment >= knot_count {
            bail!(
                "basis motion graph branch segment out of range: {} -> {} with knot_count {}",
                self.from_segment,
                self.to_segment,
                knot_count
            );
        }
        if self.from_basis == self.to_basis {
            bail!(
                "basis motion graph branch targets the same basis {}",
                self.from_basis
            );
        }
        if self.rank >= branch_top_k {
            bail!(
                "basis motion graph branch rank {} >= branch_top_k {}",
                self.rank,
                branch_top_k
            );
        }
        if format_version >= 2 {
            let Some(transition) = self.transition.as_ref() else {
                bail!(
                    "basis motion graph branch {}:{} -> {}:{} missing transition",
                    self.from_basis,
                    self.from_segment,
                    self.to_basis,
                    self.to_segment
                );
            };
            transition.validate()?;
        }
        Ok(())
    }
}

impl BasisMotionGraphTransition {
    fn validate(&self) -> Result<()> {
        if self.kind != "open_catmull_rom" {
            bail!(
                "unsupported basis motion graph transition kind '{}'",
                self.kind
            );
        }
        if self.duration_segments == 0 {
            bail!("basis motion graph transition duration_segments must be positive");
        }
        if self.knots.len() != self.duration_segments + 1 {
            bail!(
                "basis motion graph transition knot count {} != duration_segments + 1 ({})",
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
                bail!("basis motion graph transition contains non-finite value");
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
    if graph.basis_scope != BASIS_SCOPE_SHARED_LOD0 {
        bail!(
            "unsupported basis motion graph scope '{}'; expected shared_lod0",
            graph.basis_scope
        );
    }
    if graph.basis_source_lod != Some(0) {
        bail!(
            "shared_lod0 basis motion graph requires basis_source_lod=0, got {:?}",
            graph.basis_source_lod
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

    fn meta() -> BasisBankMotionMeta {
        BasisBankMotionMeta {
            format_version: 2,
            include_lods: vec![0, 1],
            source_knot_count: 4,
            exported_knot_count: 4,
            loop_closure_knots: 0,
            basis_count: 2,
            top_k: 1,
            duration_seconds: 2.5,
        }
    }

    fn graph_json() -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "format": "basis_motion_graph",
            "format_version": 1,
            "basis_scope": "shared_lod0",
            "basis_source_lod": 0,
            "node_unit": "basis_segment",
            "include_lods": [0, 1],
            "basis_count": 2,
            "knot_count": 4,
            "branch_top_k": 1,
            "score_weights": {"position": 1.0, "velocity": 1.0, "acceleration": 1.0, "usage": 0.0},
            "lods": [{
                "lod_id": 0,
                "branches": [{
                    "from_basis": 0, "from_segment": 1,
                    "to_basis": 1, "to_segment": 2,
                    "rank": 0, "score": 0.1,
                    "position_cost": 0.1, "velocity_cost": 0.0,
                    "acceleration_cost": 0.0, "usage_bonus": 0.0
                }]
            }]
        }))
        .unwrap()
    }

    #[test]
    fn validates_and_indexes_shared_basis_ids_directly() {
        let graph = parse_basis_motion_graph(&graph_json()).unwrap();
        graph.validate_against_basis_bank(&meta()).unwrap();
        let branches = graph.branches_for(0, 1);
        assert_eq!(branches.len(), 1);
        assert_eq!(branches[0].to_basis, 1);
    }

    #[test]
    fn rejects_per_lod_graph_scope() {
        let mut graph: serde_json::Value = serde_json::from_slice(&graph_json()).unwrap();
        graph["basis_scope"] = "per_lod".into();
        let error = parse_basis_motion_graph(&serde_json::to_vec(&graph).unwrap())
            .unwrap_err()
            .to_string();
        assert!(error.contains("expected shared_lod0"));
    }
}
