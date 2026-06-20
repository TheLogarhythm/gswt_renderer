use std::collections::{HashMap, HashSet};

use crate::basis_bank_motion::{
    BASIS_SCOPE_SHARED_LOD0, BasisBankMotionSet, BasisInfo, BasisSegmentKinematics,
    basis_bank_segment_kinematics,
};
use crate::basis_motion_graph::{
    BasisMotionGraph, BasisMotionGraphBranch, BasisMotionGraphTransition,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct BasisGraphNodeKey {
    pub graph_lod_id: usize,
    pub local_basis_id: usize,
    pub segment: usize,
}

impl BasisGraphNodeKey {
    pub fn new(graph_lod_id: usize, local_basis_id: usize, segment: usize) -> Self {
        Self {
            graph_lod_id,
            local_basis_id,
            segment,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BasisGraphAuthoringRefreshSummary {
    pub refreshed_lods: Vec<usize>,
    pub refreshed_node_count: usize,
    pub changed_best_target_count: usize,
    pub last_refresh_ms: f32,
}

#[derive(Clone, Debug, Default)]
pub struct BasisGraphBranchOverrides {
    branches_by_node: HashMap<BasisGraphNodeKey, Vec<BasisMotionGraphBranch>>,
}

impl BasisGraphBranchOverrides {
    pub fn set_node_branches(
        &mut self,
        graph_lod_id: usize,
        local_basis_id: usize,
        segment: usize,
        branches: Vec<BasisMotionGraphBranch>,
    ) {
        self.branches_by_node.insert(
            BasisGraphNodeKey::new(graph_lod_id, local_basis_id, segment),
            branches,
        );
    }

    pub fn branches_for(
        &self,
        graph_lod_id: usize,
        local_basis_id: usize,
        segment: usize,
    ) -> Option<&[BasisMotionGraphBranch]> {
        self.branches_by_node
            .get(&BasisGraphNodeKey::new(
                graph_lod_id,
                local_basis_id,
                segment,
            ))
            .map(Vec::as_slice)
    }

    pub fn refresh_lods(
        &mut self,
        motion: &BasisBankMotionSet,
        graph: &BasisMotionGraph,
        edited_knots: &[f32],
        graph_lod_ids: &[usize],
    ) -> BasisGraphAuthoringRefreshSummary {
        let mut refreshed_lods: Vec<usize> = graph_lod_ids.iter().copied().collect();
        refreshed_lods.sort_unstable();
        refreshed_lods.dedup();
        let refresh_lods: HashSet<usize> = refreshed_lods.iter().copied().collect();
        self.branches_by_node
            .retain(|key, _| !refresh_lods.contains(&key.graph_lod_id));

        let mut refreshed_node_count = 0;
        let mut changed_best_target_count = 0;
        for graph_lod_id in refreshed_lods.iter().copied() {
            let active_segment_count =
                active_segment_count_for_graph_lod(graph, motion, graph_lod_id);
            let local_basis_ids = local_basis_ids_for_graph_lod(motion, graph, graph_lod_id);
            let usage = normalized_usage_by_local_basis(motion, graph, graph_lod_id);
            for from_local_basis in local_basis_ids.iter().copied() {
                let Some(from_global_basis) = global_basis_id_for_graph_lod(
                    motion.basis_infos.as_slice(),
                    graph,
                    graph_lod_id,
                    from_local_basis,
                ) else {
                    continue;
                };
                for from_segment in 0..active_segment_count {
                    let Some(source) = basis_bank_segment_kinematics(
                        edited_knots,
                        motion.global_basis_count,
                        motion.meta.exported_knot_count,
                        from_global_basis,
                        from_segment,
                        1.0,
                    ) else {
                        continue;
                    };
                    let mut candidates = Vec::new();
                    for to_local_basis in local_basis_ids.iter().copied() {
                        if to_local_basis == from_local_basis {
                            continue;
                        }
                        let Some(to_global_basis) = global_basis_id_for_graph_lod(
                            motion.basis_infos.as_slice(),
                            graph,
                            graph_lod_id,
                            to_local_basis,
                        ) else {
                            continue;
                        };
                        for to_segment in 0..active_segment_count {
                            if let Some(target) = basis_bank_segment_kinematics(
                                edited_knots,
                                motion.global_basis_count,
                                motion.meta.exported_knot_count,
                                to_global_basis,
                                to_segment,
                                0.0,
                            ) {
                                candidates.push(scored_branch(
                                    graph,
                                    from_local_basis,
                                    from_segment,
                                    to_local_basis,
                                    to_segment,
                                    source,
                                    target,
                                    usage.get(to_local_basis).copied().unwrap_or(0.0),
                                    transition_duration_for_node(
                                        graph,
                                        graph_lod_id,
                                        from_local_basis,
                                        from_segment,
                                    ),
                                ));
                            }
                        }
                    }
                    candidates.sort_by(|a, b| {
                        a.score
                            .total_cmp(&b.score)
                            .then_with(|| a.to_basis.cmp(&b.to_basis))
                            .then_with(|| a.to_segment.cmp(&b.to_segment))
                    });
                    for (rank, branch) in candidates.iter_mut().take(graph.branch_top_k).enumerate()
                    {
                        branch.rank = rank;
                    }
                    let refreshed: Vec<_> =
                        candidates.into_iter().take(graph.branch_top_k).collect();
                    if !refreshed.is_empty() {
                        if baseline_best_changed(
                            graph,
                            graph_lod_id,
                            from_local_basis,
                            from_segment,
                            refreshed.first(),
                        ) {
                            changed_best_target_count += 1;
                        }
                        self.set_node_branches(
                            graph_lod_id,
                            from_local_basis,
                            from_segment,
                            refreshed,
                        );
                        refreshed_node_count += 1;
                    }
                }
            }
        }
        BasisGraphAuthoringRefreshSummary {
            refreshed_lods,
            refreshed_node_count,
            changed_best_target_count,
            last_refresh_ms: 0.0,
        }
    }

    pub fn clear(&mut self) {
        self.branches_by_node.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.branches_by_node.is_empty()
    }
}

fn active_segment_count_for_graph_lod(
    graph: &BasisMotionGraph,
    motion: &BasisBankMotionSet,
    graph_lod_id: usize,
) -> usize {
    let exported = motion.meta.exported_knot_count.max(1);
    let source = motion.meta.source_knot_count.min(exported).max(1);
    let graph_uses_closure = graph
        .lods
        .iter()
        .find(|lod| lod.lod_id == graph_lod_id)
        .is_some_and(|lod| {
            lod.branches
                .iter()
                .any(|branch| branch.from_segment >= source || branch.to_segment >= source)
        });
    if graph_uses_closure { exported } else { source }
}

fn local_basis_ids_for_graph_lod(
    motion: &BasisBankMotionSet,
    graph: &BasisMotionGraph,
    graph_lod_id: usize,
) -> Vec<usize> {
    let mut ids: Vec<_> = motion
        .basis_infos
        .iter()
        .filter(|info| graph.graph_lod_id(info.lod_id) == graph_lod_id)
        .map(|info| info.local_basis_id)
        .collect();
    ids.sort_unstable();
    ids.dedup();
    ids
}

fn global_basis_id_for_graph_lod(
    basis_infos: &[BasisInfo],
    graph: &BasisMotionGraph,
    graph_lod_id: usize,
    local_basis_id: usize,
) -> Option<usize> {
    basis_infos.iter().position(|info| {
        graph.graph_lod_id(info.lod_id) == graph_lod_id && info.local_basis_id == local_basis_id
    })
}

fn normalized_usage_by_local_basis(
    motion: &BasisBankMotionSet,
    graph: &BasisMotionGraph,
    graph_lod_id: usize,
) -> Vec<f32> {
    let mut usage = vec![0.0; graph.basis_count];
    for (global_id, info) in motion.basis_infos.iter().enumerate() {
        if graph.graph_lod_id(info.lod_id) != graph_lod_id || info.local_basis_id >= usage.len() {
            continue;
        }
        let value = motion
            .usage_stats
            .get(global_id)
            .map(|stats| stats.sum_abs_weight.max(stats.mean_abs_weight))
            .unwrap_or(0.0);
        if graph.basis_scope == BASIS_SCOPE_SHARED_LOD0 {
            usage[info.local_basis_id] += value;
        } else {
            usage[info.local_basis_id] = value;
        }
    }
    let max_usage = usage.iter().copied().fold(0.0_f32, f32::max);
    if max_usage > 0.0 && max_usage.is_finite() {
        for value in &mut usage {
            *value = (*value / max_usage).clamp(0.0, 1.0);
        }
    }
    usage
}

fn scored_branch(
    graph: &BasisMotionGraph,
    from_basis: usize,
    from_segment: usize,
    to_basis: usize,
    to_segment: usize,
    source: BasisSegmentKinematics,
    target: BasisSegmentKinematics,
    usage_bonus: f32,
    transition_duration: Option<usize>,
) -> BasisMotionGraphBranch {
    let position_cost = norm3(sub3(target.position, source.position));
    let velocity_cost = norm3(sub3(target.velocity, source.velocity));
    let acceleration_cost = norm3(sub3(target.acceleration, source.acceleration));
    let score = graph.score_weights.position * position_cost
        + graph.score_weights.velocity * velocity_cost
        + graph.score_weights.acceleration * acceleration_cost
        - graph.score_weights.usage * usage_bonus;
    BasisMotionGraphBranch {
        from_basis,
        from_segment,
        to_basis,
        to_segment,
        rank: 0,
        score,
        position_cost,
        velocity_cost,
        acceleration_cost,
        usage_bonus,
        transition: transition_duration
            .map(|duration| transition_payload(source, target, duration)),
    }
}

fn transition_duration_for_node(
    graph: &BasisMotionGraph,
    graph_lod_id: usize,
    from_basis: usize,
    from_segment: usize,
) -> Option<usize> {
    if graph.format_version < 2 {
        return None;
    }
    graph
        .lods
        .iter()
        .find(|lod| lod.lod_id == graph_lod_id)?
        .branches
        .iter()
        .find(|branch| branch.from_basis == from_basis && branch.from_segment == from_segment)?
        .transition
        .as_ref()
        .map(|transition| transition.duration_segments.max(1))
        .or(Some(1))
}

fn transition_payload(
    source: BasisSegmentKinematics,
    target: BasisSegmentKinematics,
    duration_segments: usize,
) -> BasisMotionGraphTransition {
    let duration_segments = duration_segments.max(1);
    let duration = duration_segments as f32;
    let mut knots = Vec::with_capacity(duration_segments + 1);
    for index in 0..=duration_segments {
        let t = index as f32 / duration;
        knots.push(quintic_hermite_point(
            source.position,
            scale3(source.velocity, duration),
            scale3(source.acceleration, duration * duration),
            target.position,
            scale3(target.velocity, duration),
            scale3(target.acceleration, duration * duration),
            t,
        ));
    }
    BasisMotionGraphTransition {
        kind: "open_catmull_rom".to_string(),
        duration_segments,
        knots,
        start_tangent: source.velocity,
        end_tangent: target.velocity,
    }
}

fn quintic_hermite_point(
    p0: [f32; 3],
    v0: [f32; 3],
    a0: [f32; 3],
    p1: [f32; 3],
    v1: [f32; 3],
    a1: [f32; 3],
    t: f32,
) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;
    let h00 = 1.0 - 10.0 * t3 + 15.0 * t4 - 6.0 * t5;
    let h10 = t - 6.0 * t3 + 8.0 * t4 - 3.0 * t5;
    let h20 = 0.5 * (t2 - 3.0 * t3 + 3.0 * t4 - t5);
    let h01 = 10.0 * t3 - 15.0 * t4 + 6.0 * t5;
    let h11 = -4.0 * t3 + 7.0 * t4 - 3.0 * t5;
    let h21 = 0.5 * (t3 - 2.0 * t4 + t5);
    let mut out = [0.0; 3];
    for i in 0..3 {
        out[i] = h00 * p0[i] + h10 * v0[i] + h20 * a0[i] + h01 * p1[i] + h11 * v1[i] + h21 * a1[i];
    }
    out
}

fn baseline_best_changed(
    graph: &BasisMotionGraph,
    graph_lod_id: usize,
    from_basis: usize,
    from_segment: usize,
    refreshed_best: Option<&BasisMotionGraphBranch>,
) -> bool {
    let Some(refreshed_best) = refreshed_best else {
        return false;
    };
    let Some(baseline_best) = graph
        .lods
        .iter()
        .find(|lod| lod.lod_id == graph_lod_id)
        .and_then(|lod| {
            lod.branches
                .iter()
                .filter(|branch| {
                    branch.from_basis == from_basis && branch.from_segment == from_segment
                })
                .min_by_key(|branch| branch.rank)
        })
    else {
        return false;
    };
    baseline_best.to_basis != refreshed_best.to_basis
        || baseline_best.to_segment != refreshed_best.to_segment
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale3(v: [f32; 3], scale: f32) -> [f32; 3] {
    [v[0] * scale, v[1] * scale, v[2] * scale]
}

fn norm3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis_bank_motion::{
        BASIS_BANK_FORMAT, BASIS_SCOPE_PER_LOD, BasisBankMotionMeta, BasisBankMotionSet, BasisInfo,
        BasisUsageStats,
    };
    use crate::basis_motion_graph::{
        BasisMotionGraph, BasisMotionGraphBranch, BasisMotionGraphLod,
        BasisMotionGraphScoreWeights, BasisMotionGraphTransition,
    };
    use serde_json::Value;
    use std::sync::Arc;

    fn meta(knot_count: usize) -> BasisBankMotionMeta {
        BasisBankMotionMeta {
            format: BASIS_BANK_FORMAT.to_string(),
            format_version: 1,
            delta_field: "delta_xyz".to_string(),
            basis_scope: BASIS_SCOPE_PER_LOD.to_string(),
            basis_source_lod: None,
            include_lods: vec![0],
            source_knot_count: knot_count,
            exported_knot_count: knot_count,
            loop_closure_knots: 0,
            loop_closure_method: "none".to_string(),
            motion_teacher: "volume".to_string(),
            volume_res: None,
            volume_key_count: None,
            basis_count: 3,
            top_k: 1,
            fit_report_by_lod: Value::Null,
        }
    }

    fn graph(
        format_version: u32,
        transition: Option<BasisMotionGraphTransition>,
    ) -> BasisMotionGraph {
        BasisMotionGraph {
            format: "basis_motion_graph".to_string(),
            format_version,
            basis_scope: BASIS_SCOPE_PER_LOD.to_string(),
            basis_source_lod: None,
            node_unit: "basis_segment".to_string(),
            include_lods: vec![0],
            basis_count: 3,
            knot_count: 4,
            branch_top_k: 2,
            score_weights: BasisMotionGraphScoreWeights {
                position: 1.0,
                velocity: 0.0,
                acceleration: 0.0,
                usage: 0.0,
            },
            lods: vec![BasisMotionGraphLod {
                lod_id: 0,
                branches: vec![BasisMotionGraphBranch {
                    from_basis: 0,
                    from_segment: 0,
                    to_basis: 1,
                    to_segment: 0,
                    rank: 0,
                    score: 0.0,
                    position_cost: 0.0,
                    velocity_cost: 0.0,
                    acceleration_cost: 0.0,
                    usage_bonus: 0.0,
                    transition,
                }],
            }],
        }
    }

    fn motion(knots: Vec<f32>, graph: BasisMotionGraph) -> BasisBankMotionSet {
        BasisBankMotionSet {
            meta: meta(4),
            motion_graph: Some(Arc::new(graph)),
            total_splats: 0,
            global_basis_count: 3,
            basis_infos: vec![
                BasisInfo {
                    lod_id: 0,
                    local_basis_id: 0,
                },
                BasisInfo {
                    lod_id: 0,
                    local_basis_id: 1,
                },
                BasisInfo {
                    lod_id: 0,
                    local_basis_id: 2,
                },
            ],
            usage_stats: vec![BasisUsageStats::default(); 3],
            global_basis_knots: knots,
            global_basis_ids: Vec::new(),
            global_weights: Vec::new(),
        }
    }

    fn flat_knots(points: &[[[f32; 3]; 4]; 3]) -> Vec<f32> {
        points
            .iter()
            .flat_map(|basis| basis.iter())
            .flat_map(|point| point.iter().copied())
            .collect()
    }

    #[test]
    fn refresh_searches_all_same_lod_candidates_and_changes_best_target() {
        let knots = flat_knots(&[
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [100.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
            ],
            [
                [10.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
            ],
        ]);
        let graph = graph(1, None);
        let motion = motion(knots.clone(), graph.clone());
        let mut overrides = BasisGraphBranchOverrides::default();

        let summary = overrides.refresh_lods(&motion, &graph, knots.as_slice(), &[0]);

        let branches = overrides.branches_for(0, 0, 0).unwrap();
        assert_eq!(summary.refreshed_node_count, 12);
        assert_eq!(summary.changed_best_target_count, 1);
        assert_eq!(branches[0].to_basis, 2);
        assert_eq!(branches[0].rank, 0);
        assert_eq!(branches[0].to_segment, 0);
        assert!(
            branches
                .iter()
                .all(|branch| branch.to_basis != branch.from_basis)
        );
    }

    #[test]
    fn refresh_regenerates_v2_transition_endpoints_from_edited_knots() {
        let baseline_transition = BasisMotionGraphTransition {
            kind: "open_catmull_rom".to_string(),
            duration_segments: 3,
            knots: vec![[0.0, 0.0, 0.0]; 4],
            start_tangent: [0.0, 0.0, 0.0],
            end_tangent: [0.0, 0.0, 0.0],
        };
        let knots = flat_knots(&[
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [100.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
            ],
            [
                [10.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
                [12.0, 0.0, 0.0],
                [13.0, 0.0, 0.0],
            ],
        ]);
        let graph = graph(2, Some(baseline_transition));
        let motion = motion(knots.clone(), graph.clone());
        let mut overrides = BasisGraphBranchOverrides::default();

        overrides.refresh_lods(&motion, &graph, knots.as_slice(), &[0]);

        let transition = overrides.branches_for(0, 0, 0).unwrap()[0]
            .transition
            .as_ref()
            .unwrap();
        assert_eq!(transition.duration_segments, 3);
        assert_eq!(transition.knots.first().copied(), Some([10.0, 0.0, 0.0]));
        assert_eq!(transition.knots.last().copied(), Some([10.0, 0.0, 0.0]));
    }
}
