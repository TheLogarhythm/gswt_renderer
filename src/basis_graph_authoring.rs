use std::collections::HashMap;

use crate::basis_bank_motion::{
    BasisBankMotionSet, BasisSegmentKinematics, basis_bank_segment_kinematics,
};
use crate::basis_motion_graph::{
    BasisMotionGraph, BasisMotionGraphBranch, BasisMotionGraphTransition,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct BasisGraphNodeKey {
    pub basis_id: usize,
    pub segment: usize,
}

impl BasisGraphNodeKey {
    pub fn new(basis_id: usize, segment: usize) -> Self {
        Self { basis_id, segment }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BasisGraphAuthoringRefreshSummary {
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
        basis_id: usize,
        segment: usize,
        branches: Vec<BasisMotionGraphBranch>,
    ) {
        self.branches_by_node
            .insert(BasisGraphNodeKey::new(basis_id, segment), branches);
    }

    pub fn branches_for(
        &self,
        basis_id: usize,
        segment: usize,
    ) -> Option<&[BasisMotionGraphBranch]> {
        self.branches_by_node
            .get(&BasisGraphNodeKey::new(basis_id, segment))
            .map(Vec::as_slice)
    }

    pub fn refresh(
        &mut self,
        motion: &BasisBankMotionSet,
        graph: &BasisMotionGraph,
        edited_knots: &[f32],
    ) -> BasisGraphAuthoringRefreshSummary {
        self.branches_by_node.clear();
        let active_segment_count = active_segment_count(graph, motion);
        let usage = normalized_usage(motion, graph.basis_count);
        let mut refreshed_node_count = 0;
        let mut changed_best_target_count = 0;

        for from_basis in 0..motion.basis_count {
            for from_segment in 0..active_segment_count {
                let Some(source) = basis_bank_segment_kinematics(
                    edited_knots,
                    motion.basis_count,
                    motion.meta.exported_knot_count,
                    from_basis,
                    from_segment,
                    1.0,
                ) else {
                    continue;
                };
                let mut candidates = Vec::new();
                for to_basis in 0..motion.basis_count {
                    if to_basis == from_basis {
                        continue;
                    }
                    for to_segment in 0..active_segment_count {
                        if let Some(target) = basis_bank_segment_kinematics(
                            edited_knots,
                            motion.basis_count,
                            motion.meta.exported_knot_count,
                            to_basis,
                            to_segment,
                            0.0,
                        ) {
                            candidates.push(scored_branch(
                                graph,
                                from_basis,
                                from_segment,
                                to_basis,
                                to_segment,
                                source,
                                target,
                                usage.get(to_basis).copied().unwrap_or(0.0),
                                transition_duration_for_node(graph, from_basis, from_segment),
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
                for (rank, branch) in candidates.iter_mut().take(graph.branch_top_k).enumerate() {
                    branch.rank = rank;
                }
                let refreshed: Vec<_> = candidates.into_iter().take(graph.branch_top_k).collect();
                if !refreshed.is_empty() {
                    if baseline_best_changed(graph, from_basis, from_segment, refreshed.first()) {
                        changed_best_target_count += 1;
                    }
                    self.set_node_branches(from_basis, from_segment, refreshed);
                    refreshed_node_count += 1;
                }
            }
        }

        BasisGraphAuthoringRefreshSummary {
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

fn active_segment_count(graph: &BasisMotionGraph, motion: &BasisBankMotionSet) -> usize {
    let exported = motion.meta.exported_knot_count.max(1);
    let source = motion.meta.source_knot_count.min(exported).max(1);
    let graph_uses_closure = graph.lods.first().is_some_and(|lod| {
        lod.branches
            .iter()
            .any(|branch| branch.from_segment >= source || branch.to_segment >= source)
    });
    if graph_uses_closure { exported } else { source }
}

fn normalized_usage(motion: &BasisBankMotionSet, basis_count: usize) -> Vec<f32> {
    let mut usage = vec![0.0; basis_count];
    for (basis_id, value) in usage.iter_mut().enumerate() {
        *value = motion
            .usage_stats
            .get(basis_id)
            .map(|stats| stats.sum_abs_weight.max(stats.mean_abs_weight))
            .unwrap_or(0.0);
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
    from_basis: usize,
    from_segment: usize,
) -> Option<usize> {
    if graph.format_version < 2 {
        return None;
    }
    graph
        .lods
        .first()?
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
    from_basis: usize,
    from_segment: usize,
    refreshed_best: Option<&BasisMotionGraphBranch>,
) -> bool {
    let Some(refreshed_best) = refreshed_best else {
        return false;
    };
    let Some(baseline_best) = graph.lods.first().and_then(|lod| {
        lod.branches
            .iter()
            .filter(|branch| branch.from_basis == from_basis && branch.from_segment == from_segment)
            .min_by_key(|branch| branch.rank)
    }) else {
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
