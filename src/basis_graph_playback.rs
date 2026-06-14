use crate::basis_bank_motion::BasisInfo;
use crate::basis_motion_graph::{
    BasisMotionGraph, BasisMotionGraphBranch, BasisMotionGraphTransition,
};

const MAX_GRAPH_PLAYBACK_DELTA01: f32 = 0.5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BasisGraphPlaybackPolicy {
    Continue,
    Rank0,
    Stochastic,
}

impl BasisGraphPlaybackPolicy {
    pub const ALL: [Self; 3] = [Self::Continue, Self::Rank0, Self::Stochastic];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Continue => "Continue",
            Self::Rank0 => "Rank 0",
            Self::Stochastic => "Stochastic",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BasisGraphPlaybackConfig {
    pub enabled: bool,
    pub policy: BasisGraphPlaybackPolicy,
    pub branch_probability: f32,
    pub temperature: f32,
    pub seed: u32,
    pub blend_duration: f32,
    pub max_branch_score_enabled: bool,
    pub max_branch_score: f32,
    pub min_branch_interval_segments: u32,
    pub max_position_cost_enabled: bool,
    pub max_position_cost: f32,
    pub max_velocity_cost_enabled: bool,
    pub max_velocity_cost: f32,
    pub max_acceleration_cost_enabled: bool,
    pub max_acceleration_cost: f32,
}

impl Default for BasisGraphPlaybackConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            policy: BasisGraphPlaybackPolicy::Continue,
            branch_probability: 0.2,
            temperature: 0.25,
            seed: 1,
            blend_duration: 0.5,
            max_branch_score_enabled: false,
            max_branch_score: 1.0,
            min_branch_interval_segments: 8,
            max_position_cost_enabled: true,
            max_position_cost: 0.75,
            max_velocity_cost_enabled: true,
            max_velocity_cost: 0.75,
            max_acceleration_cost_enabled: true,
            max_acceleration_cost: 0.75,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BasisBranchRejection {
    pub score: bool,
    pub position: bool,
    pub velocity: bool,
    pub acceleration: bool,
}

impl BasisBranchRejection {
    pub fn rejected(self) -> bool {
        self.score || self.position || self.velocity || self.acceleration
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BasisGraphLastEdge {
    Reset,
    Continue,
    Branch {
        rank: usize,
        to_global_basis_id: usize,
        to_segment: usize,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct BasisGraphPlaybackState {
    pub original_global_basis_id: usize,
    pub active_global_basis_id: usize,
    pub lod_id: usize,
    pub local_basis_id: usize,
    pub segment: usize,
    pub segment_phase: f32,
    pub blend_from_global_basis_id: usize,
    pub blend_from_segment: usize,
    pub blend_phase: f32,
    pub blend_weight: f32,
    pub blend_active: bool,
    pub transition_active: bool,
    pub transition_phase_segments: f32,
    pub transition_duration_segments: f32,
    pub transition_delta: [f32; 3],
    pub transition: Option<BasisMotionGraphTransition>,
    pub transition_target_global_basis_id: usize,
    pub transition_target_local_basis_id: usize,
    pub transition_target_segment: usize,
    pub rejected_branch_count: usize,
    pub rejected_branch_score_count: usize,
    pub rejected_branch_position_count: usize,
    pub rejected_branch_velocity_count: usize,
    pub rejected_branch_acceleration_count: usize,
    pub segments_since_branch: u32,
    pub last_edge: BasisGraphLastEdge,
}

pub struct BasisGraphPlaybackController {
    states: Vec<BasisGraphPlaybackState>,
    previous_time01: Option<f32>,
    rng: BasisGraphPlaybackRng,
    last_config: BasisGraphPlaybackConfig,
    initialized: bool,
}

impl BasisGraphPlaybackController {
    pub fn new(global_basis_count: usize) -> Self {
        Self {
            states: Vec::with_capacity(global_basis_count),
            previous_time01: None,
            rng: BasisGraphPlaybackRng::new(BasisGraphPlaybackConfig::default().seed),
            last_config: BasisGraphPlaybackConfig::default(),
            initialized: false,
        }
    }

    pub fn states(&self) -> &[BasisGraphPlaybackState] {
        self.states.as_slice()
    }

    #[cfg(test)]
    pub fn reset(&mut self, time01: f32, graph: &BasisMotionGraph, basis_infos: &[BasisInfo]) {
        self.reset_with_config(
            time01,
            graph,
            basis_infos,
            BasisGraphPlaybackConfig::default(),
        );
    }

    pub fn reset_with_config(
        &mut self,
        time01: f32,
        graph: &BasisMotionGraph,
        basis_infos: &[BasisInfo],
        config: BasisGraphPlaybackConfig,
    ) {
        self.states.clear();
        let (segment, segment_phase) = segment_and_phase(time01, graph.knot_count);
        for (global_basis_id, info) in basis_infos.iter().enumerate() {
            self.states.push(BasisGraphPlaybackState {
                original_global_basis_id: global_basis_id,
                active_global_basis_id: global_basis_id,
                lod_id: info.lod_id,
                local_basis_id: info.local_basis_id,
                segment,
                segment_phase,
                blend_from_global_basis_id: global_basis_id,
                blend_from_segment: segment,
                blend_phase: 1.0,
                blend_weight: 1.0,
                blend_active: false,
                transition_active: false,
                transition_phase_segments: 0.0,
                transition_duration_segments: 0.0,
                transition_delta: [0.0; 3],
                transition: None,
                transition_target_global_basis_id: global_basis_id,
                transition_target_local_basis_id: info.local_basis_id,
                transition_target_segment: segment,
                rejected_branch_count: 0,
                rejected_branch_score_count: 0,
                rejected_branch_position_count: 0,
                rejected_branch_velocity_count: 0,
                rejected_branch_acceleration_count: 0,
                segments_since_branch: config.min_branch_interval_segments,
                last_edge: BasisGraphLastEdge::Reset,
            });
        }
        self.previous_time01 = Some(time01.rem_euclid(1.0));
        self.rng = BasisGraphPlaybackRng::new(config.seed);
        self.last_config = config;
        self.initialized = true;
    }

    pub fn advance(
        &mut self,
        time01: f32,
        graph: &BasisMotionGraph,
        basis_infos: &[BasisInfo],
        config: BasisGraphPlaybackConfig,
    ) {
        if !config.enabled {
            self.previous_time01 = Some(time01.rem_euclid(1.0));
            return;
        }
        if !self.initialized
            || self.states.len() != basis_infos.len()
            || config.policy != self.last_config.policy
            || config.seed != self.last_config.seed
            || config.min_branch_interval_segments != self.last_config.min_branch_interval_segments
            || config.max_position_cost_enabled != self.last_config.max_position_cost_enabled
            || config.max_position_cost != self.last_config.max_position_cost
            || config.max_velocity_cost_enabled != self.last_config.max_velocity_cost_enabled
            || config.max_velocity_cost != self.last_config.max_velocity_cost
            || config.max_acceleration_cost_enabled
                != self.last_config.max_acceleration_cost_enabled
            || config.max_acceleration_cost != self.last_config.max_acceleration_cost
        {
            self.reset_with_config(time01, graph, basis_infos, config);
            return;
        }

        let time01 = time01.rem_euclid(1.0);
        let Some(previous_time01) = self.previous_time01 else {
            self.reset_with_config(time01, graph, basis_infos, config);
            return;
        };
        let delta01 = wrapped_forward_delta01(previous_time01, time01);
        if delta01 > MAX_GRAPH_PLAYBACK_DELTA01 {
            self.reset_with_config(time01, graph, basis_infos, config);
            return;
        }

        let mut segment_delta = delta01 * graph.knot_count as f32;
        for state in &mut self.states {
            advance_state(
                state,
                &mut segment_delta,
                graph,
                basis_infos,
                config,
                &mut self.rng,
            );
            segment_delta = delta01 * graph.knot_count as f32;
        }
        self.previous_time01 = Some(time01);
        self.last_config = config;
    }
}

pub fn pack_basis_graph_blend_overrides(
    states: Option<&[BasisGraphPlaybackState]>,
    global_basis_count: usize,
) -> Vec<[f32; 4]> {
    let mut packed = vec![[0.0, 0.0, 0.0, 1.0]; global_basis_count];
    let Some(states) = states else {
        return packed;
    };
    for state in states {
        if state.original_global_basis_id < packed.len() {
            let from_basis = if state.blend_active {
                state.blend_from_global_basis_id
            } else {
                state.active_global_basis_id
            };
            let from_segment = if state.blend_active {
                state.blend_from_segment
            } else {
                state.segment
            };
            packed[state.original_global_basis_id] = [
                from_basis as f32,
                from_segment as f32,
                state.segment_phase.clamp(0.0, 1.0),
                state.blend_weight.clamp(0.0, 1.0),
            ];
        }
    }
    packed
}

pub fn pack_basis_graph_direct_overrides(
    states: Option<&[BasisGraphPlaybackState]>,
    global_basis_count: usize,
) -> Vec<[f32; 4]> {
    let mut packed = vec![[0.0, 0.0, 0.0, 0.0]; global_basis_count];
    let Some(states) = states else {
        return packed;
    };
    for state in states {
        if state.original_global_basis_id < packed.len() && state.transition_active {
            packed[state.original_global_basis_id] = [
                state.transition_delta[0],
                state.transition_delta[1],
                state.transition_delta[2],
                1.0,
            ];
        }
    }
    packed
}

pub fn pack_basis_graph_sample_overrides(
    states: Option<&[BasisGraphPlaybackState]>,
    global_basis_count: usize,
) -> Vec<[f32; 4]> {
    let mut packed = vec![[0.0, 0.0, 0.0, 0.0]; global_basis_count];
    let Some(states) = states else {
        return packed;
    };
    for state in states {
        if state.original_global_basis_id < packed.len() {
            packed[state.original_global_basis_id] = [
                state.active_global_basis_id as f32,
                state.segment as f32,
                state.segment_phase.clamp(0.0, 1.0),
                1.0,
            ];
        }
    }
    packed
}

#[cfg(test)]
pub fn explicit_segment_time01(knot_count: usize, segment: usize, segment_phase: f32) -> f32 {
    if knot_count == 0 {
        return 0.0;
    }
    ((segment % knot_count) as f32 + segment_phase.clamp(0.0, 1.0)) / knot_count as f32
}

pub fn branch_softmax_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }
    let temperature = temperature.max(1e-4);
    let max_logit = scores
        .iter()
        .map(|score| -score / temperature)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut weights: Vec<f32> = scores
        .iter()
        .map(|score| ((-score / temperature) - max_logit).exp())
        .collect();
    let sum: f32 = weights.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        let uniform = 1.0 / scores.len() as f32;
        weights.fill(uniform);
        return weights;
    }
    for weight in &mut weights {
        *weight /= sum;
    }
    weights
}

fn advance_state(
    state: &mut BasisGraphPlaybackState,
    segment_delta: &mut f32,
    graph: &BasisMotionGraph,
    basis_infos: &[BasisInfo],
    config: BasisGraphPlaybackConfig,
    rng: &mut BasisGraphPlaybackRng,
) {
    while *segment_delta > 0.0 {
        if state.transition_active {
            advance_transition_state(state, segment_delta);
            continue;
        }

        let to_boundary = 1.0 - state.segment_phase;
        if *segment_delta < to_boundary {
            state.segment_phase += *segment_delta;
            update_blend(state, config);
            break;
        }
        state.segment_phase = 1.0;
        update_blend(state, config);
        *segment_delta -= to_boundary;
        choose_next_edge(state, graph, basis_infos, config, rng);
        state.segment_phase = 0.0;
        update_blend(state, config);
    }
}

fn advance_transition_state(state: &mut BasisGraphPlaybackState, segment_delta: &mut f32) {
    let remaining = (state.transition_duration_segments - state.transition_phase_segments).max(0.0);
    if *segment_delta < remaining {
        state.transition_phase_segments += *segment_delta;
        update_transition_delta(state);
        *segment_delta = 0.0;
        return;
    }

    *segment_delta -= remaining;
    state.transition_phase_segments = state.transition_duration_segments;
    update_transition_delta(state);
    state.transition_active = false;
    state.transition = None;
    state.active_global_basis_id = state.transition_target_global_basis_id;
    state.local_basis_id = state.transition_target_local_basis_id;
    state.segment = state.transition_target_segment;
    state.segment_phase = 0.0;
    state.blend_active = false;
    state.blend_phase = 1.0;
    state.blend_weight = 1.0;
}

fn choose_next_edge(
    state: &mut BasisGraphPlaybackState,
    graph: &BasisMotionGraph,
    basis_infos: &[BasisInfo],
    config: BasisGraphPlaybackConfig,
    rng: &mut BasisGraphPlaybackRng,
) {
    let branches = graph.branches_for(state.lod_id, state.local_basis_id, state.segment);
    let branch_cooldown_active = state.segments_since_branch < config.min_branch_interval_segments;
    let branch_selection_blocked = state.blend_active
        || branch_cooldown_active
        || config.policy == BasisGraphPlaybackPolicy::Continue;
    let mut filtered_branches = Vec::new();
    state.rejected_branch_count = 0;
    state.rejected_branch_score_count = 0;
    state.rejected_branch_position_count = 0;
    state.rejected_branch_velocity_count = 0;
    state.rejected_branch_acceleration_count = 0;
    if !branch_selection_blocked {
        for branch in branches {
            let rejection = branch_rejection(branch, config);
            if rejection.rejected() {
                state.rejected_branch_count += 1;
                if rejection.score {
                    state.rejected_branch_score_count += 1;
                }
                if rejection.position {
                    state.rejected_branch_position_count += 1;
                }
                if rejection.velocity {
                    state.rejected_branch_velocity_count += 1;
                }
                if rejection.acceleration {
                    state.rejected_branch_acceleration_count += 1;
                }
            } else {
                filtered_branches.push(branch);
            }
        }
    }
    let branch = match config.policy {
        BasisGraphPlaybackPolicy::Continue => None,
        BasisGraphPlaybackPolicy::Rank0 => {
            if state.blend_active || branch_cooldown_active {
                None
            } else {
                filtered_branches.first().copied()
            }
        }
        BasisGraphPlaybackPolicy::Stochastic => {
            if state.blend_active
                || branch_cooldown_active
                || filtered_branches.is_empty()
                || rng.next_f32() > config.branch_probability.clamp(0.0, 1.0)
            {
                None
            } else {
                choose_stochastic_branch(&filtered_branches, config.temperature, rng)
            }
        }
    };

    if let Some(branch) = branch {
        apply_branch(state, branch, graph.knot_count, basis_infos, config);
    } else {
        state.segment = (state.segment + 1) % graph.knot_count;
        state.segments_since_branch = state.segments_since_branch.saturating_add(1);
        state.last_edge = BasisGraphLastEdge::Continue;
    }
}

pub fn branch_rejection(
    branch: &BasisMotionGraphBranch,
    config: BasisGraphPlaybackConfig,
) -> BasisBranchRejection {
    BasisBranchRejection {
        score: config.max_branch_score_enabled && branch.score > config.max_branch_score,
        position: config.max_position_cost_enabled
            && branch.position_cost > config.max_position_cost,
        velocity: config.max_velocity_cost_enabled
            && branch.velocity_cost > config.max_velocity_cost,
        acceleration: config.max_acceleration_cost_enabled
            && branch.acceleration_cost > config.max_acceleration_cost,
    }
}

fn apply_branch(
    state: &mut BasisGraphPlaybackState,
    branch: &BasisMotionGraphBranch,
    knot_count: usize,
    basis_infos: &[BasisInfo],
    config: BasisGraphPlaybackConfig,
) {
    let Some(target_global_basis_id) = branch.target_global_basis_id(state.lod_id, basis_infos)
    else {
        state.last_edge = BasisGraphLastEdge::Continue;
        return;
    };
    if let Some(transition) = branch.transition.as_ref() {
        state.transition_active = true;
        state.transition_phase_segments = 0.0;
        state.transition_duration_segments = transition.duration_segments as f32;
        state.transition_delta = sample_transition_delta(transition, 0.0);
        state.transition = Some(transition.clone());
        state.transition_target_global_basis_id = target_global_basis_id;
        state.transition_target_local_basis_id = branch.to_basis;
        state.transition_target_segment = branch.to_segment;
        state.blend_active = false;
        state.blend_phase = 1.0;
        state.blend_weight = 1.0;
        state.segments_since_branch = 0;
        state.last_edge = BasisGraphLastEdge::Branch {
            rank: branch.rank,
            to_global_basis_id: target_global_basis_id,
            to_segment: branch.to_segment,
        };
        return;
    }

    let source_global_basis_id = state.active_global_basis_id;
    let source_default_segment = if knot_count == 0 {
        state.segment
    } else {
        (state.segment + 1) % knot_count
    };
    state.active_global_basis_id = target_global_basis_id;
    state.local_basis_id = branch.to_basis;
    state.segment = branch.to_segment;
    state.blend_from_global_basis_id = source_global_basis_id;
    state.blend_from_segment = source_default_segment;
    state.blend_phase = 0.0;
    state.blend_weight = if config.blend_duration <= 0.0 {
        1.0
    } else {
        0.0
    };
    state.blend_active = config.blend_duration > 0.0;
    state.segments_since_branch = 0;
    state.last_edge = BasisGraphLastEdge::Branch {
        rank: branch.rank,
        to_global_basis_id: target_global_basis_id,
        to_segment: branch.to_segment,
    };
}

fn update_transition_delta(state: &mut BasisGraphPlaybackState) {
    if !state.transition_active {
        return;
    }
    if let Some(transition) = state.transition.as_ref() {
        state.transition_delta =
            sample_transition_delta(transition, state.transition_phase_segments);
    }
}

pub fn sample_transition_delta(
    transition: &BasisMotionGraphTransition,
    phase_segments: f32,
) -> [f32; 3] {
    sample_transition_state(transition, phase_segments).0
}

#[cfg(test)]
pub fn sample_transition_velocity(
    transition: &BasisMotionGraphTransition,
    phase_segments: f32,
) -> [f32; 3] {
    sample_transition_state(transition, phase_segments).1
}

fn sample_transition_state(
    transition: &BasisMotionGraphTransition,
    phase_segments: f32,
) -> ([f32; 3], [f32; 3]) {
    let duration = transition.duration_segments.max(1);
    if transition.knots.len() < duration + 1 {
        return ([0.0; 3], [0.0; 3]);
    }
    let phase = phase_segments.clamp(0.0, duration as f32);
    let segment = if phase >= duration as f32 {
        duration - 1
    } else {
        phase.floor() as usize
    };
    let u = if phase >= duration as f32 {
        1.0
    } else {
        phase - segment as f32
    };
    let p0 = transition.knots[segment];
    let p1 = transition.knots[segment + 1];
    let m0 = transition_tangent_at(transition, segment);
    let m1 = transition_tangent_at(transition, segment + 1);
    cubic_hermite_state(p0, m0, p1, m1, u)
}

fn transition_tangent_at(transition: &BasisMotionGraphTransition, knot_index: usize) -> [f32; 3] {
    if knot_index == 0 {
        return transition.start_tangent;
    }
    if knot_index + 1 == transition.knots.len() {
        return transition.end_tangent;
    }
    let prev = transition.knots[knot_index - 1];
    let next = transition.knots[knot_index + 1];
    scale3(sub3(next, prev), 0.5)
}

fn cubic_hermite_state(
    p0: [f32; 3],
    m0: [f32; 3],
    p1: [f32; 3],
    m1: [f32; 3],
    u: f32,
) -> ([f32; 3], [f32; 3]) {
    let u = u.clamp(0.0, 1.0);
    let u2 = u * u;
    let u3 = u2 * u;
    let h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
    let h10 = u3 - 2.0 * u2 + u;
    let h01 = -2.0 * u3 + 3.0 * u2;
    let h11 = u3 - u2;
    let dh00 = 6.0 * u2 - 6.0 * u;
    let dh10 = 3.0 * u2 - 4.0 * u + 1.0;
    let dh01 = -6.0 * u2 + 6.0 * u;
    let dh11 = 3.0 * u2 - 2.0 * u;
    let mut position = [0.0; 3];
    let mut velocity = [0.0; 3];
    for c in 0..3 {
        position[c] = h00 * p0[c] + h10 * m0[c] + h01 * p1[c] + h11 * m1[c];
        velocity[c] = dh00 * p0[c] + dh10 * m0[c] + dh01 * p1[c] + dh11 * m1[c];
    }
    (position, velocity)
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale3(v: [f32; 3], scale: f32) -> [f32; 3] {
    [v[0] * scale, v[1] * scale, v[2] * scale]
}

fn update_blend(state: &mut BasisGraphPlaybackState, config: BasisGraphPlaybackConfig) {
    if !state.blend_active {
        state.blend_phase = 1.0;
        state.blend_weight = 1.0;
        return;
    }

    let blend_duration = config.blend_duration.clamp(0.0, 1.0);
    if blend_duration <= 0.0 {
        state.blend_active = false;
        state.blend_phase = 1.0;
        state.blend_weight = 1.0;
        return;
    }

    state.blend_phase = (state.segment_phase / blend_duration).clamp(0.0, 1.0);
    state.blend_weight = smoothstep01(state.blend_phase);
    if state.blend_phase >= 1.0 {
        state.blend_active = false;
        state.blend_weight = 1.0;
    }
}

fn smoothstep01(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn choose_stochastic_branch<'a>(
    branches: &'a [&'a BasisMotionGraphBranch],
    temperature: f32,
    rng: &mut BasisGraphPlaybackRng,
) -> Option<&'a BasisMotionGraphBranch> {
    let scores: Vec<f32> = branches.iter().map(|branch| branch.score).collect();
    let weights = branch_softmax_weights(scores.as_slice(), temperature);
    let mut draw = rng.next_f32();
    for (branch, weight) in branches.iter().zip(weights.iter()) {
        if draw <= *weight {
            return Some(*branch);
        }
        draw -= *weight;
    }
    branches.last().copied()
}

fn segment_and_phase(time01: f32, knot_count: usize) -> (usize, f32) {
    if knot_count == 0 {
        return (0, 0.0);
    }
    let scaled = time01.rem_euclid(1.0) * knot_count as f32;
    let segment = scaled.floor() as usize % knot_count;
    (segment, scaled - segment as f32)
}

fn wrapped_forward_delta01(previous: f32, current: f32) -> f32 {
    (current - previous).rem_euclid(1.0)
}

#[derive(Clone, Copy)]
struct BasisGraphPlaybackRng {
    state: u32,
}

impl BasisGraphPlaybackRng {
    fn new(seed: u32) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_f32(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        ((self.state >> 8) as f32) / ((u32::MAX >> 8) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis_bank_motion::{BasisInfo, basis_bank_delta};
    use crate::basis_motion_graph::{
        BasisMotionGraph, BasisMotionGraphBranch, BasisMotionGraphLod, BasisMotionGraphScoreWeights,
    };

    fn test_graph() -> BasisMotionGraph {
        BasisMotionGraph {
            format: "basis_motion_graph".to_string(),
            format_version: 1,
            basis_scope: "per_lod".to_string(),
            node_unit: "basis_segment".to_string(),
            include_lods: vec![0],
            basis_count: 3,
            knot_count: 4,
            branch_top_k: 3,
            score_weights: BasisMotionGraphScoreWeights {
                position: 1.0,
                velocity: 1.0,
                acceleration: 0.5,
                usage: 0.25,
            },
            lods: vec![BasisMotionGraphLod {
                lod_id: 0,
                branches: vec![
                    BasisMotionGraphBranch {
                        from_basis: 0,
                        from_segment: 1,
                        to_basis: 2,
                        to_segment: 1,
                        rank: 0,
                        score: 0.05,
                        position_cost: 0.05,
                        velocity_cost: 0.0,
                        acceleration_cost: 0.0,
                        usage_bonus: 0.0,
                        transition: None,
                    },
                    BasisMotionGraphBranch {
                        from_basis: 0,
                        from_segment: 0,
                        to_basis: 1,
                        to_segment: 2,
                        rank: 0,
                        score: 0.1,
                        position_cost: 0.1,
                        velocity_cost: 0.0,
                        acceleration_cost: 0.0,
                        usage_bonus: 0.0,
                        transition: None,
                    },
                    BasisMotionGraphBranch {
                        from_basis: 0,
                        from_segment: 0,
                        to_basis: 2,
                        to_segment: 3,
                        rank: 1,
                        score: 1.0,
                        position_cost: 1.0,
                        velocity_cost: 0.0,
                        acceleration_cost: 0.0,
                        usage_bonus: 0.0,
                        transition: None,
                    },
                ],
            }],
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
                lod_id: 0,
                local_basis_id: 2,
            },
        ]
    }

    fn test_transition() -> crate::basis_motion_graph::BasisMotionGraphTransition {
        crate::basis_motion_graph::BasisMotionGraphTransition {
            kind: "open_catmull_rom".to_string(),
            duration_segments: 3,
            knots: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [3.0, 1.0, 0.0],
            ],
            start_tangent: [1.0, 0.0, 0.0],
            end_tangent: [1.0, 0.0, 0.0],
        }
    }

    #[test]
    fn reset_maps_each_global_basis_to_current_segment_and_phase() {
        let mut playback = BasisGraphPlaybackController::new(3);
        playback.reset(0.375, &test_graph(), &basis_infos());

        let states = playback.states();
        assert_eq!(states.len(), 3);
        assert_eq!(states[0].original_global_basis_id, 0);
        assert_eq!(states[0].active_global_basis_id, 0);
        assert_eq!(states[0].lod_id, 0);
        assert_eq!(states[0].local_basis_id, 0);
        assert_eq!(states[0].segment, 1);
        assert!((states[0].segment_phase - 0.5).abs() < 1e-6);
    }

    #[test]
    fn default_config_uses_tuned_branch_smoothing_and_quality_gates() {
        let config = BasisGraphPlaybackConfig::default();

        assert_eq!(config.blend_duration, 0.5);
        assert!(config.max_position_cost_enabled);
        assert_eq!(config.max_position_cost, 0.75);
        assert!(config.max_velocity_cost_enabled);
        assert_eq!(config.max_velocity_cost, 0.75);
        assert!(config.max_acceleration_cost_enabled);
        assert_eq!(config.max_acceleration_cost, 0.75);
    }

    #[test]
    fn continue_policy_advances_to_implicit_successor() {
        let mut playback = BasisGraphPlaybackController::new(3);
        playback.reset(0.0, &test_graph(), &basis_infos());

        playback.advance(
            0.26,
            &test_graph(),
            &basis_infos(),
            BasisGraphPlaybackConfig {
                enabled: true,
                policy: BasisGraphPlaybackPolicy::Continue,
                branch_probability: 0.2,
                temperature: 0.25,
                seed: 1,
                ..BasisGraphPlaybackConfig::default()
            },
        );

        let state = &playback.states()[0];
        assert_eq!(state.active_global_basis_id, 0);
        assert_eq!(state.segment, 1);
        assert!(matches!(state.last_edge, BasisGraphLastEdge::Continue));
    }

    #[test]
    fn rank_zero_policy_takes_best_branch_at_boundary() {
        let mut playback = BasisGraphPlaybackController::new(3);
        let config = BasisGraphPlaybackConfig {
            enabled: true,
            policy: BasisGraphPlaybackPolicy::Rank0,
            branch_probability: 0.2,
            temperature: 0.25,
            seed: 1,
            ..BasisGraphPlaybackConfig::default()
        };
        playback.reset_with_config(0.0, &test_graph(), &basis_infos(), config);

        playback.advance(0.26, &test_graph(), &basis_infos(), config);

        let state = &playback.states()[0];
        assert_eq!(state.active_global_basis_id, 1);
        assert_eq!(state.local_basis_id, 1);
        assert_eq!(state.segment, 2);
        assert!(matches!(
            state.last_edge,
            BasisGraphLastEdge::Branch {
                rank: 0,
                to_global_basis_id: 1,
                to_segment: 2
            }
        ));
    }

    #[test]
    fn branch_creation_starts_blend_from_source_default_successor() {
        let mut playback = BasisGraphPlaybackController::new(3);
        let config = BasisGraphPlaybackConfig {
            enabled: true,
            policy: BasisGraphPlaybackPolicy::Rank0,
            branch_probability: 0.2,
            temperature: 0.25,
            seed: 1,
            blend_duration: 0.25,
            max_branch_score_enabled: false,
            max_branch_score: 1.0,
            min_branch_interval_segments: 0,
            max_position_cost_enabled: false,
            max_position_cost: 1.0,
            max_velocity_cost_enabled: false,
            max_velocity_cost: 1.0,
            max_acceleration_cost_enabled: false,
            max_acceleration_cost: 1.0,
        };
        playback.reset_with_config(0.0, &test_graph(), &basis_infos(), config);

        playback.advance(0.25, &test_graph(), &basis_infos(), config);

        let state = &playback.states()[0];
        assert!(state.blend_active);
        assert_eq!(state.blend_from_global_basis_id, 0);
        assert_eq!(state.blend_from_segment, 1);
        assert_eq!(state.blend_phase, 0.0);
        assert_eq!(state.blend_weight, 0.0);
    }

    #[test]
    fn blend_weight_increases_with_phase_and_reaches_one() {
        let mut playback = BasisGraphPlaybackController::new(3);
        let config = BasisGraphPlaybackConfig {
            enabled: true,
            policy: BasisGraphPlaybackPolicy::Rank0,
            branch_probability: 0.2,
            temperature: 0.25,
            seed: 1,
            blend_duration: 0.25,
            max_branch_score_enabled: false,
            max_branch_score: 1.0,
            min_branch_interval_segments: 0,
            max_position_cost_enabled: false,
            max_position_cost: 1.0,
            max_velocity_cost_enabled: false,
            max_velocity_cost: 1.0,
            max_acceleration_cost_enabled: false,
            max_acceleration_cost: 1.0,
        };
        playback.reset_with_config(0.0, &test_graph(), &basis_infos(), config);
        playback.advance(0.26, &test_graph(), &basis_infos(), config);
        playback.advance(0.285, &test_graph(), &basis_infos(), config);
        let mid = playback.states()[0].blend_weight;
        assert!(mid > 0.0 && mid < 1.0);

        playback.advance(0.325, &test_graph(), &basis_infos(), config);
        let state = &playback.states()[0];
        assert!(!state.blend_active);
        assert_eq!(state.blend_weight, 1.0);
    }

    #[test]
    fn blend_duration_zero_matches_hard_switch_behavior() {
        let mut playback = BasisGraphPlaybackController::new(3);
        let config = BasisGraphPlaybackConfig {
            enabled: true,
            policy: BasisGraphPlaybackPolicy::Rank0,
            branch_probability: 0.2,
            temperature: 0.25,
            seed: 1,
            blend_duration: 0.0,
            max_branch_score_enabled: false,
            max_branch_score: 1.0,
            min_branch_interval_segments: 0,
            max_position_cost_enabled: false,
            max_position_cost: 1.0,
            max_velocity_cost_enabled: false,
            max_velocity_cost: 1.0,
            max_acceleration_cost_enabled: false,
            max_acceleration_cost: 1.0,
        };
        playback.reset_with_config(0.0, &test_graph(), &basis_infos(), config);

        playback.advance(0.26, &test_graph(), &basis_infos(), config);

        let state = &playback.states()[0];
        assert!(!state.blend_active);
        assert_eq!(state.blend_weight, 1.0);
        assert_eq!(state.active_global_basis_id, 1);
    }

    #[test]
    fn active_blend_defers_new_branch_at_next_boundary() {
        let mut playback = BasisGraphPlaybackController::new(3);
        let config = BasisGraphPlaybackConfig {
            enabled: true,
            policy: BasisGraphPlaybackPolicy::Rank0,
            branch_probability: 0.2,
            temperature: 0.25,
            seed: 1,
            blend_duration: 1.0,
            max_branch_score_enabled: false,
            max_branch_score: 1.0,
            min_branch_interval_segments: 0,
            max_position_cost_enabled: false,
            max_position_cost: 1.0,
            max_velocity_cost_enabled: false,
            max_velocity_cost: 1.0,
            max_acceleration_cost_enabled: false,
            max_acceleration_cost: 1.0,
        };
        playback.reset_with_config(0.0, &test_graph(), &basis_infos(), config);
        playback.advance(0.26, &test_graph(), &basis_infos(), config);
        playback.advance(0.51, &test_graph(), &basis_infos(), config);

        let state = &playback.states()[0];
        assert_eq!(state.active_global_basis_id, 1);
        assert_eq!(state.segment, 3);
        assert!(matches!(state.last_edge, BasisGraphLastEdge::Continue));
    }

    #[test]
    fn rank_zero_uses_best_branch_below_score_threshold() {
        let mut playback = BasisGraphPlaybackController::new(3);
        let config = BasisGraphPlaybackConfig {
            enabled: true,
            policy: BasisGraphPlaybackPolicy::Rank0,
            branch_probability: 0.2,
            temperature: 0.25,
            seed: 1,
            blend_duration: 0.0,
            max_branch_score_enabled: true,
            max_branch_score: 0.2,
            min_branch_interval_segments: 0,
            max_position_cost_enabled: false,
            max_position_cost: 1.0,
            max_velocity_cost_enabled: false,
            max_velocity_cost: 1.0,
            max_acceleration_cost_enabled: false,
            max_acceleration_cost: 1.0,
        };
        playback.reset_with_config(0.0, &test_graph(), &basis_infos(), config);

        playback.advance(0.26, &test_graph(), &basis_infos(), config);

        assert_eq!(playback.states()[0].active_global_basis_id, 1);
        assert_eq!(playback.states()[0].rejected_branch_count, 1);
    }

    #[test]
    fn all_over_threshold_candidates_fall_back_to_continue() {
        let mut playback = BasisGraphPlaybackController::new(3);
        let config = BasisGraphPlaybackConfig {
            enabled: true,
            policy: BasisGraphPlaybackPolicy::Rank0,
            branch_probability: 0.2,
            temperature: 0.25,
            seed: 1,
            blend_duration: 0.0,
            max_branch_score_enabled: true,
            max_branch_score: 0.01,
            min_branch_interval_segments: 0,
            max_position_cost_enabled: false,
            max_position_cost: 1.0,
            max_velocity_cost_enabled: false,
            max_velocity_cost: 1.0,
            max_acceleration_cost_enabled: false,
            max_acceleration_cost: 1.0,
        };
        playback.reset_with_config(0.0, &test_graph(), &basis_infos(), config);

        playback.advance(0.26, &test_graph(), &basis_infos(), config);

        let state = &playback.states()[0];
        assert_eq!(state.active_global_basis_id, 0);
        assert_eq!(state.segment, 1);
        assert_eq!(state.rejected_branch_count, 2);
        assert_eq!(state.rejected_branch_score_count, 2);
        assert!(matches!(state.last_edge, BasisGraphLastEdge::Continue));
    }

    #[test]
    fn stochastic_samples_only_branches_below_score_threshold() {
        let mut playback = BasisGraphPlaybackController::new(3);
        let config = BasisGraphPlaybackConfig {
            enabled: true,
            policy: BasisGraphPlaybackPolicy::Stochastic,
            branch_probability: 1.0,
            temperature: 10.0,
            seed: 7,
            blend_duration: 0.0,
            max_branch_score_enabled: true,
            max_branch_score: 0.2,
            min_branch_interval_segments: 0,
            max_position_cost_enabled: false,
            max_position_cost: 1.0,
            max_velocity_cost_enabled: false,
            max_velocity_cost: 1.0,
            max_acceleration_cost_enabled: false,
            max_acceleration_cost: 1.0,
        };
        playback.reset_with_config(0.0, &test_graph(), &basis_infos(), config);

        playback.advance(0.26, &test_graph(), &basis_infos(), config);

        let state = &playback.states()[0];
        assert_eq!(state.active_global_basis_id, 1);
        assert_eq!(state.rejected_branch_count, 1);
        assert_eq!(state.rejected_branch_score_count, 1);
    }

    fn quality_graph() -> BasisMotionGraph {
        let mut graph = test_graph();
        graph.lods[0].branches = vec![
            BasisMotionGraphBranch {
                from_basis: 0,
                from_segment: 0,
                to_basis: 1,
                to_segment: 1,
                rank: 0,
                score: 0.1,
                position_cost: 0.9,
                velocity_cost: 0.1,
                acceleration_cost: 0.1,
                usage_bonus: 0.0,
                transition: None,
            },
            BasisMotionGraphBranch {
                from_basis: 0,
                from_segment: 0,
                to_basis: 2,
                to_segment: 2,
                rank: 1,
                score: 0.2,
                position_cost: 0.1,
                velocity_cost: 0.8,
                acceleration_cost: 0.1,
                usage_bonus: 0.0,
                transition: None,
            },
            BasisMotionGraphBranch {
                from_basis: 0,
                from_segment: 0,
                to_basis: 2,
                to_segment: 3,
                rank: 2,
                score: 0.3,
                position_cost: 0.1,
                velocity_cost: 0.1,
                acceleration_cost: 0.7,
                usage_bonus: 0.0,
                transition: None,
            },
        ];
        graph
    }

    #[test]
    fn disabled_term_gates_preserve_current_branch_selection() {
        let graph = quality_graph();
        let mut playback = BasisGraphPlaybackController::new(3);
        let config = BasisGraphPlaybackConfig {
            enabled: true,
            policy: BasisGraphPlaybackPolicy::Rank0,
            blend_duration: 0.0,
            min_branch_interval_segments: 0,
            max_position_cost_enabled: false,
            max_velocity_cost_enabled: false,
            max_acceleration_cost_enabled: false,
            ..BasisGraphPlaybackConfig::default()
        };
        playback.reset_with_config(0.0, &graph, &basis_infos(), config);

        playback.advance(0.25, &graph, &basis_infos(), config);

        let state = &playback.states()[0];
        assert_eq!(state.active_global_basis_id, 1);
        assert_eq!(state.rejected_branch_count, 0);
        assert_eq!(state.rejected_branch_position_count, 0);
        assert_eq!(state.rejected_branch_velocity_count, 0);
        assert_eq!(state.rejected_branch_acceleration_count, 0);
    }

    #[test]
    fn position_gate_rejects_high_position_branches_before_rank_zero_choice() {
        let graph = quality_graph();
        let mut playback = BasisGraphPlaybackController::new(3);
        let config = BasisGraphPlaybackConfig {
            enabled: true,
            policy: BasisGraphPlaybackPolicy::Rank0,
            blend_duration: 0.0,
            min_branch_interval_segments: 0,
            max_position_cost_enabled: true,
            max_position_cost: 0.2,
            max_velocity_cost_enabled: false,
            max_acceleration_cost_enabled: false,
            ..BasisGraphPlaybackConfig::default()
        };
        playback.reset_with_config(0.0, &graph, &basis_infos(), config);

        playback.advance(0.25, &graph, &basis_infos(), config);

        let state = &playback.states()[0];
        assert_eq!(state.active_global_basis_id, 2);
        assert_eq!(state.segment, 2);
        assert_eq!(state.rejected_branch_count, 1);
        assert_eq!(state.rejected_branch_position_count, 1);
    }

    #[test]
    fn velocity_gate_rejects_high_velocity_branches() {
        let graph = quality_graph();
        let mut playback = BasisGraphPlaybackController::new(3);
        let config = BasisGraphPlaybackConfig {
            enabled: true,
            policy: BasisGraphPlaybackPolicy::Rank0,
            blend_duration: 0.0,
            min_branch_interval_segments: 0,
            max_velocity_cost_enabled: true,
            max_velocity_cost: 0.2,
            max_position_cost_enabled: false,
            max_acceleration_cost_enabled: false,
            ..BasisGraphPlaybackConfig::default()
        };
        playback.reset_with_config(0.0, &graph, &basis_infos(), config);

        playback.advance(0.25, &graph, &basis_infos(), config);

        let state = &playback.states()[0];
        assert_eq!(state.active_global_basis_id, 1);
        assert_eq!(state.rejected_branch_count, 1);
        assert_eq!(state.rejected_branch_velocity_count, 1);
    }

    #[test]
    fn acceleration_gate_rejects_high_acceleration_branches() {
        let graph = quality_graph();
        let mut playback = BasisGraphPlaybackController::new(3);
        let config = BasisGraphPlaybackConfig {
            enabled: true,
            policy: BasisGraphPlaybackPolicy::Rank0,
            blend_duration: 0.0,
            min_branch_interval_segments: 0,
            max_acceleration_cost_enabled: true,
            max_acceleration_cost: 0.2,
            max_position_cost_enabled: false,
            max_velocity_cost_enabled: false,
            ..BasisGraphPlaybackConfig::default()
        };
        playback.reset_with_config(0.0, &graph, &basis_infos(), config);

        playback.advance(0.25, &graph, &basis_infos(), config);

        let state = &playback.states()[0];
        assert_eq!(state.active_global_basis_id, 1);
        assert_eq!(state.rejected_branch_count, 1);
        assert_eq!(state.rejected_branch_acceleration_count, 1);
    }

    #[test]
    fn combined_term_gates_reject_when_any_enabled_term_fails() {
        let graph = quality_graph();
        let mut playback = BasisGraphPlaybackController::new(3);
        let config = BasisGraphPlaybackConfig {
            enabled: true,
            policy: BasisGraphPlaybackPolicy::Rank0,
            blend_duration: 0.0,
            min_branch_interval_segments: 0,
            max_position_cost_enabled: true,
            max_position_cost: 0.2,
            max_velocity_cost_enabled: true,
            max_velocity_cost: 0.2,
            max_acceleration_cost_enabled: true,
            max_acceleration_cost: 0.2,
            ..BasisGraphPlaybackConfig::default()
        };
        playback.reset_with_config(0.0, &graph, &basis_infos(), config);

        playback.advance(0.25, &graph, &basis_infos(), config);

        let state = &playback.states()[0];
        assert_eq!(state.active_global_basis_id, 0);
        assert_eq!(state.segment, 1);
        assert_eq!(state.rejected_branch_count, 3);
        assert_eq!(state.rejected_branch_position_count, 1);
        assert_eq!(state.rejected_branch_velocity_count, 1);
        assert_eq!(state.rejected_branch_acceleration_count, 1);
        assert!(matches!(state.last_edge, BasisGraphLastEdge::Continue));
    }

    #[test]
    fn stochastic_samples_only_branches_surviving_term_filters() {
        let graph = quality_graph();
        let mut playback = BasisGraphPlaybackController::new(3);
        let config = BasisGraphPlaybackConfig {
            enabled: true,
            policy: BasisGraphPlaybackPolicy::Stochastic,
            branch_probability: 1.0,
            temperature: 10.0,
            seed: 1,
            blend_duration: 0.0,
            min_branch_interval_segments: 0,
            max_position_cost_enabled: true,
            max_position_cost: 0.2,
            max_velocity_cost_enabled: false,
            max_acceleration_cost_enabled: true,
            max_acceleration_cost: 0.2,
            ..BasisGraphPlaybackConfig::default()
        };
        playback.reset_with_config(0.0, &graph, &basis_infos(), config);

        playback.advance(0.25, &graph, &basis_infos(), config);

        let state = &playback.states()[0];
        assert_eq!(state.active_global_basis_id, 2);
        assert_eq!(state.segment, 2);
        assert_eq!(state.rejected_branch_count, 2);
    }

    #[test]
    fn cooldown_blocked_boundary_does_not_increment_quality_rejection_counts() {
        let graph = quality_graph();
        let mut playback = BasisGraphPlaybackController::new(3);
        let config = BasisGraphPlaybackConfig {
            enabled: true,
            policy: BasisGraphPlaybackPolicy::Rank0,
            blend_duration: 0.0,
            min_branch_interval_segments: 8,
            max_position_cost_enabled: true,
            max_position_cost: 0.2,
            ..BasisGraphPlaybackConfig::default()
        };
        playback.reset_with_config(0.0, &graph, &basis_infos(), config);
        playback.advance(0.25, &graph, &basis_infos(), config);
        playback.advance(0.50, &graph, &basis_infos(), config);

        let state = &playback.states()[0];
        assert_eq!(state.active_global_basis_id, 2);
        assert_eq!(state.rejected_branch_position_count, 0);
    }

    fn cooldown_graph() -> BasisMotionGraph {
        let mut graph = test_graph();
        graph.lods[0].branches = vec![
            BasisMotionGraphBranch {
                from_basis: 0,
                from_segment: 0,
                to_basis: 1,
                to_segment: 0,
                rank: 0,
                score: 0.1,
                position_cost: 0.1,
                velocity_cost: 0.0,
                acceleration_cost: 0.0,
                usage_bonus: 0.0,
                transition: None,
            },
            BasisMotionGraphBranch {
                from_basis: 1,
                from_segment: 0,
                to_basis: 2,
                to_segment: 0,
                rank: 0,
                score: 0.1,
                position_cost: 0.1,
                velocity_cost: 0.0,
                acceleration_cost: 0.0,
                usage_bonus: 0.0,
                transition: None,
            },
            BasisMotionGraphBranch {
                from_basis: 1,
                from_segment: 2,
                to_basis: 2,
                to_segment: 0,
                rank: 0,
                score: 0.1,
                position_cost: 0.1,
                velocity_cost: 0.0,
                acceleration_cost: 0.0,
                usage_bonus: 0.0,
                transition: None,
            },
        ];
        graph
    }

    #[test]
    fn reset_initializes_branch_interval_as_eligible() {
        let graph = cooldown_graph();
        let config = BasisGraphPlaybackConfig {
            enabled: true,
            policy: BasisGraphPlaybackPolicy::Rank0,
            min_branch_interval_segments: 8,
            ..BasisGraphPlaybackConfig::default()
        };
        let mut playback = BasisGraphPlaybackController::new(3);

        playback.reset_with_config(0.0, &graph, &basis_infos(), config);

        assert_eq!(playback.states()[0].segments_since_branch, 8);
    }

    #[test]
    fn min_branch_interval_zero_preserves_immediate_branching() {
        let graph = cooldown_graph();
        let config = BasisGraphPlaybackConfig {
            enabled: true,
            policy: BasisGraphPlaybackPolicy::Rank0,
            blend_duration: 0.0,
            min_branch_interval_segments: 0,
            ..BasisGraphPlaybackConfig::default()
        };
        let mut playback = BasisGraphPlaybackController::new(3);
        playback.reset_with_config(0.0, &graph, &basis_infos(), config);

        playback.advance(0.25, &graph, &basis_infos(), config);
        playback.advance(0.50, &graph, &basis_infos(), config);

        let state = &playback.states()[0];
        assert_eq!(state.active_global_basis_id, 2);
        assert_eq!(state.segments_since_branch, 0);
    }

    #[test]
    fn rank_zero_waits_for_min_branch_interval_before_next_branch() {
        let graph = cooldown_graph();
        let config = BasisGraphPlaybackConfig {
            enabled: true,
            policy: BasisGraphPlaybackPolicy::Rank0,
            blend_duration: 0.0,
            min_branch_interval_segments: 2,
            ..BasisGraphPlaybackConfig::default()
        };
        let mut playback = BasisGraphPlaybackController::new(3);
        playback.reset_with_config(0.0, &graph, &basis_infos(), config);

        playback.advance(0.25, &graph, &basis_infos(), config);
        assert_eq!(playback.states()[0].active_global_basis_id, 1);
        assert_eq!(playback.states()[0].segments_since_branch, 0);

        playback.advance(0.50, &graph, &basis_infos(), config);
        assert_eq!(playback.states()[0].active_global_basis_id, 1);
        assert_eq!(playback.states()[0].segment, 1);
        assert_eq!(playback.states()[0].segments_since_branch, 1);

        playback.advance(0.75, &graph, &basis_infos(), config);
        assert_eq!(playback.states()[0].active_global_basis_id, 1);
        assert_eq!(playback.states()[0].segment, 2);
        assert_eq!(playback.states()[0].segments_since_branch, 2);

        playback.advance(0.01, &graph, &basis_infos(), config);
        assert_eq!(playback.states()[0].active_global_basis_id, 2);
        assert_eq!(playback.states()[0].segments_since_branch, 0);
    }

    #[test]
    fn stochastic_respects_min_branch_interval_even_at_full_probability() {
        let graph = cooldown_graph();
        let config = BasisGraphPlaybackConfig {
            enabled: true,
            policy: BasisGraphPlaybackPolicy::Stochastic,
            branch_probability: 1.0,
            temperature: 0.25,
            seed: 1,
            blend_duration: 0.0,
            min_branch_interval_segments: 2,
            ..BasisGraphPlaybackConfig::default()
        };
        let mut playback = BasisGraphPlaybackController::new(3);
        playback.reset_with_config(0.0, &graph, &basis_infos(), config);

        playback.advance(0.25, &graph, &basis_infos(), config);
        playback.advance(0.50, &graph, &basis_infos(), config);

        let state = &playback.states()[0];
        assert_eq!(state.active_global_basis_id, 1);
        assert_eq!(state.segment, 1);
        assert_eq!(state.segments_since_branch, 1);
    }

    #[test]
    fn transition_duration_does_not_count_as_branch_interval() {
        let mut graph = cooldown_graph();
        graph.format_version = 2;
        graph.lods[0].branches[0].transition = Some(test_transition());
        let config = BasisGraphPlaybackConfig {
            enabled: true,
            policy: BasisGraphPlaybackPolicy::Rank0,
            min_branch_interval_segments: 1,
            ..BasisGraphPlaybackConfig::default()
        };
        let mut playback = BasisGraphPlaybackController::new(3);
        playback.reset_with_config(0.0, &graph, &basis_infos(), config);

        playback.advance(0.25, &graph, &basis_infos(), config);
        playback.advance(0.75, &graph, &basis_infos(), config);
        playback.advance(0.01, &graph, &basis_infos(), config);

        let state = &playback.states()[0];
        assert!(!state.transition_active);
        assert_eq!(state.active_global_basis_id, 1);
        assert_eq!(state.segments_since_branch, 0);
    }

    #[test]
    fn stochastic_policy_is_reproducible_with_same_seed() {
        let graph = test_graph();
        let infos = basis_infos();
        let config = BasisGraphPlaybackConfig {
            enabled: true,
            policy: BasisGraphPlaybackPolicy::Stochastic,
            branch_probability: 1.0,
            temperature: 0.25,
            seed: 7,
            ..BasisGraphPlaybackConfig::default()
        };
        let mut a = BasisGraphPlaybackController::new(3);
        let mut b = BasisGraphPlaybackController::new(3);
        a.reset_with_config(0.0, &graph, &infos, config);
        b.reset_with_config(0.0, &graph, &infos, config);

        for step in 1..8 {
            let time = step as f32 * 0.26;
            a.advance(time, &graph, &infos, config);
            b.advance(time, &graph, &infos, config);
        }

        assert_eq!(a.states(), b.states());
    }

    #[test]
    fn stochastic_softmax_prefers_lower_scores() {
        let weights = branch_softmax_weights(&[0.1, 1.0], 0.25);

        assert_eq!(weights.len(), 2);
        assert!(weights[0] > weights[1]);
        assert!((weights[0] + weights[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn open_transition_sampling_hits_endpoints_and_tangents() {
        let transition = crate::basis_motion_graph::BasisMotionGraphTransition {
            kind: "open_catmull_rom".to_string(),
            duration_segments: 3,
            knots: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [3.0, 1.0, 0.0],
            ],
            start_tangent: [2.0, 0.0, 0.0],
            end_tangent: [2.0, 0.0, 0.0],
        };

        assert_eq!(sample_transition_delta(&transition, 0.0), [0.0, 0.0, 0.0]);
        assert_eq!(sample_transition_delta(&transition, 3.0), [3.0, 1.0, 0.0]);
        assert_eq!(
            sample_transition_velocity(&transition, 0.0),
            [2.0, 0.0, 0.0]
        );
        assert_eq!(
            sample_transition_velocity(&transition, 3.0),
            [2.0, 0.0, 0.0]
        );
    }

    #[test]
    fn rank_zero_branch_enters_exported_transition_before_target() {
        let mut graph = test_graph();
        graph.format_version = 2;
        graph.lods[0].branches[1].transition = Some(test_transition());
        let mut playback = BasisGraphPlaybackController::new(3);
        let config = BasisGraphPlaybackConfig {
            enabled: true,
            policy: BasisGraphPlaybackPolicy::Rank0,
            branch_probability: 0.2,
            temperature: 0.25,
            seed: 1,
            ..BasisGraphPlaybackConfig::default()
        };
        playback.reset_with_config(0.0, &graph, &basis_infos(), config);

        playback.advance(0.25, &graph, &basis_infos(), config);

        let state = &playback.states()[0];
        assert!(state.transition_active);
        assert_eq!(state.active_global_basis_id, 0);
        assert_eq!(state.transition_target_global_basis_id, 1);
        assert_eq!(state.transition_target_segment, 2);
        assert_eq!(state.transition_duration_segments, 3.0);
        assert_eq!(state.transition_delta, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn exported_transition_blocks_new_branch_until_it_completes() {
        let mut graph = test_graph();
        graph.format_version = 2;
        graph.lods[0].branches[1].transition = Some(test_transition());
        let mut playback = BasisGraphPlaybackController::new(3);
        let config = BasisGraphPlaybackConfig {
            enabled: true,
            policy: BasisGraphPlaybackPolicy::Rank0,
            branch_probability: 0.2,
            temperature: 0.25,
            seed: 1,
            ..BasisGraphPlaybackConfig::default()
        };
        playback.reset_with_config(0.0, &graph, &basis_infos(), config);
        playback.advance(0.25, &graph, &basis_infos(), config);
        playback.advance(0.75, &graph, &basis_infos(), config);

        let state = &playback.states()[0];
        assert!(state.transition_active);
        assert_eq!(state.active_global_basis_id, 0);

        playback.advance(0.01, &graph, &basis_infos(), config);
        let state = &playback.states()[0];
        assert!(!state.transition_active);
        assert_eq!(state.active_global_basis_id, 1);
        assert_eq!(state.segment, 2);
        assert!(state.segment_phase > 0.0);
    }

    #[test]
    fn graph_direct_packing_emits_transition_delta() {
        let mut graph = test_graph();
        graph.format_version = 2;
        graph.lods[0].branches[1].transition = Some(test_transition());
        let mut playback = BasisGraphPlaybackController::new(3);
        let config = BasisGraphPlaybackConfig {
            enabled: true,
            policy: BasisGraphPlaybackPolicy::Rank0,
            branch_probability: 0.2,
            temperature: 0.25,
            seed: 1,
            ..BasisGraphPlaybackConfig::default()
        };
        playback.reset_with_config(0.0, &graph, &basis_infos(), config);
        playback.advance(0.375, &graph, &basis_infos(), config);

        let packed = pack_basis_graph_direct_overrides(Some(playback.states()), 3);

        assert_eq!(packed.len(), 3);
        assert_eq!(packed[0][3], 1.0);
        assert!(packed[0][0] > 0.0);
        assert_eq!(packed[1], [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn graph_override_packing_can_emit_disabled_defaults() {
        let disabled = pack_basis_graph_sample_overrides(None, 3);

        assert_eq!(disabled, vec![[0.0, 0.0, 0.0, 0.0]; 3]);
    }

    #[test]
    fn graph_override_packing_emits_one_vec4_per_global_basis() {
        let mut playback = BasisGraphPlaybackController::new(3);
        playback.reset(0.375, &test_graph(), &basis_infos());

        let packed = pack_basis_graph_sample_overrides(Some(playback.states()), 3);

        assert_eq!(packed.len(), 3);
        assert_eq!(packed[0], [0.0, 1.0, 0.5, 1.0]);
        assert_eq!(packed[2], [2.0, 1.0, 0.5, 1.0]);
    }

    #[test]
    fn graph_blend_packing_emits_one_vec4_per_global_basis() {
        let mut playback = BasisGraphPlaybackController::new(3);
        let config = BasisGraphPlaybackConfig {
            enabled: true,
            policy: BasisGraphPlaybackPolicy::Rank0,
            branch_probability: 0.2,
            temperature: 0.25,
            seed: 1,
            blend_duration: 0.25,
            max_branch_score_enabled: false,
            max_branch_score: 1.0,
            min_branch_interval_segments: 0,
            max_position_cost_enabled: false,
            max_position_cost: 1.0,
            max_velocity_cost_enabled: false,
            max_velocity_cost: 1.0,
            max_acceleration_cost_enabled: false,
            max_acceleration_cost: 1.0,
        };
        playback.reset_with_config(0.0, &test_graph(), &basis_infos(), config);
        playback.advance(0.26, &test_graph(), &basis_infos(), config);

        let packed = pack_basis_graph_blend_overrides(Some(playback.states()), 3);

        assert_eq!(packed.len(), 3);
        assert_eq!(packed[0][0], 0.0);
        assert_eq!(packed[0][1], 1.0);
        assert!((packed[0][2] - 0.04).abs() < 1e-5);
        assert!((packed[0][3] - smoothstep01(0.04 / 0.25)).abs() < 1e-5);
        assert_eq!(packed[1][0], 1.0);
        assert_eq!(packed[1][1], 1.0);
        assert!((packed[1][2] - 0.04).abs() < 1e-5);
        assert_eq!(packed[1][3], 1.0);
    }

    #[test]
    fn explicit_segment_time_matches_normal_basis_sampling() {
        let knots = vec![
            0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, //
            2.0, 0.0, 0.0, //
            3.0, 0.0, 0.0, //
        ];
        let time = explicit_segment_time01(4, 1, 0.5);

        assert_eq!(
            basis_bank_delta(&knots, 1, 4, 0, time),
            basis_bank_delta(&knots, 1, 4, 0, 0.375)
        );
    }
}
