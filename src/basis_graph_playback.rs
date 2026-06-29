use crate::basis_bank_edit::BasisEditOverride;
use crate::basis_branch_regions::{BasisGraphRegionConfig, graph_state_index};
use crate::basis_graph_authoring::BasisGraphBranchOverrides;
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
        to_basis_id: usize,
        to_segment: usize,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct BasisGraphPlaybackState {
    pub region_id: usize,
    pub original_basis_id: usize,
    pub active_basis_id: usize,
    pub segment: usize,
    pub segment_phase: f32,
    pub blend_from_basis_id: usize,
    pub blend_from_segment: usize,
    pub blend_phase: f32,
    pub blend_weight: f32,
    pub blend_active: bool,
    pub transition_active: bool,
    pub transition_phase_segments: f32,
    pub transition_duration_segments: f32,
    pub transition_delta: [f32; 3],
    pub transition: Option<BasisMotionGraphTransition>,
    pub transition_target_basis_id: usize,
    pub transition_target_segment: usize,
    pub rejected_branch_count: usize,
    pub rejected_branch_score_count: usize,
    pub rejected_branch_position_count: usize,
    pub rejected_branch_velocity_count: usize,
    pub rejected_branch_acceleration_count: usize,
    pub segments_since_branch: u32,
    pub last_edge: BasisGraphLastEdge,
    rng: BasisGraphPlaybackRng,
}
#[derive(Clone, Copy, Debug, PartialEq)]
struct BasisPlaybackTiming {
    time_scale: f32,
    phase_offset: f32,
}

impl Default for BasisPlaybackTiming {
    fn default() -> Self {
        Self {
            time_scale: 1.0,
            phase_offset: 0.0,
        }
    }
}

impl BasisPlaybackTiming {
    fn from_edit(edit: Option<&BasisEditOverride>) -> Self {
        let Some(edit) = edit.filter(|edit| edit.enabled) else {
            return Self::default();
        };
        Self {
            time_scale: if edit.time_scale.is_finite() {
                edit.time_scale.clamp(0.0, 4.0)
            } else {
                1.0
            },
            phase_offset: if edit.phase_offset.is_finite() {
                edit.phase_offset
            } else {
                0.0
            },
        }
    }

    fn edited_time(self, time01: f32) -> f32 {
        time01 * self.time_scale + self.phase_offset
    }
}

pub struct BasisGraphPlaybackController {
    states: Vec<BasisGraphPlaybackState>,
    previous_time01: Option<f32>,
    last_config: BasisGraphPlaybackConfig,
    last_region_config: BasisGraphRegionConfig,
    timings: Vec<BasisPlaybackTiming>,
    initialized: bool,
}

impl BasisGraphPlaybackController {
    pub fn new(basis_count: usize) -> Self {
        Self {
            states: Vec::with_capacity(basis_count),
            previous_time01: None,
            last_config: BasisGraphPlaybackConfig::default(),
            last_region_config: BasisGraphRegionConfig::default(),
            initialized: false,
            timings: vec![BasisPlaybackTiming::default(); basis_count],
        }
    }

    pub fn states(&self) -> &[BasisGraphPlaybackState] {
        self.states.as_slice()
    }

    #[cfg(test)]
    pub fn reset(&mut self, time01: f32, graph: &BasisMotionGraph, basis_count: usize) {
        self.reset_with_region_config_and_edits(
            time01,
            graph,
            basis_count,
            BasisGraphPlaybackConfig::default(),
            BasisGraphRegionConfig::default(),
            &[],
        );
    }

    pub fn reset_with_region_config_and_edits(
        &mut self,
        time01: f32,
        graph: &BasisMotionGraph,
        basis_count: usize,
        config: BasisGraphPlaybackConfig,
        region_config: BasisGraphRegionConfig,
        edits: &[BasisEditOverride],
    ) {
        let region_config = region_config.sanitized();
        let region_count = region_config.effective_region_count() as usize;
        self.timings = (0..basis_count)
            .map(|basis_id| BasisPlaybackTiming::from_edit(edits.get(basis_id)))
            .collect();
        self.states.clear();
        self.states
            .reserve(region_count.saturating_mul(basis_count));
        for region_id in 0..region_count {
            for basis_id in 0..basis_count {
                let timing = self.timings[basis_id];
                self.states.push(new_playback_state(
                    region_id,
                    basis_id,
                    timing.edited_time(time01),
                    graph.knot_count,
                    config,
                ));
            }
        }
        self.previous_time01 = Some(time01.rem_euclid(1.0));
        self.last_config = config;
        self.last_region_config = region_config;
        self.initialized = true;
    }
    pub fn advance_with_region_config_and_overrides_and_edits(
        &mut self,
        time01: f32,
        graph: &BasisMotionGraph,
        basis_count: usize,
        config: BasisGraphPlaybackConfig,
        region_config: BasisGraphRegionConfig,
        branch_overrides: Option<&BasisGraphBranchOverrides>,
        edits: &[BasisEditOverride],
    ) {
        let region_config = region_config.sanitized();
        let timings: Vec<_> = (0..basis_count)
            .map(|basis_id| BasisPlaybackTiming::from_edit(edits.get(basis_id)))
            .collect();
        if !config.enabled {
            self.previous_time01 = Some(time01.rem_euclid(1.0));
            self.timings = timings;
            return;
        }
        let expected_state_count =
            basis_count.saturating_mul(region_config.effective_region_count() as usize);
        if !self.initialized
            || self.states.len() != expected_state_count
            || self.timings.len() != basis_count
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
            || region_config != self.last_region_config
        {
            self.reset_with_region_config_and_edits(
                time01,
                graph,
                basis_count,
                config,
                region_config,
                edits,
            );
            return;
        }

        let time01 = time01.rem_euclid(1.0);
        let Some(previous_time01) = self.previous_time01 else {
            self.reset_with_region_config_and_edits(
                time01,
                graph,
                basis_count,
                config,
                region_config,
                edits,
            );
            return;
        };
        let delta01 = wrapped_forward_delta01(previous_time01, time01);
        if delta01 > MAX_GRAPH_PLAYBACK_DELTA01 {
            self.reset_with_region_config_and_edits(
                time01,
                graph,
                basis_count,
                config,
                region_config,
                edits,
            );
            return;
        }

        let timing_changed: Vec<_> = timings
            .iter()
            .zip(self.timings.iter())
            .map(|(current, previous)| current != previous)
            .collect();
        for state in &mut self.states {
            let basis_id = state.original_basis_id;
            let timing = timings[basis_id];
            if timing_changed[basis_id] {
                *state = new_playback_state(
                    state.region_id,
                    basis_id,
                    timing.edited_time(time01),
                    graph.knot_count,
                    config,
                );
            } else {
                let mut segment_delta = delta01 * timing.time_scale * graph.knot_count as f32;
                advance_state(
                    state,
                    &mut segment_delta,
                    graph,
                    basis_count,
                    config,
                    branch_overrides,
                );
            }
        }
        self.timings = timings;
        self.previous_time01 = Some(time01);
        self.last_config = config;
        self.last_region_config = region_config;
    }
}

fn new_playback_state(
    region_id: usize,
    basis_id: usize,
    time01: f32,
    knot_count: usize,
    config: BasisGraphPlaybackConfig,
) -> BasisGraphPlaybackState {
    let (segment, segment_phase) = segment_and_phase(time01, knot_count);
    BasisGraphPlaybackState {
        region_id,
        original_basis_id: basis_id,
        active_basis_id: basis_id,
        segment,
        segment_phase,
        blend_from_basis_id: basis_id,
        blend_from_segment: segment,
        blend_phase: 1.0,
        blend_weight: 1.0,
        blend_active: false,
        transition_active: false,
        transition_phase_segments: 0.0,
        transition_duration_segments: 0.0,
        transition_delta: [0.0; 3],
        transition: None,
        transition_target_basis_id: basis_id,
        transition_target_segment: segment,
        rejected_branch_count: 0,
        rejected_branch_score_count: 0,
        rejected_branch_position_count: 0,
        rejected_branch_velocity_count: 0,
        rejected_branch_acceleration_count: 0,
        segments_since_branch: config.min_branch_interval_segments,
        last_edge: BasisGraphLastEdge::Reset,
        rng: BasisGraphPlaybackRng::new(state_rng_seed(config.seed, region_id, basis_id)),
    }
}
pub fn pack_basis_graph_blend_overrides(
    states: Option<&[BasisGraphPlaybackState]>,
    basis_count: usize,
    graph_region_count: usize,
) -> Vec<[f32; 4]> {
    let mut packed = vec![[0.0, 0.0, 0.0, 1.0]; packed_graph_len(basis_count, graph_region_count)];
    let Some(states) = states else {
        return packed;
    };
    for state in states {
        if let Some(index) =
            graph_state_index(state.region_id, state.original_basis_id, basis_count)
        {
            if index < packed.len() {
                let from_basis = if state.blend_active {
                    state.blend_from_basis_id
                } else {
                    state.active_basis_id
                };
                let from_segment = if state.blend_active {
                    state.blend_from_segment
                } else {
                    state.segment
                };
                packed[index] = [
                    from_basis as f32,
                    from_segment as f32,
                    state.segment_phase.clamp(0.0, 1.0),
                    state.blend_weight.clamp(0.0, 1.0),
                ];
            }
        }
    }
    packed
}

pub fn pack_basis_graph_direct_overrides(
    states: Option<&[BasisGraphPlaybackState]>,
    basis_count: usize,
    graph_region_count: usize,
) -> Vec<[f32; 4]> {
    let mut packed = vec![[0.0, 0.0, 0.0, 0.0]; packed_graph_len(basis_count, graph_region_count)];
    let Some(states) = states else {
        return packed;
    };
    for state in states {
        if let Some(index) =
            graph_state_index(state.region_id, state.original_basis_id, basis_count)
        {
            if index < packed.len() && state.transition_active {
                packed[index] = [
                    state.transition_delta[0],
                    state.transition_delta[1],
                    state.transition_delta[2],
                    1.0,
                ];
            }
        }
    }
    packed
}

pub fn pack_basis_graph_sample_overrides(
    states: Option<&[BasisGraphPlaybackState]>,
    basis_count: usize,
    graph_region_count: usize,
) -> Vec<[f32; 4]> {
    let mut packed = vec![[0.0, 0.0, 0.0, 0.0]; packed_graph_len(basis_count, graph_region_count)];
    let Some(states) = states else {
        return packed;
    };
    for state in states {
        if let Some(index) =
            graph_state_index(state.region_id, state.original_basis_id, basis_count)
        {
            if index < packed.len() {
                packed[index] = [
                    state.active_basis_id as f32,
                    state.segment as f32,
                    state.segment_phase.clamp(0.0, 1.0),
                    1.0,
                ];
            }
        }
    }
    packed
}

fn packed_graph_len(basis_count: usize, graph_region_count: usize) -> usize {
    basis_count.saturating_mul(graph_region_count.max(1))
}

fn state_rng_seed(seed: u32, region_id: usize, basis_id: usize) -> u32 {
    let mut h = seed.max(1);
    h ^= (region_id as u32).wrapping_mul(0x9E37_79B9);
    h = h.rotate_left(13);
    h ^= (basis_id as u32).wrapping_mul(0x85EB_CA6B);
    h.max(1)
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
    basis_count: usize,
    config: BasisGraphPlaybackConfig,
    branch_overrides: Option<&BasisGraphBranchOverrides>,
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
        choose_next_edge(state, graph, basis_count, config, branch_overrides);
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
    state.active_basis_id = state.transition_target_basis_id;
    state.segment = state.transition_target_segment;
    state.segment_phase = 0.0;
    state.blend_active = false;
    state.blend_phase = 1.0;
    state.blend_weight = 1.0;
}

fn choose_next_edge(
    state: &mut BasisGraphPlaybackState,
    graph: &BasisMotionGraph,
    basis_count: usize,
    config: BasisGraphPlaybackConfig,
    branch_overrides: Option<&BasisGraphBranchOverrides>,
) {
    let override_branches = branch_overrides
        .and_then(|overrides| overrides.branches_for(state.active_basis_id, state.segment));
    let baseline_branches;
    let branches: Vec<&BasisMotionGraphBranch> = if let Some(branches) = override_branches {
        branches.iter().collect()
    } else {
        baseline_branches = graph.branches_for(state.active_basis_id, state.segment);
        baseline_branches
    };
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
                || state.rng.next_f32() > config.branch_probability.clamp(0.0, 1.0)
            {
                None
            } else {
                choose_stochastic_branch(&filtered_branches, config.temperature, &mut state.rng)
            }
        }
    };

    if let Some(branch) = branch {
        apply_branch(state, branch, graph.knot_count, basis_count, config);
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
    basis_count: usize,
    config: BasisGraphPlaybackConfig,
) {
    if branch.to_basis >= basis_count {
        state.last_edge = BasisGraphLastEdge::Continue;
        return;
    }
    let target_basis_id = branch.to_basis;
    if let Some(transition) = branch.transition.as_ref() {
        state.transition_active = true;
        state.transition_phase_segments = 0.0;
        state.transition_duration_segments = transition.duration_segments as f32;
        state.transition_delta = sample_transition_delta(transition, 0.0);
        state.transition = Some(transition.clone());
        state.transition_target_basis_id = target_basis_id;
        state.transition_target_segment = branch.to_segment;
        state.blend_active = false;
        state.blend_phase = 1.0;
        state.blend_weight = 1.0;
        state.segments_since_branch = 0;
        state.last_edge = BasisGraphLastEdge::Branch {
            rank: branch.rank,
            to_basis_id: target_basis_id,
            to_segment: branch.to_segment,
        };
        return;
    }

    let source_basis_id = state.active_basis_id;
    let source_default_segment = if knot_count == 0 {
        state.segment
    } else {
        (state.segment + 1) % knot_count
    };
    state.active_basis_id = target_basis_id;
    state.segment = branch.to_segment;
    state.blend_from_basis_id = source_basis_id;
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
        to_basis_id: target_basis_id,
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

#[derive(Clone, Debug, PartialEq)]
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
    use crate::basis_bank_edit::BasisEditOverride;
    use crate::basis_motion_graph::{BasisMotionGraphLod, BasisMotionGraphScoreWeights};

    fn graph() -> BasisMotionGraph {
        BasisMotionGraph {
            format: "basis_motion_graph".to_string(),
            format_version: 1,
            basis_scope: "shared_lod0".to_string(),
            basis_source_lod: Some(0),
            node_unit: "basis_segment".to_string(),
            include_lods: vec![0],
            basis_count: 2,
            knot_count: 4,
            branch_top_k: 1,
            score_weights: BasisMotionGraphScoreWeights {
                position: 1.0,
                velocity: 1.0,
                acceleration: 1.0,
                usage: 0.0,
            },
            lods: vec![BasisMotionGraphLod {
                lod_id: 0,
                branches: Vec::new(),
            }],
        }
    }

    #[test]
    fn reset_creates_one_direct_state_per_shared_basis() {
        let mut playback = BasisGraphPlaybackController::new(2);
        playback.reset(0.25, &graph(), 2);
        assert_eq!(playback.states().len(), 2);
        assert_eq!(playback.states()[0].original_basis_id, 0);
        assert_eq!(playback.states()[1].original_basis_id, 1);
        assert_eq!(playback.states()[1].active_basis_id, 1);
    }

    fn enabled_config() -> BasisGraphPlaybackConfig {
        BasisGraphPlaybackConfig {
            enabled: true,
            ..BasisGraphPlaybackConfig::default()
        }
    }

    fn timing_edit(time_scale: f32, phase_offset: f32) -> BasisEditOverride {
        BasisEditOverride {
            enabled: true,
            amplitude_scale: 1.0,
            phase_offset,
            time_scale,
        }
    }

    #[test]
    fn edited_graph_clocks_scale_freeze_and_offset_per_basis() {
        let graph = graph();
        let mut playback = BasisGraphPlaybackController::new(2);
        let edits = [timing_edit(2.0, 0.25), timing_edit(0.0, 0.0)];
        playback.reset_with_region_config_and_edits(
            0.0,
            &graph,
            2,
            enabled_config(),
            BasisGraphRegionConfig::default(),
            &edits,
        );
        assert_eq!(playback.states()[0].segment, 1);
        assert_eq!(playback.states()[0].segment_phase, 0.0);
        assert_eq!(playback.states()[1].segment, 0);

        playback.advance_with_region_config_and_overrides_and_edits(
            0.125,
            &graph,
            2,
            enabled_config(),
            BasisGraphRegionConfig::default(),
            None,
            &edits,
        );
        assert_eq!(playback.states()[0].segment, 2);
        assert_eq!(playback.states()[0].segment_phase, 0.0);
        assert_eq!(playback.states()[1].segment, 0);
        assert_eq!(playback.states()[1].segment_phase, 0.0);
    }

    #[test]
    fn timing_change_rebases_only_affected_basis_states() {
        let graph = graph();
        let mut playback = BasisGraphPlaybackController::new(2);
        let default_edits = [BasisEditOverride::default(), BasisEditOverride::default()];
        playback.reset_with_region_config_and_edits(
            0.0,
            &graph,
            2,
            enabled_config(),
            BasisGraphRegionConfig::default(),
            &default_edits,
        );
        playback.advance_with_region_config_and_overrides_and_edits(
            0.125,
            &graph,
            2,
            enabled_config(),
            BasisGraphRegionConfig::default(),
            None,
            &default_edits,
        );

        let changed_edits = [timing_edit(2.0, 0.0), BasisEditOverride::default()];
        playback.advance_with_region_config_and_overrides_and_edits(
            0.25,
            &graph,
            2,
            enabled_config(),
            BasisGraphRegionConfig::default(),
            None,
            &changed_edits,
        );
        assert_eq!(playback.states()[0].segment, 2);
        assert_eq!(playback.states()[0].segment_phase, 0.0);
        assert_eq!(playback.states()[1].segment, 1);
        assert_eq!(playback.states()[1].segment_phase, 0.0);
    }
}
