const WORKGROUP_SIZE: u32 = 128u;

struct MotionUniform {
    time01: f32,
    splat_count: u32,
    gaussian_tex_width: u32,
    knot_count: u32,
    top_k: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0)
var<uniform> u_motion: MotionUniform;

@group(0) @binding(1)
var t_base_gaussian: texture_2d<u32>;

@group(0) @binding(2)
var<storage, read> s_basis_knots: array<vec4<f32>>;

@group(0) @binding(3)
var<storage, read> s_basis_ids: array<u32>;

@group(0) @binding(4)
var<storage, read> s_weights: array<f32>;

@group(0) @binding(5)
var<storage, read> s_basis_edits: array<vec4<f32>>;

@group(0) @binding(6)
var<storage, read> s_graph_samples: array<vec4<f32>>;

@group(0) @binding(7)
var<storage, read> s_graph_blends: array<vec4<f32>>;

@group(0) @binding(8)
var<storage, read> s_graph_directs: array<vec4<f32>>;

@group(0) @binding(9)
var t_output_gaussian: texture_storage_2d<rgba32uint, write>;

fn basis_knot(basis_id: u32, knot_index: u32) -> vec3<f32> {
    let index = basis_id * u_motion.knot_count + knot_index;
    return s_basis_knots[index].xyz;
}

fn sample_basis_delta(basis_id: u32, time01: f32) -> vec3<f32> {
    let knot_count = max(u_motion.knot_count, 1u);
    let scaled = fract(time01) * f32(knot_count);
    let segment = u32(floor(scaled)) % knot_count;
    let u = scaled - f32(segment);
    return sample_basis_delta_segment(basis_id, segment, u);
}

fn sample_basis_delta_segment(basis_id: u32, segment_in: u32, segment_phase: f32) -> vec3<f32> {
    let knot_count = max(u_motion.knot_count, 1u);
    let segment = segment_in % knot_count;
    let u = clamp(segment_phase, 0.0, 1.0);
    let u2 = u * u;
    let u3 = u2 * u;

    let p0 = basis_knot(basis_id, (segment + knot_count - 1u) % knot_count);
    let p1 = basis_knot(basis_id, segment);
    let p2 = basis_knot(basis_id, (segment + 1u) % knot_count);
    let p3 = basis_knot(basis_id, (segment + 2u) % knot_count);

    return 0.5 * (
        2.0 * p1
        + (-p0 + p2) * u
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3
    );
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= u_motion.splat_count {
        return;
    }

    let splats_per_row = u_motion.gaussian_tex_width / 2u;
    let tex_u = (idx % splats_per_row) * 2u;
    let tex_v = idx / splats_per_row;
    let coord0 = vec2<i32>(i32(tex_u), i32(tex_v));
    let coord1 = vec2<i32>(i32(tex_u + 1u), i32(tex_v));

    let base0 = textureLoad(t_base_gaussian, coord0, 0);
    let base1 = textureLoad(t_base_gaussian, coord1, 0);
    let base_mean = bitcast<vec3<f32>>(base0.rgb);

    var delta = vec3<f32>(0.0);
    let coeff_base = idx * u_motion.top_k;
    for (var slot = 0u; slot < u_motion.top_k; slot = slot + 1u) {
        let coeff_index = coeff_base + slot;
        let basis_id = s_basis_ids[coeff_index];
        let weight = s_weights[coeff_index];
        let edit = s_basis_edits[basis_id];
        let graph_sample = s_graph_samples[basis_id];
        let graph_direct = s_graph_directs[basis_id];
        let graph_enabled = graph_sample.w >= 0.5;
        let graph_direct_enabled = graph_direct.w >= 0.5;
        let edit_enabled = edit.x >= 0.5;
        let amplitude_scale = select(1.0, edit.y, edit_enabled);
        let sample_time = select(u_motion.time01, u_motion.time01 * edit.w + edit.z, edit_enabled);
        let graph_target_basis = u32(round(graph_sample.x));
        let graph_segment = u32(round(graph_sample.y));
        var sample_delta = sample_basis_delta(basis_id, sample_time);
        if graph_direct_enabled {
            sample_delta = graph_direct.xyz;
        } else if graph_enabled {
            let graph_target_delta =
                sample_basis_delta_segment(graph_target_basis, graph_segment, graph_sample.z);
            let graph_blend = s_graph_blends[basis_id];
            let blend_weight = clamp(graph_blend.w, 0.0, 1.0);
            if blend_weight < 1.0 {
                let blend_from_basis = u32(round(graph_blend.x));
                let blend_from_segment = u32(round(graph_blend.y));
                let blend_from_delta =
                    sample_basis_delta_segment(blend_from_basis, blend_from_segment, graph_blend.z);
                sample_delta = mix(blend_from_delta, graph_target_delta, blend_weight);
            } else {
                sample_delta = graph_target_delta;
            }
        }
        delta = delta + weight * amplitude_scale * sample_delta;
    }
    let deformed_mean = base_mean + delta;

    textureStore(
        t_output_gaussian,
        coord0,
        vec4<u32>(
            bitcast<u32>(deformed_mean.x),
            bitcast<u32>(deformed_mean.y),
            bitcast<u32>(deformed_mean.z),
            base0.a
        )
    );
    textureStore(t_output_gaussian, coord1, base1);
}
