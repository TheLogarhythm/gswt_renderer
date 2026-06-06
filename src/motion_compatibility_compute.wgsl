const WORKGROUP_SIZE: u32 = 128u;

struct CompatibilityUniform {
    item_count: u32,
    splat_count: u32,
    knot_count: u32,
    knot_texture_width: u32,
    selected_knot: u32,
    compare_all: u32,
    sample_stride: u32,
    sample_count: u32,
}

struct NetworkMetaUniform {
    counts0: vec4<u32>,
    counts1: vec4<u32>,
    aabb_max: vec4<f32>,
    aabb_min: vec4<f32>,
    scale_and_pad: vec4<f32>,
    volume_counts: vec4<u32>,
}

struct PartialStats {
    count: u32,
    worst_item: u32,
    nonzero_error_count: u32,
    nonzero_spline_count: u32,
    nonzero_volume_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    sum_error: f32,
    sum_error_sq: f32,
    max_error: f32,
    sum_spline_magnitude: f32,
    max_spline_magnitude: f32,
    sum_volume_magnitude: f32,
    max_volume_magnitude: f32,
    _pad3: f32,
}

struct DeformDelta {
    dx: vec3<f32>,
    dr: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> u_compare: CompatibilityUniform;

@group(0) @binding(1)
var<uniform> u_meta: NetworkMetaUniform;

@group(0) @binding(2)
var<storage, read> s_orig_means: array<vec4<f32>>;

@group(0) @binding(3)
var<storage, read> s_volume_data: array<u32>;

@group(0) @binding(4)
var t_knots: texture_2d<f32>;

@group(0) @binding(5)
var<storage, read> s_sample_times: array<f32>;

@group(0) @binding(6)
var<storage, read_write> s_partial_stats: array<PartialStats>;

@group(0) @binding(7)
var<storage, read_write> s_sample_errors: array<f32>;

var<workgroup> wg_sum: array<f32, 128>;
var<workgroup> wg_sum_sq: array<f32, 128>;
var<workgroup> wg_max_error: array<f32, 128>;
var<workgroup> wg_max_item: array<u32, 128>;
var<workgroup> wg_valid: array<u32, 128>;
var<workgroup> wg_nonzero_error: array<u32, 128>;
var<workgroup> wg_nonzero_spline: array<u32, 128>;
var<workgroup> wg_nonzero_volume: array<u32, 128>;
var<workgroup> wg_sum_spline_magnitude: array<f32, 128>;
var<workgroup> wg_max_spline_magnitude: array<f32, 128>;
var<workgroup> wg_sum_volume_magnitude: array<f32, 128>;
var<workgroup> wg_max_volume_magnitude: array<f32, 128>;

fn load_volume_delta(key: u32, x: u32, y: u32, z: u32) -> DeformDelta {
    let res = u_meta.volume_counts.x;
    let words_per_sample = u_meta.volume_counts.z;
    let sample_idx = ((key * res + z) * res + y) * res + x;
    let base = sample_idx * words_per_sample;
    let w0 = unpack2x16float(s_volume_data[base + 0u]);
    let w1 = unpack2x16float(s_volume_data[base + 1u]);
    let w2 = unpack2x16float(s_volume_data[base + 2u]);
    let w3 = unpack2x16float(s_volume_data[base + 3u]);
    return DeformDelta(
        vec3<f32>(w0.x, w0.y, w1.x),
        vec4<f32>(w1.y, w2.x, w2.y, w3.x)
    );
}

fn lerp_delta(a: DeformDelta, b: DeformDelta, t: f32) -> DeformDelta {
    return DeformDelta(mix(a.dx, b.dx, t), mix(a.dr, b.dr, t));
}

fn sample_volume_trilinear(key: u32, coord: vec3<f32>) -> DeformDelta {
    let res = u_meta.volume_counts.x;
    let max_coord = f32(res - 1u);
    let c = clamp(coord, vec3<f32>(0.0), vec3<f32>(max_coord));
    let x0 = min(u32(floor(c.x)), res - 1u);
    let y0 = min(u32(floor(c.y)), res - 1u);
    let z0 = min(u32(floor(c.z)), res - 1u);
    let x1 = min(x0 + 1u, res - 1u);
    let y1 = min(y0 + 1u, res - 1u);
    let z1 = min(z0 + 1u, res - 1u);
    let tx = c.x - f32(x0);
    let ty = c.y - f32(y0);
    let tz = c.z - f32(z0);

    let c000 = load_volume_delta(key, x0, y0, z0);
    let c100 = load_volume_delta(key, x1, y0, z0);
    let c010 = load_volume_delta(key, x0, y1, z0);
    let c110 = load_volume_delta(key, x1, y1, z0);
    let c001 = load_volume_delta(key, x0, y0, z1);
    let c101 = load_volume_delta(key, x1, y0, z1);
    let c011 = load_volume_delta(key, x0, y1, z1);
    let c111 = load_volume_delta(key, x1, y1, z1);

    let c00 = lerp_delta(c000, c100, tx);
    let c10 = lerp_delta(c010, c110, tx);
    let c01 = lerp_delta(c001, c101, tx);
    let c11 = lerp_delta(c011, c111, tx);
    let c0 = lerp_delta(c00, c10, ty);
    let c1 = lerp_delta(c01, c11, ty);
    return lerp_delta(c0, c1, tz);
}

fn knot_texel_coord(flat_index: u32) -> vec2<i32> {
    let width = max(u_compare.knot_texture_width, 1u);
    return vec2<i32>(
        i32(flat_index % width),
        i32(flat_index / width)
    );
}

fn load_delta_knot(splat_index: u32, knot_index: u32) -> vec3<f32> {
    let flat_index = splat_index * u_compare.knot_count + knot_index;
    return textureLoad(t_knots, knot_texel_coord(flat_index), 0).xyz;
}

fn volume_delta_for_splat(splat_index: u32, time01: f32) -> vec3<f32> {
    let orig = s_orig_means[splat_index].xyz;
    let res = u_meta.volume_counts.x;
    let keys = max(u_meta.volume_counts.y, 1u);
    let max_coord = f32(res - 1u);
    var volume_coord = vec3<f32>(0.0);
    for (var axis: u32 = 0u; axis < 3u; axis = axis + 1u) {
        let aabb_max = u_meta.aabb_max[axis];
        let aabb_min = u_meta.aabb_min[axis];
        let denom = aabb_min - aabb_max;
        let norm01 = clamp((orig[axis] - aabb_max) / denom, 0.0, 1.0);
        volume_coord[axis] = norm01 * max_coord;
    }

    let time_scaled = clamp(time01, 0.0, 1.0) * f32(keys - 1u);
    let k0 = u32(floor(time_scaled));
    let k1 = min(k0 + 1u, keys - 1u);
    let kt = time_scaled - f32(k0);
    let d0 = sample_volume_trilinear(k0, volume_coord);
    let d1 = sample_volume_trilinear(k1, volume_coord);
    return lerp_delta(d0, d1, kt).dx * u_meta.scale_and_pad.x;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let item = gid.x;
    let local = lid.x;
    var err = 0.0;
    var spline_magnitude = 0.0;
    var volume_magnitude = 0.0;
    var valid = 0u;
    if item < u_compare.item_count {
        var splat_index = item;
        var knot_index = min(u_compare.selected_knot, u_compare.knot_count - 1u);
        if u_compare.compare_all != 0u {
            knot_index = item / u_compare.splat_count;
            splat_index = item - knot_index * u_compare.splat_count;
        }
        let time01 = s_sample_times[knot_index];
        let spline_delta = load_delta_knot(splat_index, knot_index);
        let volume_delta = volume_delta_for_splat(splat_index, time01);
        err = length(spline_delta - volume_delta);
        spline_magnitude = length(spline_delta);
        volume_magnitude = length(volume_delta);
        valid = 1u;
        if item % max(u_compare.sample_stride, 1u) == 0u {
            let sample_index = item / max(u_compare.sample_stride, 1u);
            if sample_index < u_compare.sample_count {
                s_sample_errors[sample_index] = err;
            }
        }
    }
    wg_sum[local] = err;
    wg_sum_sq[local] = err * err;
    wg_max_error[local] = err;
    wg_max_item[local] = item;
    wg_valid[local] = valid;
    wg_nonzero_error[local] = select(0u, 1u, err > 1.0e-8);
    wg_nonzero_spline[local] = select(0u, 1u, spline_magnitude > 1.0e-8);
    wg_nonzero_volume[local] = select(0u, 1u, volume_magnitude > 1.0e-8);
    wg_sum_spline_magnitude[local] = spline_magnitude;
    wg_max_spline_magnitude[local] = spline_magnitude;
    wg_sum_volume_magnitude[local] = volume_magnitude;
    wg_max_volume_magnitude[local] = volume_magnitude;
    workgroupBarrier();

    var stride = WORKGROUP_SIZE / 2u;
    loop {
        if stride == 0u {
            break;
        }
        if local < stride {
            let other = local + stride;
            if wg_valid[other] != 0u {
                wg_sum[local] = wg_sum[local] + wg_sum[other];
                wg_sum_sq[local] = wg_sum_sq[local] + wg_sum_sq[other];
                wg_valid[local] = wg_valid[local] + wg_valid[other];
                wg_nonzero_error[local] = wg_nonzero_error[local] + wg_nonzero_error[other];
                wg_nonzero_spline[local] = wg_nonzero_spline[local] + wg_nonzero_spline[other];
                wg_nonzero_volume[local] = wg_nonzero_volume[local] + wg_nonzero_volume[other];
                wg_sum_spline_magnitude[local] = wg_sum_spline_magnitude[local] + wg_sum_spline_magnitude[other];
                wg_sum_volume_magnitude[local] = wg_sum_volume_magnitude[local] + wg_sum_volume_magnitude[other];
            }
            if wg_max_error[other] > wg_max_error[local] {
                wg_max_error[local] = wg_max_error[other];
                wg_max_item[local] = wg_max_item[other];
            }
            if wg_max_spline_magnitude[other] > wg_max_spline_magnitude[local] {
                wg_max_spline_magnitude[local] = wg_max_spline_magnitude[other];
            }
            if wg_max_volume_magnitude[other] > wg_max_volume_magnitude[local] {
                wg_max_volume_magnitude[local] = wg_max_volume_magnitude[other];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if local == 0u {
        let stats_index = wid.x;
        s_partial_stats[stats_index].count = wg_valid[0];
        s_partial_stats[stats_index].sum_error = wg_sum[0];
        s_partial_stats[stats_index].sum_error_sq = wg_sum_sq[0];
        s_partial_stats[stats_index].max_error = wg_max_error[0];
        s_partial_stats[stats_index].worst_item = wg_max_item[0];
        s_partial_stats[stats_index].nonzero_error_count = wg_nonzero_error[0];
        s_partial_stats[stats_index].nonzero_spline_count = wg_nonzero_spline[0];
        s_partial_stats[stats_index].nonzero_volume_count = wg_nonzero_volume[0];
        s_partial_stats[stats_index].sum_spline_magnitude = wg_sum_spline_magnitude[0];
        s_partial_stats[stats_index].max_spline_magnitude = wg_max_spline_magnitude[0];
        s_partial_stats[stats_index].sum_volume_magnitude = wg_sum_volume_magnitude[0];
        s_partial_stats[stats_index].max_volume_magnitude = wg_max_volume_magnitude[0];
    }
}
