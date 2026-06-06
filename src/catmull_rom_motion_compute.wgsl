const WORKGROUP_SIZE: u32 = 128u;

struct MotionUniform {
    time01: f32,
    splat_count: u32,
    gaussian_tex_width: u32,
    knot_count: u32,
    knot_texture_width: u32,
    time_sampling: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0)
var<uniform> u_motion: MotionUniform;

@group(0) @binding(1)
var t_base_gaussian: texture_2d<u32>;

@group(0) @binding(2)
var t_knots: texture_2d<f32>;

@group(0) @binding(3)
var t_output_gaussian: texture_storage_2d<rgba32uint, write>;

fn knot_texel_coord(flat_index: u32) -> vec2<i32> {
    let width = max(u_motion.knot_texture_width, 1u);
    return vec2<i32>(
        i32(flat_index % width),
        i32(flat_index / width)
    );
}

fn load_delta_knot(splat_index: u32, knot_index: u32) -> vec3<f32> {
    let flat_index = splat_index * u_motion.knot_count + knot_index;
    return textureLoad(t_knots, knot_texel_coord(flat_index), 0).xyz;
}

fn sample_catmull_rom_delta(splat_index: u32, time01: f32) -> vec3<f32> {
    let knot_count = max(u_motion.knot_count, 1u);
    var scaled = fract(time01) * f32(knot_count);
    if u_motion.time_sampling == 1u {
        scaled = clamp(time01, 0.0, 1.0) * f32(max(knot_count, 1u) - 1u);
    }
    let segment = min(u32(floor(scaled)), knot_count - 1u);
    let u = scaled - f32(segment);
    let u2 = u * u;
    let u3 = u2 * u;

    var i0 = (segment + knot_count - 1u) % knot_count;
    var i1 = segment % knot_count;
    var i2 = (segment + 1u) % knot_count;
    var i3 = (segment + 2u) % knot_count;
    if u_motion.time_sampling == 1u {
        i0 = 0u;
        if segment > 0u {
            i0 = segment - 1u;
        }
        i1 = segment;
        i2 = min(segment + 1u, knot_count - 1u);
        i3 = min(segment + 2u, knot_count - 1u);
    }

    let p0 = load_delta_knot(splat_index, i0);
    let p1 = load_delta_knot(splat_index, i1);
    let p2 = load_delta_knot(splat_index, i2);
    let p3 = load_delta_knot(splat_index, i3);

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
    let delta = sample_catmull_rom_delta(idx, u_motion.time01);
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
