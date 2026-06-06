const WORKGROUP_SIZE: u32 = 128u;

struct TextureCompareUniform {
    item_count: u32,
    splat_count: u32,
    gaussian_tex_width: u32,
    sample_stride: u32,
    sample_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct TextureComparePartialStats {
    count: u32,
    worst_item: u32,
    nonzero_error_count: u32,
    _pad0: u32,
    sum_error: f32,
    sum_error_sq: f32,
    max_error: f32,
    sum_cat_magnitude: f32,
    max_cat_magnitude: f32,
    sum_volume_magnitude: f32,
    max_volume_magnitude: f32,
    _pad1: f32,
}

@group(0) @binding(0)
var<uniform> u_compare: TextureCompareUniform;

@group(0) @binding(1)
var t_catmull_rom_gaussian: texture_2d<u32>;

@group(0) @binding(2)
var t_volume_gaussian: texture_2d<u32>;

@group(0) @binding(3)
var<storage, read_write> s_partial_stats: array<TextureComparePartialStats>;

@group(0) @binding(4)
var<storage, read_write> s_sample_errors: array<f32>;

var<workgroup> wg_sum: array<f32, 128>;
var<workgroup> wg_sum_sq: array<f32, 128>;
var<workgroup> wg_max_error: array<f32, 128>;
var<workgroup> wg_max_item: array<u32, 128>;
var<workgroup> wg_valid: array<u32, 128>;
var<workgroup> wg_nonzero_error: array<u32, 128>;
var<workgroup> wg_sum_cat_magnitude: array<f32, 128>;
var<workgroup> wg_max_cat_magnitude: array<f32, 128>;
var<workgroup> wg_sum_volume_magnitude: array<f32, 128>;
var<workgroup> wg_max_volume_magnitude: array<f32, 128>;

fn gaussian_texel0_coord(splat_index: u32) -> vec2<i32> {
    let splats_per_row = max(u_compare.gaussian_tex_width / 2u, 1u);
    let tex_u = (splat_index % splats_per_row) * 2u;
    let tex_v = splat_index / splats_per_row;
    return vec2<i32>(i32(tex_u), i32(tex_v));
}

fn load_catmull_rom_mean(splat_index: u32) -> vec3<f32> {
    let encoded = textureLoad(t_catmull_rom_gaussian, gaussian_texel0_coord(splat_index), 0);
    return vec3<f32>(
        bitcast<f32>(encoded.x),
        bitcast<f32>(encoded.y),
        bitcast<f32>(encoded.z)
    );
}

fn load_volume_mean(splat_index: u32) -> vec3<f32> {
    let encoded = textureLoad(t_volume_gaussian, gaussian_texel0_coord(splat_index), 0);
    return vec3<f32>(
        bitcast<f32>(encoded.x),
        bitcast<f32>(encoded.y),
        bitcast<f32>(encoded.z)
    );
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
    var cat_magnitude = 0.0;
    var volume_magnitude = 0.0;
    var valid = 0u;

    if item < u_compare.item_count && item < u_compare.splat_count {
        let cat_mean = load_catmull_rom_mean(item);
        let volume_mean = load_volume_mean(item);
        err = length(cat_mean - volume_mean);
        cat_magnitude = length(cat_mean);
        volume_magnitude = length(volume_mean);
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
    wg_sum_cat_magnitude[local] = cat_magnitude;
    wg_max_cat_magnitude[local] = cat_magnitude;
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
            wg_sum[local] = wg_sum[local] + wg_sum[other];
            wg_sum_sq[local] = wg_sum_sq[local] + wg_sum_sq[other];
            wg_valid[local] = wg_valid[local] + wg_valid[other];
            wg_nonzero_error[local] = wg_nonzero_error[local] + wg_nonzero_error[other];
            wg_sum_cat_magnitude[local] = wg_sum_cat_magnitude[local] + wg_sum_cat_magnitude[other];
            wg_sum_volume_magnitude[local] = wg_sum_volume_magnitude[local] + wg_sum_volume_magnitude[other];

            if wg_max_error[other] > wg_max_error[local] {
                wg_max_error[local] = wg_max_error[other];
                wg_max_item[local] = wg_max_item[other];
            }
            if wg_max_cat_magnitude[other] > wg_max_cat_magnitude[local] {
                wg_max_cat_magnitude[local] = wg_max_cat_magnitude[other];
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
        s_partial_stats[stats_index].worst_item = wg_max_item[0];
        s_partial_stats[stats_index].nonzero_error_count = wg_nonzero_error[0];
        s_partial_stats[stats_index].sum_error = wg_sum[0];
        s_partial_stats[stats_index].sum_error_sq = wg_sum_sq[0];
        s_partial_stats[stats_index].max_error = wg_max_error[0];
        s_partial_stats[stats_index].sum_cat_magnitude = wg_sum_cat_magnitude[0];
        s_partial_stats[stats_index].max_cat_magnitude = wg_max_cat_magnitude[0];
        s_partial_stats[stats_index].sum_volume_magnitude = wg_sum_volume_magnitude[0];
        s_partial_stats[stats_index].max_volume_magnitude = wg_max_volume_magnitude[0];
    }
}
