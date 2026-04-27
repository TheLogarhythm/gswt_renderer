const WORKGROUP_SIZE: u32 = 128u;
const MAX_GRID_FEATURES: u32 = 512u;
const MAX_MLP_WIDTH: u32 = 512u;
const QUAT_NORM_EPS: f32 = 1.0e-12;

struct NetworkMeta {
    counts0: vec4<u32>, // n_grid_levels, n_planes_per_level, feature_dim, net_width
    counts1: vec4<u32>, // n_time_frames, feature_layers, pos_layers, rot_layers
    aabb_max: vec4<f32>,
    aabb_min: vec4<f32>,
    scale_and_pad: vec4<f32>, // scale_factor
}

struct AnimationUniform {
    time01: f32,
    splat_count: u32,
    gaussian_tex_width: u32,
    _pad0: u32,
}

struct PlaneDesc {
    width: u32,
    height: u32,
    channels: u32,
    data_offset: u32,
}

struct LayerDesc {
    in_features: u32,
    out_features: u32,
    has_relu_before: u32,
    weight_offset: u32,
    bias_offset: u32,
}

@group(0) @binding(0)
var<uniform> u_meta: NetworkMeta;
@group(0) @binding(1)
var<uniform> u_anim: AnimationUniform;

@group(0) @binding(2)
var<storage, read> s_orig_means: array<vec4<f32>>;
@group(0) @binding(3)
var<storage, read> s_orig_quats: array<vec4<f32>>;
@group(0) @binding(4)
var<storage, read> s_base_tile_means: array<vec4<f32>>;
@group(0) @binding(5)
var<storage, read> s_base_scales: array<vec4<f32>>;
@group(0) @binding(6)
var<storage, read> s_base_rgba: array<vec4<u32>>;

@group(0) @binding(7)
var<storage, read> s_plane_descs: array<PlaneDesc>;
@group(0) @binding(8)
var<storage, read> s_plane_data: array<f32>;

@group(0) @binding(9)
var<storage, read> s_feature_layers: array<LayerDesc>;
@group(0) @binding(10)
var<storage, read> s_feature_weights: array<f32>;
@group(0) @binding(11)
var<storage, read> s_feature_bias: array<f32>;

@group(0) @binding(12)
var<storage, read> s_pos_layers: array<LayerDesc>;
@group(0) @binding(13)
var<storage, read> s_pos_weights: array<f32>;
@group(0) @binding(14)
var<storage, read> s_pos_bias: array<f32>;

@group(0) @binding(15)
var<storage, read> s_rot_layers: array<LayerDesc>;
@group(0) @binding(16)
var<storage, read> s_rot_weights: array<f32>;
@group(0) @binding(17)
var<storage, read> s_rot_bias: array<f32>;

@group(0) @binding(18)
var t_output_gaussian: texture_storage_2d<rgba32uint, write>;

fn plane_uv_from_idx(plane_idx: u32, coords4: vec4<f32>) -> vec2<f32> {
    switch plane_idx {
        case 0u: { return vec2<f32>(coords4.x, coords4.y); } // (0, 1)
        case 1u: { return vec2<f32>(coords4.x, coords4.z); } // (0, 2)
        case 2u: { return vec2<f32>(coords4.x, coords4.w); } // (0, 3)
        case 3u: { return vec2<f32>(coords4.y, coords4.z); } // (1, 2)
        case 4u: { return vec2<f32>(coords4.y, coords4.w); } // (1, 3)
        default: { return vec2<f32>(coords4.z, coords4.w); } // (2, 3)
    }
}

fn sample_plane_bilinear(desc: PlaneDesc, coord_u: f32, coord_v: f32, channel: u32) -> f32 {
    let w = desc.width;
    let h = desc.height;

    var x: f32 = 0.0;
    if w > 1u {
        x = clamp((coord_u + 1.0) * 0.5 * (f32(w) - 1.0), 0.0, f32(w) - 1.0);
    }

    var y: f32 = 0.0;
    if h > 1u {
        y = clamp((coord_v + 1.0) * 0.5 * (f32(h) - 1.0), 0.0, f32(h) - 1.0);
    }

    let x0 = u32(floor(x));
    let y0 = u32(floor(y));
    let x1 = min(x0 + 1u, w - 1u);
    let y1 = min(y0 + 1u, h - 1u);

    let tx = x - f32(x0);
    let ty = y - f32(y0);
    let w00 = (1.0 - tx) * (1.0 - ty);
    let w10 = tx * (1.0 - ty);
    let w01 = (1.0 - tx) * ty;
    let w11 = tx * ty;

    let base = desc.data_offset + channel * (h * w);
    let v00 = s_plane_data[base + y0 * w + x0];
    let v10 = s_plane_data[base + y0 * w + x1];
    let v01 = s_plane_data[base + y1 * w + x0];
    let v11 = s_plane_data[base + y1 * w + x1];
    return v00 * w00 + v10 * w10 + v01 * w01 + v11 * w11;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= u_anim.splat_count {
        return;
    }

    let n_grid_levels = u_meta.counts0.x;
    let n_planes_per_level = u_meta.counts0.y;
    let feature_dim = u_meta.counts0.z;
    let net_width = u_meta.counts0.w;

    let orig = s_orig_means[idx].xyz;
    let base_tile = s_base_tile_means[idx].xyz;
    let orig_q = s_orig_quats[idx];
    let base_scale = s_base_scales[idx].xyz;
    let base_rgba = s_base_rgba[idx];

    var coords4 = vec4<f32>(0.0, 0.0, 0.0, clamp(u_anim.time01, 0.0, 1.0));
    for (var axis: u32 = 0u; axis < 3u; axis = axis + 1u) {
        let aabb_max = u_meta.aabb_max[axis];
        let aabb_min = u_meta.aabb_min[axis];
        let denom = aabb_min - aabb_max;
        coords4[axis] = (orig[axis] - aabb_max) * (2.0 / denom) - 1.0;
    }

    var grid_feature: array<f32, MAX_GRID_FEATURES>;
    var grid_offset: u32 = 0u;
    for (var level: u32 = 0u; level < n_grid_levels; level = level + 1u) {
        for (var c: u32 = 0u; c < feature_dim; c = c + 1u) {
            var level_val = 1.0;
            for (var p: u32 = 0u; p < n_planes_per_level; p = p + 1u) {
                let plane_idx = level * n_planes_per_level + p;
                let desc = s_plane_descs[plane_idx];
                let uv = plane_uv_from_idx(p, coords4);
                level_val = level_val * sample_plane_bilinear(desc, uv.x, uv.y, c);
            }
            grid_feature[grid_offset + c] = level_val;
        }
        grid_offset = grid_offset + feature_dim;
    }

    var feat_a: array<f32, MAX_MLP_WIDTH>;
    var feat_b: array<f32, MAX_MLP_WIDTH>;
    let feature_input_len = n_grid_levels * feature_dim;
    for (var i: u32 = 0u; i < feature_input_len; i = i + 1u) {
        feat_a[i] = grid_feature[i];
    }

    var feature_in_a = true;
    var feature_out_len = feature_input_len;
    for (var li: u32 = 0u; li < u_meta.counts1.y; li = li + 1u) {
        let layer = s_feature_layers[li];
        if feature_in_a {
            for (var o: u32 = 0u; o < layer.out_features; o = o + 1u) {
                var sum = s_feature_bias[layer.bias_offset + o];
                let row = layer.weight_offset + o * layer.in_features;
                for (var j: u32 = 0u; j < layer.in_features; j = j + 1u) {
                    var x = feat_a[j];
                    if layer.has_relu_before != 0u {
                        x = max(0.0, x);
                    }
                    sum = sum + s_feature_weights[row + j] * x;
                }
                feat_b[o] = sum;
            }
        } else {
            for (var o: u32 = 0u; o < layer.out_features; o = o + 1u) {
                var sum = s_feature_bias[layer.bias_offset + o];
                let row = layer.weight_offset + o * layer.in_features;
                for (var j: u32 = 0u; j < layer.in_features; j = j + 1u) {
                    var x = feat_b[j];
                    if layer.has_relu_before != 0u {
                        x = max(0.0, x);
                    }
                    sum = sum + s_feature_weights[row + j] * x;
                }
                feat_a[o] = sum;
            }
        }
        feature_out_len = layer.out_features;
        feature_in_a = !feature_in_a;
    }

    var hidden: array<f32, MAX_MLP_WIDTH>;
    for (var i: u32 = 0u; i < feature_out_len; i = i + 1u) {
        if feature_in_a {
            hidden[i] = feat_a[i];
        } else {
            hidden[i] = feat_b[i];
        }
    }

    var mlp_a: array<f32, MAX_MLP_WIDTH>;
    var mlp_b: array<f32, MAX_MLP_WIDTH>;

    for (var i: u32 = 0u; i < net_width; i = i + 1u) {
        mlp_a[i] = hidden[i];
    }
    var pos_in_a = true;
    for (var li: u32 = 0u; li < u_meta.counts1.z; li = li + 1u) {
        let layer = s_pos_layers[li];
        if pos_in_a {
            for (var o: u32 = 0u; o < layer.out_features; o = o + 1u) {
                var sum = s_pos_bias[layer.bias_offset + o];
                let row = layer.weight_offset + o * layer.in_features;
                for (var j: u32 = 0u; j < layer.in_features; j = j + 1u) {
                    var x = mlp_a[j];
                    if layer.has_relu_before != 0u {
                        x = max(0.0, x);
                    }
                    sum = sum + s_pos_weights[row + j] * x;
                }
                mlp_b[o] = sum;
            }
        } else {
            for (var o: u32 = 0u; o < layer.out_features; o = o + 1u) {
                var sum = s_pos_bias[layer.bias_offset + o];
                let row = layer.weight_offset + o * layer.in_features;
                for (var j: u32 = 0u; j < layer.in_features; j = j + 1u) {
                    var x = mlp_b[j];
                    if layer.has_relu_before != 0u {
                        x = max(0.0, x);
                    }
                    sum = sum + s_pos_weights[row + j] * x;
                }
                mlp_a[o] = sum;
            }
        }
        pos_in_a = !pos_in_a;
    }
    var dx = vec3<f32>(0.0, 0.0, 0.0);
    if pos_in_a {
        dx = vec3<f32>(mlp_a[0], mlp_a[1], mlp_a[2]);
    } else {
        dx = vec3<f32>(mlp_b[0], mlp_b[1], mlp_b[2]);
    }

    for (var i: u32 = 0u; i < net_width; i = i + 1u) {
        mlp_a[i] = hidden[i];
    }
    var rot_in_a = true;
    for (var li: u32 = 0u; li < u_meta.counts1.w; li = li + 1u) {
        let layer = s_rot_layers[li];
        if rot_in_a {
            for (var o: u32 = 0u; o < layer.out_features; o = o + 1u) {
                var sum = s_rot_bias[layer.bias_offset + o];
                let row = layer.weight_offset + o * layer.in_features;
                for (var j: u32 = 0u; j < layer.in_features; j = j + 1u) {
                    var x = mlp_a[j];
                    if layer.has_relu_before != 0u {
                        x = max(0.0, x);
                    }
                    sum = sum + s_rot_weights[row + j] * x;
                }
                mlp_b[o] = sum;
            }
        } else {
            for (var o: u32 = 0u; o < layer.out_features; o = o + 1u) {
                var sum = s_rot_bias[layer.bias_offset + o];
                let row = layer.weight_offset + o * layer.in_features;
                for (var j: u32 = 0u; j < layer.in_features; j = j + 1u) {
                    var x = mlp_b[j];
                    if layer.has_relu_before != 0u {
                        x = max(0.0, x);
                    }
                    sum = sum + s_rot_weights[row + j] * x;
                }
                mlp_a[o] = sum;
            }
        }
        rot_in_a = !rot_in_a;
    }
    var dr = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    if rot_in_a {
        dr = vec4<f32>(mlp_a[0], mlp_a[1], mlp_a[2], mlp_a[3]);
    } else {
        dr = vec4<f32>(mlp_b[0], mlp_b[1], mlp_b[2], mlp_b[3]);
    }

    let new_tile = base_tile + dx * u_meta.scale_and_pad.x;
    let q_raw = orig_q + dr;
    let q = q_raw / max(length(q_raw), QUAT_NORM_EPS);

    let r = mat3x3<f32>(
        vec3<f32>(
            1.0 - 2.0 * (q.z * q.z + q.w * q.w),
            2.0 * (q.y * q.z + q.x * q.w),
            2.0 * (q.y * q.w - q.x * q.z)
        ),
        vec3<f32>(
            2.0 * (q.y * q.z - q.x * q.w),
            1.0 - 2.0 * (q.y * q.y + q.w * q.w),
            2.0 * (q.z * q.w + q.x * q.y)
        ),
        vec3<f32>(
            2.0 * (q.y * q.w + q.x * q.z),
            2.0 * (q.z * q.w - q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
    );
    let s = mat3x3<f32>(
        vec3<f32>(base_scale.x, 0.0, 0.0),
        vec3<f32>(0.0, base_scale.y, 0.0),
        vec3<f32>(0.0, 0.0, base_scale.z)
    );
    let m = r * s;

    let sigma0 = m[0][0] * m[0][0] + m[1][0] * m[1][0] + m[2][0] * m[2][0];
    let sigma1 = m[0][0] * m[0][1] + m[1][0] * m[1][1] + m[2][0] * m[2][1];
    let sigma2 = m[0][0] * m[0][2] + m[1][0] * m[1][2] + m[2][0] * m[2][2];
    let sigma3 = m[0][1] * m[0][1] + m[1][1] * m[1][1] + m[2][1] * m[2][1];
    let sigma4 = m[0][1] * m[0][2] + m[1][1] * m[1][2] + m[2][1] * m[2][2];
    let sigma5 = m[0][2] * m[0][2] + m[1][2] * m[1][2] + m[2][2] * m[2][2];

    let cov0 = pack2x16float(vec2<f32>(4.0 * sigma0, 4.0 * sigma1));
    let cov1 = pack2x16float(vec2<f32>(4.0 * sigma2, 4.0 * sigma3));
    let cov2 = pack2x16float(vec2<f32>(4.0 * sigma4, 4.0 * sigma5));

    let splats_per_row = u_anim.gaussian_tex_width / 2u;
    let tex_u = (idx % splats_per_row) * 2u;
    let tex_v = idx / splats_per_row;

    textureStore(
        t_output_gaussian,
        vec2<i32>(i32(tex_u), i32(tex_v)),
        vec4<u32>(
            bitcast<u32>(new_tile.x),
            bitcast<u32>(new_tile.y),
            bitcast<u32>(new_tile.z),
            base_rgba.x
        )
    );
    textureStore(
        t_output_gaussian,
        vec2<i32>(i32(tex_u + 1u), i32(tex_v)),
        vec4<u32>(cov0, cov1, cov2, base_rgba.y)
    );
}
