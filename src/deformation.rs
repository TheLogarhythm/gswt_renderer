use std::fmt;

const PLANE_DIMS_4D: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
const QUAT_NORM_EPS: f32 = 1.0e-12;

#[derive(Debug, Clone, PartialEq)]
pub enum DeformationError {
    BadMagic([u8; 4]),
    UnsupportedVersion(u32),
    UnexpectedEof {
        offset: usize,
        needed: usize,
        len: usize,
    },
    InvalidFormat(String),
    ShapeMismatch(String),
}

impl fmt::Display for DeformationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeformationError::BadMagic(magic) => {
                write!(f, "bad DFWT magic: {:?} (expected b\"DFWT\")", magic)
            }
            DeformationError::UnsupportedVersion(version) => {
                write!(f, "unsupported DFWT version: {}", version)
            }
            DeformationError::UnexpectedEof {
                offset,
                needed,
                len,
            } => write!(
                f,
                "unexpected EOF at offset {}, needed {} bytes, buffer len {}",
                offset, needed, len
            ),
            DeformationError::InvalidFormat(msg) => write!(f, "invalid DFWT format: {}", msg),
            DeformationError::ShapeMismatch(msg) => write!(f, "shape mismatch: {}", msg),
        }
    }
}

impl std::error::Error for DeformationError {}

struct ByteReader<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> ByteReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, offset: 0 }
    }

    fn read_exact(&mut self, n: usize) -> Result<&'a [u8], DeformationError> {
        if self.offset + n > self.data.len() {
            return Err(DeformationError::UnexpectedEof {
                offset: self.offset,
                needed: n,
                len: self.data.len(),
            });
        }
        let start = self.offset;
        self.offset += n;
        Ok(&self.data[start..start + n])
    }

    fn read_u32(&mut self) -> Result<u32, DeformationError> {
        let bytes = self.read_exact(4)?;
        let mut arr = [0_u8; 4];
        arr.copy_from_slice(bytes);
        Ok(u32::from_le_bytes(arr))
    }

    fn read_f32(&mut self) -> Result<f32, DeformationError> {
        let bytes = self.read_exact(4)?;
        let mut arr = [0_u8; 4];
        arr.copy_from_slice(bytes);
        Ok(f32::from_le_bytes(arr))
    }

    fn read_f32_vec(&mut self, n: usize) -> Result<Vec<f32>, DeformationError> {
        let bytes = self.read_exact(n * 4)?;
        let mut out = Vec::with_capacity(n);
        for chunk in bytes.chunks_exact(4) {
            let mut arr = [0_u8; 4];
            arr.copy_from_slice(chunk);
            out.push(f32::from_le_bytes(arr));
        }
        Ok(out)
    }

    fn remaining(&self) -> usize {
        self.data.len() - self.offset
    }
}

#[derive(Debug, Clone)]
struct GridPlane {
    channels: usize,
    height: usize,
    width: usize,
    data: Vec<f32>, // [C, H, W]
}

impl GridPlane {
    fn sample_bilinear_border(&self, coord_u: f32, coord_v: f32) -> Vec<f32> {
        let mut out = vec![0.0_f32; self.channels];
        let w = self.width as f32;
        let h = self.height as f32;

        let x = if self.width <= 1 {
            0.0
        } else {
            ((coord_u + 1.0) * 0.5 * (w - 1.0)).clamp(0.0, w - 1.0)
        };
        let y = if self.height <= 1 {
            0.0
        } else {
            ((coord_v + 1.0) * 0.5 * (h - 1.0)).clamp(0.0, h - 1.0)
        };

        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(self.width.saturating_sub(1));
        let y1 = (y0 + 1).min(self.height.saturating_sub(1));

        let tx = x - x0 as f32;
        let ty = y - y0 as f32;
        let w00 = (1.0 - tx) * (1.0 - ty);
        let w10 = tx * (1.0 - ty);
        let w01 = (1.0 - tx) * ty;
        let w11 = tx * ty;

        for (c, out_c) in out.iter_mut().enumerate().take(self.channels) {
            let base = c * self.height * self.width;
            let v00 = self.data[base + y0 * self.width + x0];
            let v10 = self.data[base + y0 * self.width + x1];
            let v01 = self.data[base + y1 * self.width + x0];
            let v11 = self.data[base + y1 * self.width + x1];
            *out_c = v00 * w00 + v10 * w10 + v01 * w01 + v11 * w11;
        }

        out
    }
}

#[derive(Debug, Clone)]
struct MlpLayer {
    in_features: usize,
    out_features: usize,
    has_relu_before: bool,
    weights: Vec<f32>, // [out, in], row-major
    bias: Vec<f32>,    // [out]
}

impl MlpLayer {
    fn forward(&self, input: &[f32]) -> Result<Vec<f32>, DeformationError> {
        if input.len() != self.in_features {
            return Err(DeformationError::ShapeMismatch(format!(
                "MLP layer input len {} != expected {}",
                input.len(),
                self.in_features
            )));
        }

        let mut working = if self.has_relu_before {
            input.iter().map(|v| v.max(0.0)).collect::<Vec<f32>>()
        } else {
            input.to_vec()
        };

        // Avoid realloc in common path.
        if !self.has_relu_before {
            // keep input already copied
        } else {
            // already relu-transformed above
        }

        let mut out = vec![0.0_f32; self.out_features];
        for (o, out_o) in out.iter_mut().enumerate().take(self.out_features) {
            let row_start = o * self.in_features;
            let row = &self.weights[row_start..row_start + self.in_features];
            let dot = row
                .iter()
                .zip(working.iter())
                .map(|(w, x)| w * x)
                .sum::<f32>();
            *out_o = dot + self.bias[o];
        }
        working.clear();
        Ok(out)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    layers: Vec<MlpLayer>,
}

impl Mlp {
    fn from_reader(reader: &mut ByteReader<'_>, name: &str) -> Result<Self, DeformationError> {
        let n_layers = reader.read_u32()? as usize;
        if n_layers == 0 {
            return Err(DeformationError::InvalidFormat(format!(
                "MLP '{}' has zero layers",
                name
            )));
        }

        let mut layers = Vec::with_capacity(n_layers);
        for layer_idx in 0..n_layers {
            let in_features = reader.read_u32()? as usize;
            let out_features = reader.read_u32()? as usize;
            let has_relu_before_raw = reader.read_u32()?;
            let has_relu_before = match has_relu_before_raw {
                0 => false,
                1 => true,
                _ => {
                    return Err(DeformationError::InvalidFormat(format!(
                        "MLP '{}' layer {} has invalid has_relu_before value {}",
                        name, layer_idx, has_relu_before_raw
                    )));
                }
            };
            if in_features == 0 || out_features == 0 {
                return Err(DeformationError::InvalidFormat(format!(
                    "MLP '{}' layer {} has zero dimension in={} out={}",
                    name, layer_idx, in_features, out_features
                )));
            }
            let weight_len = in_features * out_features;
            let weights = reader.read_f32_vec(weight_len)?;
            let bias = reader.read_f32_vec(out_features)?;

            layers.push(MlpLayer {
                in_features,
                out_features,
                has_relu_before,
                weights,
                bias,
            });
        }

        Ok(Self { layers })
    }

    fn forward(&self, input: &[f32]) -> Result<Vec<f32>, DeformationError> {
        let mut cur = input.to_vec();
        for layer in &self.layers {
            cur = layer.forward(&cur)?;
        }
        Ok(cur)
    }

    fn input_dim(&self) -> usize {
        self.layers
            .first()
            .map(|l| l.in_features)
            .unwrap_or_default()
    }

    fn output_dim(&self) -> usize {
        self.layers
            .last()
            .map(|l| l.out_features)
            .unwrap_or_default()
    }
}

#[derive(Debug, Clone)]
pub struct DeformationMetadata {
    pub n_grid_levels: usize,
    pub n_planes_per_level: usize,
    pub feature_dim: usize,
    pub net_width: usize,
    pub n_time_frames: usize,
    pub aabb_max: [f32; 3],
    pub aabb_min: [f32; 3],
    pub scale_factor: f32,
    pub source_offset: [f32; 3],
}

#[derive(Debug, Clone)]
pub struct DeformationNetwork {
    metadata: DeformationMetadata,
    grid_levels: Vec<Vec<GridPlane>>,
    feature_out: Mlp,
    pos_deform: Mlp,
    rotations_deform: Mlp,
}

#[derive(Debug, Clone, Copy)]
pub struct PackedPlaneDesc {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub data_offset: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct PackedMlpLayerDesc {
    pub in_features: u32,
    pub out_features: u32,
    pub has_relu_before: u32,
    pub weight_offset: u32,
    pub bias_offset: u32,
}

#[derive(Debug, Clone)]
pub struct PackedDeformationNetwork {
    pub metadata: DeformationMetadata,
    pub plane_descs: Vec<PackedPlaneDesc>,
    pub plane_data: Vec<f32>,
    pub feature_layers: Vec<PackedMlpLayerDesc>,
    pub feature_weights: Vec<f32>,
    pub feature_bias: Vec<f32>,
    pub pos_layers: Vec<PackedMlpLayerDesc>,
    pub pos_weights: Vec<f32>,
    pub pos_bias: Vec<f32>,
    pub rot_layers: Vec<PackedMlpLayerDesc>,
    pub rot_weights: Vec<f32>,
    pub rot_bias: Vec<f32>,
}

fn usize_to_u32(value: usize, label: &str) -> Result<u32, DeformationError> {
    u32::try_from(value).map_err(|_| {
        DeformationError::InvalidFormat(format!(
            "{}={} does not fit into u32 for GPU packing",
            label, value
        ))
    })
}

fn pack_mlp_layers(
    mlp: &Mlp,
    label: &str,
) -> Result<(Vec<PackedMlpLayerDesc>, Vec<f32>, Vec<f32>), DeformationError> {
    let mut layer_descs = Vec::with_capacity(mlp.layers.len());
    let mut all_weights: Vec<f32> = Vec::new();
    let mut all_bias: Vec<f32> = Vec::new();
    let mut weight_offset: usize = 0;
    let mut bias_offset: usize = 0;

    for (idx, layer) in mlp.layers.iter().enumerate() {
        let expected_weight_len = layer.in_features * layer.out_features;
        if layer.weights.len() != expected_weight_len || layer.bias.len() != layer.out_features {
            return Err(DeformationError::ShapeMismatch(format!(
                "MLP '{}' layer {} has invalid shape: in={} out={} weights={} bias={}",
                label,
                idx,
                layer.in_features,
                layer.out_features,
                layer.weights.len(),
                layer.bias.len()
            )));
        }
        layer_descs.push(PackedMlpLayerDesc {
            in_features: usize_to_u32(layer.in_features, "in_features")?,
            out_features: usize_to_u32(layer.out_features, "out_features")?,
            has_relu_before: if layer.has_relu_before { 1 } else { 0 },
            weight_offset: usize_to_u32(weight_offset, "weight_offset")?,
            bias_offset: usize_to_u32(bias_offset, "bias_offset")?,
        });
        all_weights.extend_from_slice(layer.weights.as_slice());
        all_bias.extend_from_slice(layer.bias.as_slice());
        weight_offset += layer.weights.len();
        bias_offset += layer.bias.len();
    }

    Ok((layer_descs, all_weights, all_bias))
}

impl DeformationNetwork {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DeformationError> {
        let mut reader = ByteReader::new(bytes);

        let magic = reader.read_exact(4)?;
        if magic != b"DFWT" {
            let mut got = [0_u8; 4];
            got.copy_from_slice(magic);
            return Err(DeformationError::BadMagic(got));
        }

        let version = reader.read_u32()?;
        if version != 1 {
            return Err(DeformationError::UnsupportedVersion(version));
        }

        let n_grid_levels = reader.read_u32()? as usize;
        let n_planes_per_level = reader.read_u32()? as usize;
        let feature_dim = reader.read_u32()? as usize;
        let net_width = reader.read_u32()? as usize;
        let n_time_frames = reader.read_u32()? as usize;

        let aabb_max_v = reader.read_f32_vec(3)?;
        let aabb_min_v = reader.read_f32_vec(3)?;
        let scale_factor = reader.read_f32()?;
        let source_offset_v = reader.read_f32_vec(3)?;
        let _pad = reader.read_u32()?;

        if n_grid_levels == 0 {
            return Err(DeformationError::InvalidFormat(
                "n_grid_levels cannot be zero".to_string(),
            ));
        }
        if n_planes_per_level != PLANE_DIMS_4D.len() {
            return Err(DeformationError::InvalidFormat(format!(
                "expected {} planes per level, got {}",
                PLANE_DIMS_4D.len(),
                n_planes_per_level
            )));
        }
        if feature_dim == 0 || net_width == 0 || n_time_frames == 0 {
            return Err(DeformationError::InvalidFormat(format!(
                "invalid dimensions feature_dim={} net_width={} n_time_frames={}",
                feature_dim, net_width, n_time_frames
            )));
        }

        let aabb_max = [aabb_max_v[0], aabb_max_v[1], aabb_max_v[2]];
        let aabb_min = [aabb_min_v[0], aabb_min_v[1], aabb_min_v[2]];
        for i in 0..3 {
            if (aabb_min[i] - aabb_max[i]).abs() < 1.0e-8 {
                return Err(DeformationError::InvalidFormat(format!(
                    "degenerate aabb axis {}: min={} max={}",
                    i, aabb_min[i], aabb_max[i]
                )));
            }
        }

        let mut grid_levels: Vec<Vec<GridPlane>> = Vec::with_capacity(n_grid_levels);
        for level_idx in 0..n_grid_levels {
            let mut planes = Vec::with_capacity(n_planes_per_level);
            for plane_idx in 0..n_planes_per_level {
                let height = reader.read_u32()? as usize;
                let width = reader.read_u32()? as usize;
                if height == 0 || width == 0 {
                    return Err(DeformationError::InvalidFormat(format!(
                        "grid plane[{}/{}] has invalid shape {}x{}",
                        level_idx, plane_idx, height, width
                    )));
                }
                let data = reader.read_f32_vec(feature_dim * height * width)?;
                planes.push(GridPlane {
                    channels: feature_dim,
                    height,
                    width,
                    data,
                });
            }
            grid_levels.push(planes);
        }

        let feature_out = Mlp::from_reader(&mut reader, "feature_out")?;
        let pos_deform = Mlp::from_reader(&mut reader, "pos_deform")?;
        let rotations_deform = Mlp::from_reader(&mut reader, "rotations_deform")?;

        if reader.remaining() != 0 {
            return Err(DeformationError::InvalidFormat(format!(
                "trailing bytes after parse: {}",
                reader.remaining()
            )));
        }

        let grid_feature_dim = n_grid_levels * feature_dim;
        if feature_out.input_dim() != grid_feature_dim {
            return Err(DeformationError::ShapeMismatch(format!(
                "feature_out input dim {} != grid feature dim {}",
                feature_out.input_dim(),
                grid_feature_dim
            )));
        }
        if feature_out.output_dim() != net_width {
            return Err(DeformationError::ShapeMismatch(format!(
                "feature_out output dim {} != net_width {}",
                feature_out.output_dim(),
                net_width
            )));
        }
        if pos_deform.input_dim() != net_width || pos_deform.output_dim() != 3 {
            return Err(DeformationError::ShapeMismatch(format!(
                "pos_deform dims in={} out={}, expected in={} out=3",
                pos_deform.input_dim(),
                pos_deform.output_dim(),
                net_width
            )));
        }
        if rotations_deform.input_dim() != net_width || rotations_deform.output_dim() != 4 {
            return Err(DeformationError::ShapeMismatch(format!(
                "rotations_deform dims in={} out={}, expected in={} out=4",
                rotations_deform.input_dim(),
                rotations_deform.output_dim(),
                net_width
            )));
        }

        let metadata = DeformationMetadata {
            n_grid_levels,
            n_planes_per_level,
            feature_dim,
            net_width,
            n_time_frames,
            aabb_max,
            aabb_min,
            scale_factor,
            source_offset: [source_offset_v[0], source_offset_v[1], source_offset_v[2]],
        };

        Ok(Self {
            metadata,
            grid_levels,
            feature_out,
            pos_deform,
            rotations_deform,
        })
    }

    pub fn metadata(&self) -> &DeformationMetadata {
        &self.metadata
    }

    pub fn pack_for_gpu(&self) -> Result<PackedDeformationNetwork, DeformationError> {
        let mut plane_descs: Vec<PackedPlaneDesc> =
            Vec::with_capacity(self.metadata.n_grid_levels * self.metadata.n_planes_per_level);
        let mut plane_data: Vec<f32> = Vec::new();
        let mut plane_offset = 0_usize;
        for (level_idx, level) in self.grid_levels.iter().enumerate() {
            if level.len() != self.metadata.n_planes_per_level {
                return Err(DeformationError::ShapeMismatch(format!(
                    "grid level {} has {} planes, expected {}",
                    level_idx,
                    level.len(),
                    self.metadata.n_planes_per_level
                )));
            }
            for (plane_idx, plane) in level.iter().enumerate() {
                if plane.channels != self.metadata.feature_dim {
                    return Err(DeformationError::ShapeMismatch(format!(
                        "grid plane[{}][{}] channels {} != feature_dim {}",
                        level_idx, plane_idx, plane.channels, self.metadata.feature_dim
                    )));
                }
                let expected_plane_len = plane.channels * plane.height * plane.width;
                if plane.data.len() != expected_plane_len {
                    return Err(DeformationError::ShapeMismatch(format!(
                        "grid plane[{}][{}] data len {} != expected {}",
                        level_idx,
                        plane_idx,
                        plane.data.len(),
                        expected_plane_len
                    )));
                }
                plane_descs.push(PackedPlaneDesc {
                    width: usize_to_u32(plane.width, "plane_width")?,
                    height: usize_to_u32(plane.height, "plane_height")?,
                    channels: usize_to_u32(plane.channels, "plane_channels")?,
                    data_offset: usize_to_u32(plane_offset, "plane_data_offset")?,
                });
                plane_data.extend_from_slice(plane.data.as_slice());
                plane_offset += plane.data.len();
            }
        }

        let (feature_layers, feature_weights, feature_bias) =
            pack_mlp_layers(&self.feature_out, "feature_out")?;
        let (pos_layers, pos_weights, pos_bias) = pack_mlp_layers(&self.pos_deform, "pos_deform")?;
        let (rot_layers, rot_weights, rot_bias) =
            pack_mlp_layers(&self.rotations_deform, "rotations_deform")?;

        Ok(PackedDeformationNetwork {
            metadata: self.metadata.clone(),
            plane_descs,
            plane_data,
            feature_layers,
            feature_weights,
            feature_bias,
            pos_layers,
            pos_weights,
            pos_bias,
            rot_layers,
            rot_weights,
            rot_bias,
        })
    }

    pub fn deform_single(
        &self,
        orig_means: [f32; 3],
        tile_means: [f32; 3],
        orig_quat: [f32; 4],
        time: f32,
    ) -> Result<([f32; 3], [f32; 4]), DeformationError> {
        let (dx, dr) = self.deform_delta_single(orig_means, time)?;

        let mut new_means = [0.0_f32; 3];
        for axis in 0..3 {
            new_means[axis] = tile_means[axis] + dx[axis] * self.metadata.scale_factor;
        }

        let raw_quat = [
            orig_quat[0] + dr[0],
            orig_quat[1] + dr[1],
            orig_quat[2] + dr[2],
            orig_quat[3] + dr[3],
        ];
        let new_quat = normalize_quaternion(raw_quat);

        Ok((new_means, new_quat))
    }

    pub fn deform_delta_single(
        &self,
        orig_means: [f32; 3],
        time: f32,
    ) -> Result<([f32; 3], [f32; 4]), DeformationError> {
        let mut coords_4d = [0.0_f32; 4];
        for axis in 0..3 {
            // Match Python normalize_aabb with stored ordering [aabb_max, aabb_min].
            let denom = self.metadata.aabb_min[axis] - self.metadata.aabb_max[axis];
            coords_4d[axis] =
                (orig_means[axis] - self.metadata.aabb_max[axis]) * (2.0 / denom) - 1.0;
        }
        coords_4d[3] = time;

        let mut grid_feature: Vec<f32> =
            Vec::with_capacity(self.metadata.n_grid_levels * self.metadata.feature_dim);
        for level in &self.grid_levels {
            let mut level_feature = vec![1.0_f32; self.metadata.feature_dim];
            for (plane_idx, plane) in level.iter().enumerate() {
                let (u_dim, v_dim) = PLANE_DIMS_4D[plane_idx];
                let sampled = plane.sample_bilinear_border(coords_4d[u_dim], coords_4d[v_dim]);
                for c in 0..self.metadata.feature_dim {
                    level_feature[c] *= sampled[c];
                }
            }
            grid_feature.extend(level_feature);
        }

        let hidden = self.feature_out.forward(&grid_feature)?;
        let dx = self.pos_deform.forward(&hidden)?;
        let dr = self.rotations_deform.forward(&hidden)?;

        Ok(([dx[0], dx[1], dx[2]], [dr[0], dr[1], dr[2], dr[3]]))
    }

    pub fn deform_batch(
        &self,
        orig_means: &[[f32; 3]],
        tile_means: &[[f32; 3]],
        orig_quats: &[[f32; 4]],
        time: f32,
    ) -> Result<(Vec<[f32; 3]>, Vec<[f32; 4]>), DeformationError> {
        if orig_means.len() != tile_means.len() || orig_means.len() != orig_quats.len() {
            return Err(DeformationError::ShapeMismatch(format!(
                "batch input lengths mismatch: orig_means={} tile_means={} orig_quats={}",
                orig_means.len(),
                tile_means.len(),
                orig_quats.len()
            )));
        }

        let n = orig_means.len();
        let mut out_means = Vec::with_capacity(n);
        let mut out_quats = Vec::with_capacity(n);

        for i in 0..n {
            let (means, quat) =
                self.deform_single(orig_means[i], tile_means[i], orig_quats[i], time)?;
            out_means.push(means);
            out_quats.push(quat);
        }

        Ok((out_means, out_quats))
    }
}

fn normalize_quaternion(q: [f32; 4]) -> [f32; 4] {
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let inv = 1.0 / norm.max(QUAT_NORM_EPS);
    [q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack_u32(v: u32, out: &mut Vec<u8>) {
        out.extend_from_slice(&v.to_le_bytes());
    }

    fn pack_f32(v: f32, out: &mut Vec<u8>) {
        out.extend_from_slice(&v.to_le_bytes());
    }

    fn pack_f32_slice(values: &[f32], out: &mut Vec<u8>) {
        for v in values {
            pack_f32(*v, out);
        }
    }

    fn build_minimal_valid_bin() -> Vec<u8> {
        let n_grid_levels = 1_u32;
        let n_planes = 6_u32;
        let feature_dim = 2_u32;
        let net_width = 3_u32;
        let n_time_frames = 4_u32;

        let mut out = Vec::new();
        out.extend_from_slice(b"DFWT");
        pack_u32(1, &mut out); // version
        pack_u32(n_grid_levels, &mut out);
        pack_u32(n_planes, &mut out);
        pack_u32(feature_dim, &mut out);
        pack_u32(net_width, &mut out);
        pack_u32(n_time_frames, &mut out);
        pack_f32_slice(&[1.0, 1.0, 1.0], &mut out); // aabb_max
        pack_f32_slice(&[-1.0, -1.0, -1.0], &mut out); // aabb_min
        pack_f32(1.0, &mut out); // scale_factor
        pack_f32_slice(&[0.0, 0.0, 0.0], &mut out); // source_offset
        pack_u32(0, &mut out); // pad

        // grid planes: [C=2, H=2, W=2]
        for plane_idx in 0..6 {
            let _ = plane_idx;
            pack_u32(2, &mut out); // H
            pack_u32(2, &mut out); // W
            pack_f32_slice(&[0.1, 0.2, 0.3, 0.4, 1.0, 1.1, 1.2, 1.3], &mut out);
        }

        // feature_out MLP: 1 layer in=2 out=3
        pack_u32(1, &mut out);
        pack_u32(2, &mut out);
        pack_u32(3, &mut out);
        pack_u32(0, &mut out); // no relu before
        pack_f32_slice(
            &[
                1.0, 0.0, // row0
                0.0, 1.0, // row1
                1.0, 1.0, // row2
            ],
            &mut out,
        );
        pack_f32_slice(&[0.0, 0.0, 0.0], &mut out);

        // pos_deform MLP: 2 layers
        pack_u32(2, &mut out);
        // layer 0: in=3 out=3
        pack_u32(3, &mut out);
        pack_u32(3, &mut out);
        pack_u32(0, &mut out);
        pack_f32_slice(
            &[
                1.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, //
                0.0, 0.0, 1.0, //
            ],
            &mut out,
        );
        pack_f32_slice(&[0.0, 0.0, 0.0], &mut out);
        // layer 1: in=3 out=3, relu before
        pack_u32(3, &mut out);
        pack_u32(3, &mut out);
        pack_u32(1, &mut out);
        pack_f32_slice(
            &[
                0.1, 0.0, 0.0, //
                0.0, 0.1, 0.0, //
                0.0, 0.0, 0.1, //
            ],
            &mut out,
        );
        pack_f32_slice(&[0.01, -0.02, 0.03], &mut out);

        // rotations_deform MLP: 2 layers
        pack_u32(2, &mut out);
        // layer 0: in=3 out=3
        pack_u32(3, &mut out);
        pack_u32(3, &mut out);
        pack_u32(0, &mut out);
        pack_f32_slice(
            &[
                0.5, 0.0, 0.0, //
                0.0, 0.5, 0.0, //
                0.0, 0.0, 0.5, //
            ],
            &mut out,
        );
        pack_f32_slice(&[0.0, 0.0, 0.0], &mut out);
        // layer 1: in=3 out=4, relu before
        pack_u32(3, &mut out);
        pack_u32(4, &mut out);
        pack_u32(1, &mut out);
        pack_f32_slice(
            &[
                0.2, 0.0, 0.0, //
                0.0, 0.2, 0.0, //
                0.0, 0.0, 0.2, //
                0.1, 0.1, 0.1, //
            ],
            &mut out,
        );
        pack_f32_slice(&[0.0, 0.0, 0.0, 0.0], &mut out);

        out
    }

    #[test]
    fn parser_rejects_bad_magic() {
        let mut bin = build_minimal_valid_bin();
        bin[0] = b'X';
        let err = DeformationNetwork::from_bytes(&bin).unwrap_err();
        assert!(matches!(err, DeformationError::BadMagic(_)));
    }

    #[test]
    fn parser_rejects_bad_version() {
        let mut bin = build_minimal_valid_bin();
        // version offset: 4..8
        bin[4..8].copy_from_slice(&2_u32.to_le_bytes());
        let err = DeformationNetwork::from_bytes(&bin).unwrap_err();
        assert!(matches!(err, DeformationError::UnsupportedVersion(2)));
    }

    #[test]
    fn parser_rejects_truncated_payload() {
        let mut bin = build_minimal_valid_bin();
        bin.truncate(bin.len() - 7);
        let err = DeformationNetwork::from_bytes(&bin).unwrap_err();
        assert!(matches!(err, DeformationError::UnexpectedEof { .. }));
    }

    #[test]
    fn parser_accepts_minimal_valid_payload() {
        let bin = build_minimal_valid_bin();
        let net = DeformationNetwork::from_bytes(&bin).unwrap();
        let meta = net.metadata();
        assert_eq!(meta.n_grid_levels, 1);
        assert_eq!(meta.n_planes_per_level, 6);
        assert_eq!(meta.feature_dim, 2);
        assert_eq!(meta.net_width, 3);
        assert_eq!(meta.n_time_frames, 4);
    }

    #[test]
    fn bilinear_sampling_matches_expected_center_average() {
        let plane = GridPlane {
            channels: 1,
            height: 2,
            width: 2,
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        let sampled = plane.sample_bilinear_border(0.0, 0.0);
        assert!((sampled[0] - 2.5).abs() < 1.0e-6);
    }

    #[test]
    fn quaternion_normalization_is_stable() {
        let q = normalize_quaternion([2.0, 0.0, 0.0, 0.0]);
        assert!((q[0] - 1.0).abs() < 1.0e-6);
        assert!(q[1].abs() < 1.0e-6);
        assert!(q[2].abs() < 1.0e-6);
        assert!(q[3].abs() < 1.0e-6);

        let zero = normalize_quaternion([0.0, 0.0, 0.0, 0.0]);
        assert_eq!(zero, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn deform_batch_runs_and_returns_shapes() {
        let bin = build_minimal_valid_bin();
        let net = DeformationNetwork::from_bytes(&bin).unwrap();

        let orig_means = vec![[0.2, -0.1, 0.3], [0.0, 0.0, 0.0]];
        let tile_means = vec![[1.0, 2.0, 3.0], [0.5, -0.5, 1.5]];
        let orig_quats = vec![[1.0, 0.0, 0.0, 0.0], [0.707, 0.0, 0.707, 0.0]];

        let (means, quats) = net
            .deform_batch(&orig_means, &tile_means, &orig_quats, 0.25)
            .unwrap();
        assert_eq!(means.len(), 2);
        assert_eq!(quats.len(), 2);
        assert!(means.iter().all(|m| m.iter().all(|v| v.is_finite())));
        assert!(quats.iter().all(|q| q.iter().all(|v| v.is_finite())));
    }

    #[test]
    fn pack_for_gpu_produces_consistent_offsets() {
        let bin = build_minimal_valid_bin();
        let net = DeformationNetwork::from_bytes(&bin).unwrap();
        let packed = net.pack_for_gpu().unwrap();

        assert_eq!(
            packed.plane_descs.len(),
            net.metadata().n_grid_levels * net.metadata().n_planes_per_level
        );
        assert!(!packed.plane_data.is_empty());
        assert_eq!(packed.feature_layers.len(), 1);
        assert_eq!(packed.pos_layers.len(), 2);
        assert_eq!(packed.rot_layers.len(), 2);
        assert_eq!(packed.feature_layers[0].weight_offset, 0);
        assert_eq!(packed.feature_layers[0].bias_offset, 0);
    }
}
