use bus::Bus;
use core::f32;
use regex::Regex;
use std::{
    cmp::Ordering,
    collections::HashMap,
    io::{BufRead, BufReader, Cursor, Read, Seek, SeekFrom},
    sync::{Arc, Mutex},
};
//use wasm_thread as thread;

use crate::catmull_rom_motion::{
    CATMULL_ROM_META_FILENAME, CatmullRomMotionSet, CatmullRomMotionZipEntry, detect_motion_file,
    load_catmull_rom_motion_from_zip,
};
use crate::log;
use crate::utils::*;

const MAX_HEADER_LINES: usize = 256;
const SH_C0: f32 = 0.28209479177387814;

#[derive(Clone, Copy, Debug)]
enum PlyScalarType {
    Char,
    UChar,
    Short,
    UShort,
    Int,
    UInt,
    Float,
    Double,
}

impl PlyScalarType {
    fn from_token(token: &str) -> Option<Self> {
        match token {
            "char" | "int8" => Some(Self::Char),
            "uchar" | "uint8" => Some(Self::UChar),
            "short" | "int16" => Some(Self::Short),
            "ushort" | "uint16" => Some(Self::UShort),
            "int" | "int32" => Some(Self::Int),
            "uint" | "uint32" => Some(Self::UInt),
            "float" | "float32" => Some(Self::Float),
            "double" | "float64" => Some(Self::Double),
            _ => None,
        }
    }

    fn byte_size(self) -> usize {
        match self {
            Self::Char | Self::UChar => 1,
            Self::Short | Self::UShort => 2,
            Self::Int | Self::UInt | Self::Float => 4,
            Self::Double => 8,
        }
    }

    fn read_f32_le(self, row: &[u8], offset: usize) -> Result<f32, String> {
        let read = |n: usize| -> Result<&[u8], String> {
            row.get(offset..offset + n).ok_or_else(|| {
                format!(
                    "Scene::load(): row underflow while reading property at offset {} (+{})",
                    offset, n
                )
            })
        };
        match self {
            Self::Char => {
                let v = i8::from_le_bytes([read(1)?[0]]);
                Ok(v as f32)
            }
            Self::UChar => Ok(read(1)?[0] as f32),
            Self::Short => {
                let mut arr = [0_u8; 2];
                arr.copy_from_slice(read(2)?);
                Ok(i16::from_le_bytes(arr) as f32)
            }
            Self::UShort => {
                let mut arr = [0_u8; 2];
                arr.copy_from_slice(read(2)?);
                Ok(u16::from_le_bytes(arr) as f32)
            }
            Self::Int => {
                let mut arr = [0_u8; 4];
                arr.copy_from_slice(read(4)?);
                Ok(i32::from_le_bytes(arr) as f32)
            }
            Self::UInt => {
                let mut arr = [0_u8; 4];
                arr.copy_from_slice(read(4)?);
                Ok(u32::from_le_bytes(arr) as f32)
            }
            Self::Float => {
                let mut arr = [0_u8; 4];
                arr.copy_from_slice(read(4)?);
                Ok(f32::from_le_bytes(arr))
            }
            Self::Double => {
                let mut arr = [0_u8; 8];
                arr.copy_from_slice(read(8)?);
                Ok(f64::from_le_bytes(arr) as f32)
            }
        }
    }
}

#[derive(Clone, Debug)]
struct PlyPropertySpec {
    name: String,
    data_type: PlyScalarType,
    byte_offset: usize,
}

#[derive(Clone, Debug)]
pub struct PlyHeader {
    file_header_size: usize,
    pub(crate) splat_count: usize,
    row_stride: usize,
    properties: Vec<PlyPropertySpec>,
}

#[derive(Clone)]
#[repr(C)]
struct SerializedSplat {
    position: [f32; 3],   // center of the Gaussian ellipsoid
    n: [f32; 3],          // unused normal
    color: [f32; 3 * 16], // RGB(3) + SH(45)
    alpha: f32,           // opacity
    scale: [f32; 3],      // scale of the Gaussian
    rotation: [f32; 4],   // quaternion
} // 62*f32 (62*4=248bytes) in total
impl Default for SerializedSplat {
    fn default() -> Self {
        unsafe { std::mem::MaybeUninit::<SerializedSplat>::zeroed().assume_init() }
    }
}

#[derive(Clone)]
#[repr(C)]
pub struct SerializedSplat2 {
    // Scaniverse PLY format (no normals) / SPZ format
    pub position: [f32; 3],
    pub scale: [f32; 3],
    pub rotation: [f32; 4],
    pub alpha: f32,
    pub color: [f32; 3 * 16],
} // 59*f32 (59*4=236bytes) in total
impl Default for SerializedSplat2 {
    fn default() -> Self {
        unsafe { std::mem::MaybeUninit::<SerializedSplat2>::zeroed().assume_init() }
    }
}

/// A point cloud of Gaussian splats
pub struct Scene {
    pub splat_count: usize,
    pub(crate) buffer: Vec<u8>,
    pub(crate) tex_data: Vec<u32>,
    pub(crate) tex_width: usize,
    pub(crate) tex_height: usize,
    pub(crate) orig_means: Option<Vec<[f32; 3]>>,
    pub(crate) orig_quats: Option<Vec<[f32; 4]>>,
    /// Maps renderer-local sorted splat indices back to the source file row.
    pub(crate) source_row_indices: Vec<u32>,
    prev_vp: Mutex<Vec<f32>>,
}
impl Scene {
    pub fn new() -> Self {
        Self {
            splat_count: 0,
            buffer: Vec::<u8>::new(),
            tex_data: Vec::<u32>::new(),
            tex_width: 0,
            tex_height: 0,
            orig_means: None,
            orig_quats: None,
            source_row_indices: Vec::new(),
            prev_vp: Mutex::new(Vec::<f32>::new()),
        }
    }

    /// Parses the header of a PLY file
    /// Returns the header length in bytes, the number of splats in the file, and the file cursor
    pub fn parse_file_header(bytes: Vec<u8>) -> Result<(PlyHeader, Cursor<Vec<u8>>), String> {
        let mut reader = BufReader::new(Cursor::new(bytes));
        let mut line = String::new();
        let mut splat_count: usize = 0;
        let mut row_stride: usize = 0;
        let mut properties: Vec<PlyPropertySpec> = Vec::new();
        let mut current_element = String::new();
        let mut format_is_binary_le = false;
        let mut success = false;
        let mut i = 0;

        loop {
            reader.read_line(&mut line).unwrap();
            if line == "end_header\n" || line == "end_header\r\n" {
                success = true;
                break;
            }

            let line_trimmed = line.trim_end();
            if line_trimmed.starts_with("format ") {
                let tokens: Vec<&str> = line_trimmed.split_ascii_whitespace().collect();
                if tokens.len() >= 2 && tokens[1] == "binary_little_endian" {
                    format_is_binary_le = true;
                }
            } else if line_trimmed.starts_with("element ") {
                let tokens: Vec<&str> = line_trimmed.split_ascii_whitespace().collect();
                if tokens.len() == 3 {
                    current_element = tokens[1].to_string();
                    if current_element == "vertex" {
                        splat_count = tokens[2].parse().map_err(|_| {
                            format!(
                                "Scene::parse_file_header(): invalid vertex count '{}'",
                                tokens[2]
                            )
                        })?;
                    }
                }
            } else if line_trimmed.starts_with("property ") && current_element == "vertex" {
                let tokens: Vec<&str> = line_trimmed.split_ascii_whitespace().collect();
                if tokens.len() != 3 || tokens[1] == "list" {
                    return Err(format!(
                        "Scene::parse_file_header(): unsupported vertex property syntax '{}'",
                        line_trimmed
                    ));
                }
                let scalar_type = PlyScalarType::from_token(tokens[1]).ok_or_else(|| {
                    format!(
                        "Scene::parse_file_header(): unsupported PLY scalar type '{}'",
                        tokens[1]
                    )
                })?;
                properties.push(PlyPropertySpec {
                    name: tokens[2].to_string(),
                    data_type: scalar_type,
                    byte_offset: row_stride,
                });
                row_stride += scalar_type.byte_size();
            }
            line.clear();

            i += 1;
            if i > MAX_HEADER_LINES {
                break;
            }
        }

        if !success {
            let error = "Scene::parse_file_header(): ERROR: the file is not correctly formatted.";
            log!("{}, i={}", error, i);
            return Err(error.to_string());
        }

        if !format_is_binary_le {
            return Err(
                "Scene::parse_file_header(): only binary_little_endian PLY is supported."
                    .to_string(),
            );
        }
        if properties.is_empty() || row_stride == 0 {
            return Err(
                "Scene::parse_file_header(): vertex element has no scalar properties.".to_string(),
            );
        }

        let file_header_size = reader.stream_position().unwrap() as usize;
        let cursor = reader.into_inner();
        log!(
            "Scene::parse_file_header(): i={}, file_header_size={}, splat_count={}, row_stride={}",
            i,
            file_header_size,
            splat_count,
            row_stride
        );

        Ok((
            PlyHeader {
                file_header_size,
                splat_count,
                row_stride,
                properties,
            },
            cursor,
        ))
    }

    /// Loads an entire PLY file into WASM memory
    pub fn load(
        &mut self,
        cursor: &mut Cursor<Vec<u8>>,
        ply_header: &PlyHeader,
    ) -> Result<(), String> {
        self.orig_means = None;
        self.orig_quats = None;
        self.source_row_indices.clear();
        self.splat_count = ply_header.splat_count;

        let properties = &ply_header.properties;
        let mut property_map: HashMap<&str, usize> = HashMap::new();
        for (idx, prop) in properties.iter().enumerate() {
            property_map.insert(prop.name.as_str(), idx);
        }
        let get_prop_idx = |name: &str| -> Result<usize, String> {
            property_map
                .get(name)
                .copied()
                .ok_or_else(|| format!("Scene::load(): required PLY property '{}' not found", name))
        };
        let read_prop = |row: &[u8], prop_idx: usize| -> Result<f32, String> {
            let prop = &properties[prop_idx];
            prop.data_type.read_f32_le(row, prop.byte_offset)
        };

        let x_idx = get_prop_idx("x")?;
        let y_idx = get_prop_idx("y")?;
        let z_idx = get_prop_idx("z")?;
        let opacity_idx = get_prop_idx("opacity")?;
        let f_dc_0_idx = get_prop_idx("f_dc_0")?;
        let f_dc_1_idx = get_prop_idx("f_dc_1")?;
        let f_dc_2_idx = get_prop_idx("f_dc_2")?;
        let scale_0_idx = get_prop_idx("scale_0")?;
        let scale_1_idx = get_prop_idx("scale_1")?;
        let scale_2_idx = get_prop_idx("scale_2")?;
        let rot_0_idx = get_prop_idx("rot_0")?;
        let rot_1_idx = get_prop_idx("rot_1")?;
        let rot_2_idx = get_prop_idx("rot_2")?;
        let rot_3_idx = get_prop_idx("rot_3")?;

        let mut f_rest_props: Vec<(usize, usize)> = properties
            .iter()
            .enumerate()
            .filter_map(|(idx, p)| {
                p.name
                    .strip_prefix("f_rest_")
                    .and_then(|s| s.parse::<usize>().ok().map(|order| (order, idx)))
            })
            .collect();
        f_rest_props.sort_by_key(|e| e.0);
        if f_rest_props.is_empty() {
            return Err(
                "Scene::load(): required PLY property pattern 'f_rest_*' not found".to_string(),
            );
        }

        let has_orig_prefix = property_map.contains_key("orig_x")
            || property_map.contains_key("orig_y")
            || property_map.contains_key("orig_z");
        let has_ox_prefix = property_map.contains_key("ox")
            || property_map.contains_key("oy")
            || property_map.contains_key("oz");
        let orig_xyz_idx = if property_map.contains_key("orig_x")
            && property_map.contains_key("orig_y")
            && property_map.contains_key("orig_z")
        {
            Some((
                get_prop_idx("orig_x")?,
                get_prop_idx("orig_y")?,
                get_prop_idx("orig_z")?,
            ))
        } else if property_map.contains_key("ox")
            && property_map.contains_key("oy")
            && property_map.contains_key("oz")
        {
            Some((
                get_prop_idx("ox")?,
                get_prop_idx("oy")?,
                get_prop_idx("oz")?,
            ))
        } else if has_orig_prefix || has_ox_prefix {
            return Err(
                "Scene::load(): found partial orig position properties; expected orig_x/y/z or ox/oy/oz."
                    .to_string(),
            );
        } else {
            None
        };

        cursor
            .seek(SeekFrom::Start(ply_header.file_header_size as u64))
            .map_err(|e| format!("Scene::load(): seek failed: {}", e))?;
        let payload_size = ply_header
            .splat_count
            .checked_mul(ply_header.row_stride)
            .ok_or_else(|| "Scene::load(): payload size overflow".to_string())?;
        let mut payload = vec![0_u8; payload_size];
        cursor
            .read_exact(payload.as_mut_slice())
            .map_err(|e| format!("Scene::load(): failed to read payload: {}", e))?;

        let mut size_list = vec![0_f32; self.splat_count];
        let mut size_index = vec![0_u32; self.splat_count];

        let mut positions = vec![[0_f32; 3]; self.splat_count];
        let mut scales_log = vec![[0_f32; 3]; self.splat_count];
        let mut rot_raw = vec![[0_f32; 4]; self.splat_count];
        let mut f_dc = vec![[0_f32; 3]; self.splat_count];
        let mut opacities = vec![0_f32; self.splat_count];
        let mut orig_means_raw = orig_xyz_idx.map(|_| vec![[0_f32; 3]; self.splat_count]);

        for i in 0..self.splat_count {
            let row_start = i * ply_header.row_stride;
            let row_end = row_start + ply_header.row_stride;
            let row = &payload[row_start..row_end];
            size_index[i] = i as u32;

            positions[i] = [
                read_prop(row, x_idx)?,
                read_prop(row, y_idx)?,
                read_prop(row, z_idx)?,
            ];
            scales_log[i] = [
                read_prop(row, scale_0_idx)?,
                read_prop(row, scale_1_idx)?,
                read_prop(row, scale_2_idx)?,
            ];
            rot_raw[i] = [
                read_prop(row, rot_0_idx)?,
                read_prop(row, rot_1_idx)?,
                read_prop(row, rot_2_idx)?,
                read_prop(row, rot_3_idx)?,
            ];
            f_dc[i] = [
                read_prop(row, f_dc_0_idx)?,
                read_prop(row, f_dc_1_idx)?,
                read_prop(row, f_dc_2_idx)?,
            ];
            let opacity = read_prop(row, opacity_idx)?;
            opacities[i] = opacity;
            let scale = scales_log[i][0].exp() * scales_log[i][1].exp() * scales_log[i][2].exp();
            size_list[i] = scale * (1.0 / (1.0 + (-opacity).exp()));

            if let (Some((ox_idx, oy_idx, oz_idx)), Some(orig_means)) =
                (orig_xyz_idx, orig_means_raw.as_mut())
            {
                orig_means[i] = [
                    read_prop(row, ox_idx)?,
                    read_prop(row, oy_idx)?,
                    read_prop(row, oz_idx)?,
                ];
            }
        }

        size_index.sort_by(|&a, &b| {
            size_list[b as usize]
                .partial_cmp(&size_list[a as usize])
                .unwrap_or(Ordering::Equal)
        });
        if !size_index.is_empty() {
            log!(
                "Scene::load(): size_list[0]={}, size_list[-1]={}",
                size_list[size_index[0] as usize],
                size_list[size_index[size_index.len() - 1] as usize]
            );
        }

        // XYZ - position (f32)
        // XYZ - scale (f32, exp)
        // RGBA - color (u8)
        // IJKL - quaternion (u8, normalized+quantized)
        let row_length = 3 * 4 + 3 * 4 + 4 + 4;
        let mut buffer = vec![0_u8; row_length * self.splat_count];
        let mut sorted_orig_quats = Vec::with_capacity(self.splat_count);
        let mut sorted_orig_means = orig_means_raw
            .as_ref()
            .map(|_| Vec::with_capacity(self.splat_count));
        let mut source_row_indices = Vec::with_capacity(self.splat_count);
        for i in 0..self.splat_count {
            let row = size_index[i] as usize;
            source_row_indices.push(row as u32);

            let mut start = i * row_length;
            let mut end = start + 3 * 4;
            {
                let position: &mut [f32] = transmute_slice_mut::<_, f32>(&mut buffer[start..end]);
                position[0] = positions[row][0];
                position[1] = positions[row][1];
                position[2] = positions[row][2];
            }

            start = end;
            end = start + 3 * 4;
            {
                let scales: &mut [f32] = transmute_slice_mut::<_, f32>(&mut buffer[start..end]);
                scales[0] = scales_log[row][0].exp();
                scales[1] = scales_log[row][1].exp();
                scales[2] = scales_log[row][2].exp();
            }

            start = end;
            end = start + 4;
            {
                let rgba: &mut [u8] = transmute_slice_mut::<_, u8>(&mut buffer[start..end]);
                rgba[0] = ((0.5 + SH_C0 * f_dc[row][0]) * 255.0) as u8;
                rgba[1] = ((0.5 + SH_C0 * f_dc[row][1]) * 255.0) as u8;
                rgba[2] = ((0.5 + SH_C0 * f_dc[row][2]) * 255.0) as u8;
                rgba[3] = ((1.0 / (1.0 + (-opacities[row]).exp())) * 255.0) as u8;
            }

            start = end;
            end = start + 4;
            {
                let rot: &mut [u8] = transmute_slice_mut::<_, u8>(&mut buffer[start..end]);
                let q = rot_raw[row];
                let qlen = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
                    .sqrt()
                    .max(1.0e-8);
                rot[0] = (((q[0] / qlen) + 1.0) * 0.5 * 255.0) as u8;
                rot[1] = (((q[1] / qlen) + 1.0) * 0.5 * 255.0) as u8;
                rot[2] = (((q[2] / qlen) + 1.0) * 0.5 * 255.0) as u8;
                rot[3] = (((q[3] / qlen) + 1.0) * 0.5 * 255.0) as u8;
                sorted_orig_quats.push(q);
            }

            if let (Some(orig_raw), Some(orig_sorted)) =
                (orig_means_raw.as_ref(), sorted_orig_means.as_mut())
            {
                orig_sorted.push(orig_raw[row]);
            }
        }

        self.buffer = buffer;
        self.orig_quats = Some(sorted_orig_quats);
        self.orig_means = sorted_orig_means;
        self.source_row_indices = source_row_indices;
        Ok(())
    }

    /// Loads an entire PLY file (w/o normals) into WASM memory
    pub fn load_no_normal(&mut self, serialized_splats: Vec<SerializedSplat2>) {
        self.orig_means = None;
        self.orig_quats = None;
        self.source_row_indices.clear();
        // TODO: remove code redundancy w/ load()
        // calculate importance of each splat
        let mut size_list = vec![0_f32; self.splat_count];
        let mut size_index = vec![0_u32; self.splat_count];
        for i in 0..self.splat_count {
            let s = &serialized_splats[i];
            size_index[i] = i as u32;
            let size = s.scale[0].exp() * s.scale[1].exp() * s.scale[2].exp();
            let opacity = 1.0 / (1.0 + (-s.alpha).exp());
            size_list[i] = (size as f32) * opacity;
        }

        // sort the indices of splats based on size_list in descending order
        size_index.sort_by(|&a, &b| {
            size_list[b as usize]
                .partial_cmp(&size_list[a as usize])
                .unwrap_or(Ordering::Equal)
        });
        log!(
            "Scene::load_no_normal(): size_list[0]={}, size_list[-1]={}",
            size_list[size_index[0] as usize],
            size_list[size_index[size_index.len() - 1] as usize]
        );

        // construct a new binary buffer where each row corresponds to a splat in the sorted order.
        // XYZ - position (f32)
        // XYZ - scale (f32)
        // RGBA - color (u8)
        // IJKL - quaternion (u8)
        let row_length = 3 * 4 + 3 * 4 + 4 + 4; // 32bytes
        let mut buffer = vec![0_u8; row_length * self.splat_count];
        let mut sorted_orig_quats = Vec::with_capacity(self.splat_count);
        let mut source_row_indices = Vec::with_capacity(self.splat_count);
        for i in 0..self.splat_count {
            let row = size_index[i] as usize;
            let s = &serialized_splats[row];
            source_row_indices.push(row as u32);

            let mut start = i * row_length;
            let mut end = start + 3 * 4;
            {
                // read 3x f32
                let position: &mut [f32] = transmute_slice_mut::<_, f32>(&mut buffer[start..end]);
                position[0] = s.position[0];
                position[1] = s.position[1];
                position[2] = s.position[2];
            }

            start = end;
            end = start + 3 * 4;
            {
                // read 3x f32
                let scales: &mut [f32] = transmute_slice_mut::<_, f32>(&mut buffer[start..end]);
                scales[0] = s.scale[0].exp();
                scales[1] = s.scale[1].exp();
                scales[2] = s.scale[2].exp();
            }

            // In Rust, float-to-integer casts saturate
            // (i.e., excess values are converted to T::MAX or T::MIN. NaN is converted to 0).

            start = end;
            end = start + 4;
            {
                // read 4x u8
                let rgba: &mut [u8] = transmute_slice_mut::<_, u8>(&mut buffer[start..end]);
                rgba[0] = ((0.5 + SH_C0 * s.color[0]) * 255.0) as u8;
                rgba[1] = ((0.5 + SH_C0 * s.color[1]) * 255.0) as u8;
                rgba[2] = ((0.5 + SH_C0 * s.color[2]) * 255.0) as u8;
                rgba[3] = ((1.0 / (1.0 + (-s.alpha).exp())) * 255.0) as u8; // opacity from sigmoid
            }

            start = end;
            end = start + 4;
            {
                // read 4x u8
                let rot: &mut [u8] = transmute_slice_mut::<_, u8>(&mut buffer[start..end]);
                let qlen = (s.rotation[0].powi(2)
                    + s.rotation[1].powi(2)
                    + s.rotation[2].powi(2)
                    + s.rotation[3].powi(2))
                .sqrt();
                // [-1, 1] -> [0, 255]
                rot[0] = (((s.rotation[0] / qlen) + 1.0) * 0.5 * 255.0) as u8;
                rot[1] = (((s.rotation[1] / qlen) + 1.0) * 0.5 * 255.0) as u8;
                rot[2] = (((s.rotation[2] / qlen) + 1.0) * 0.5 * 255.0) as u8;
                rot[3] = (((s.rotation[3] / qlen) + 1.0) * 0.5 * 255.0) as u8;
                sorted_orig_quats.push(s.rotation);
            }
        }
        self.buffer = buffer;
        self.orig_quats = Some(sorted_orig_quats);
        self.source_row_indices = source_row_indices;
    }

    /// Generates a 2D texture from the splats
    pub fn generate_texture(&mut self) {
        // TODO: parallelize
        if self.buffer.is_empty() {
            return;
        }
        let f_buffer: &[f32] = transmute_slice::<_, f32>(self.buffer.as_slice());
        let u_buffer: &[u8] = transmute_slice::<_, u8>(self.buffer.as_slice());

        let texwidth = 1024 * 2 as usize;
        let texheight = ((2 * self.splat_count) as f64 / texwidth as f64).ceil() as usize;
        let len_texdata = texwidth * texheight * 4 as usize; // 4 components per pixel (RGBA)
        log!(
            "Scene::generate_texture(): texheight={}, len_texdata={}",
            texheight,
            len_texdata
        );
        let mut texdata = vec![0_u32; len_texdata];
        // texdata structure: 32B (2 pixels) per gs, 1024 gs (2048 pixels) per row
        // |                          32B                          |
        // |  4B  |  4B  |  4B  |  4B  |  4B  |  4B  |  4B  |  4B  |
        // | posx | posy | posz | none |  ab  |  cd  |  ef  | rgba |

        {
            let texdata_f = transmute_slice_mut::<_, f32>(texdata.as_mut_slice());
            for i in 0..self.splat_count {
                // x, y, z components of the i-th splat in f_buffer
                let index_f: usize = 8 * i;
                texdata_f[index_f + 0] = f_buffer[index_f + 0];
                texdata_f[index_f + 1] = f_buffer[index_f + 1];
                texdata_f[index_f + 2] = f_buffer[index_f + 2];
            }
        }

        {
            let texdata_c = transmute_slice_mut::<_, u8>(texdata.as_mut_slice());
            for i in 0..self.splat_count {
                // r, g, b, a components of the i-th splat in u_buffer
                let index_c: usize = 4 * (8 * i + 7);
                let index_u: usize = 32 * i + 3 * 4 + 3 * 4;
                texdata_c[index_c + 0] = u_buffer[index_u + 0];
                texdata_c[index_c + 1] = u_buffer[index_u + 1];
                texdata_c[index_c + 2] = u_buffer[index_u + 2];
                texdata_c[index_c + 3] = u_buffer[index_u + 3];
            }
        }

        for i in 0..self.splat_count {
            let index_f: usize = 8 * i;
            let scale = [
                f_buffer[index_f + 3],
                f_buffer[index_f + 4],
                f_buffer[index_f + 5],
            ];

            let index_u: usize = 32 * i + 3 * 4 + 3 * 4 + 4;
            let rot = [
                // [0, 255] -> [-1, 1]
                ((u_buffer[index_u + 0] as f32) / 255.0) * 2.0 - 1.0, // qw
                ((u_buffer[index_u + 1] as f32) / 255.0) * 2.0 - 1.0, // qx
                ((u_buffer[index_u + 2] as f32) / 255.0) * 2.0 - 1.0, // qy
                ((u_buffer[index_u + 3] as f32) / 255.0) * 2.0 - 1.0, // qz
            ];

            let r = Mat3::new(
                // column-major
                1.0 - 2.0 * (rot[2] * rot[2] + rot[3] * rot[3]),
                2.0 * (rot[1] * rot[2] + rot[0] * rot[3]),
                2.0 * (rot[1] * rot[3] - rot[0] * rot[2]),
                2.0 * (rot[1] * rot[2] - rot[0] * rot[3]),
                1.0 - 2.0 * (rot[1] * rot[1] + rot[3] * rot[3]),
                2.0 * (rot[2] * rot[3] + rot[0] * rot[1]),
                2.0 * (rot[1] * rot[3] + rot[0] * rot[2]),
                2.0 * (rot[2] * rot[3] - rot[0] * rot[1]),
                1.0 - 2.0 * (rot[1] * rot[1] + rot[2] * rot[2]),
            );

            let s = Mat3::new(scale[0], 0.0, 0.0, 0.0, scale[1], 0.0, 0.0, 0.0, scale[2]);

            let m = r * s;
            let m = &[
                // column-major: [col][row]
                m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2],
            ];

            // M * M^T = R * S * S^T * R^T
            let sigma = [
                m[0] * m[0] + m[3] * m[3] + m[6] * m[6],
                m[0] * m[1] + m[3] * m[4] + m[6] * m[7],
                m[0] * m[2] + m[3] * m[5] + m[6] * m[8],
                m[1] * m[1] + m[4] * m[4] + m[7] * m[7],
                m[1] * m[2] + m[4] * m[5] + m[7] * m[8],
                m[2] * m[2] + m[5] * m[5] + m[8] * m[8],
            ];

            // JavaScript typically uses the host system's endianness
            // (x86-64 and Apple CPUs are little-endian).
            // WASM's linear memory is always little-endian.
            texdata[index_f + 4] = pack_half_2x16(4.0 * sigma[0], 4.0 * sigma[1]); // a, b
            texdata[index_f + 5] = pack_half_2x16(4.0 * sigma[2], 4.0 * sigma[3]); // c, d
            texdata[index_f + 6] = pack_half_2x16(4.0 * sigma[4], 4.0 * sigma[5]); // e, f
        }

        self.tex_data = texdata;
        self.tex_width = texwidth;
        self.tex_height = texheight;
    }

    /// Sorts the splats based on their depth using 16-bit single-pass counting sort
    pub fn sort(scene: &Arc<Self>, view_proj: &[f32], bus: &mut Bus<Vec<u32>>, n_threads: usize) {
        if scene.buffer.is_empty() {
            return;
        }
        let f_buffer: &[f32] = transmute_slice::<_, f32>(scene.buffer.as_slice());

        {
            let mut mutex = scene.prev_vp.lock().unwrap();
            if (*mutex).is_empty() {
                (*mutex).push(view_proj[2]);
                (*mutex).push(view_proj[6]);
                (*mutex).push(view_proj[10]);
            } else {
                let dot = (*mutex)[0] * view_proj[2]
                    + (*mutex)[1] * view_proj[6]
                    + (*mutex)[2] * view_proj[10];
                if (dot - 1.0).abs() < 0.01 {
                    return;
                }
            }
        }

        // calculates the depth for each splat based on the view projection matrix
        // and updates sizeList with the calculated depths.
        let mut max_depth = i32::MIN;
        let mut min_depth = i32::MAX;
        /*
        let mut size_list = vec![0_i32; scene.splat_count];
        for i in 0..scene.splat_count {
            let index_f = 8*i as usize;
            let depth = (
                (
                    view_proj[2] * f_buffer[index_f + 0] +
                    view_proj[6] * f_buffer[index_f + 1] +
                    view_proj[10] * f_buffer[index_f + 2]
                ) * 4096.0
            ) as i32;
            size_list[i] = depth;
            if depth > max_depth { max_depth = depth; }
            if depth < min_depth { min_depth = depth; }
        }
        */
        let size_list: Vec<i32> = (0..scene.splat_count)
            .map(|i| {
                let index_f = 8 * i as usize;
                let depth = ((view_proj[2] * f_buffer[index_f + 0]
                    + view_proj[6] * f_buffer[index_f + 1]
                    + view_proj[10] * f_buffer[index_f + 2])
                    * 4096.0) as i32;
                if depth > max_depth {
                    max_depth = depth;
                }
                if depth < min_depth {
                    min_depth = depth;
                }
                depth
            })
            .collect();
        let mut size_list = size_list;
        //log!("Scene::sort(): max_depth={:?}, min_depth={:?}", max_depth, min_depth);

        let size16: usize = 256 * 256; // 65,536
        let depth_inv = (size16 - 1) as f32 / (max_depth - min_depth) as f32;

        let mut counts0 = vec![0_u32; size16];
        // count the occurrences of each depth
        for i in 0..scene.splat_count {
            let depth = ((size_list[i] - min_depth) as f32 * depth_inv).floor() as i32;
            let depth = depth.clamp(0, size16 as i32 - 1);
            size_list[i] = depth;
            counts0[depth as usize] += 1;
        }
        let mut starts0 = vec![0_u32; size16];
        // store the cumulative count of elements
        for i in 1..size16 {
            starts0[i] = starts0[i - 1] + counts0[i - 1];
        }

        let mut depth_index = vec![0_u32; scene.splat_count];
        for i in 0..scene.splat_count {
            let depth = size_list[i] as usize;
            let j = starts0[depth] as usize;
            depth_index[j] = i as u32;
            starts0[depth] += 1;
        }
        depth_index.reverse(); // FIXME

        //////////////////////////////////
        // no cloning is happening for the single-consumer case
        let _ = bus.try_broadcast(depth_index);
        //////////////////////////////////

        {
            let mut mutex = scene.prev_vp.lock().unwrap();
            (*mutex)[0] = view_proj[2];
            (*mutex)[1] = view_proj[6];
            (*mutex)[2] = view_proj[10];
        }
    }

    pub fn sort_self(&self, view_proj: &[f32]) -> (Vec<u32>, Vec<i32>) {
        let f_buffer: &[f32] = transmute_slice::<_, f32>(self.buffer.as_slice());

        // calculates the depth for each splat based on the view projection matrix
        // and updates sizeList with the calculated depths.
        let mut max_depth = i32::MIN;
        let mut min_depth = i32::MAX;
        /*
        let mut size_list = vec![0_i32; self.splat_count];
        for i in 0..self.splat_count {
            let index_f = 8*i as usize;
            let depth = (
                (
                    view_proj[2] * f_buffer[index_f + 0] +
                    view_proj[6] * f_buffer[index_f + 1] +
                    view_proj[10] * f_buffer[index_f + 2]
                ) * 4096.0
            ) as i32;
            size_list[i] = depth;
            if depth > max_depth { max_depth = depth; }
            if depth < min_depth { min_depth = depth; }
        }
        */
        let size_list: Vec<i32> = (0..self.splat_count)
            .map(|i| {
                let index_f = 8 * i as usize;
                let depth = ((view_proj[2] * f_buffer[index_f + 0]
                    + view_proj[6] * f_buffer[index_f + 1]
                    + view_proj[10] * f_buffer[index_f + 2])
                    * 4096.0) as i32;
                if depth > max_depth {
                    max_depth = depth;
                }
                if depth < min_depth {
                    min_depth = depth;
                }
                depth
            })
            .collect();
        let raw_depth = size_list.clone();
        let mut size_list = size_list;
        //log!("Scene::sort(): max_depth={:?}, min_depth={:?}", max_depth, min_depth);

        let size16: usize = 256 * 256; // 65,536
        let depth_inv = (size16 - 1) as f32 / (max_depth - min_depth) as f32;

        let mut counts0 = vec![0_u32; size16];
        // count the occurrences of each depth
        for i in 0..self.splat_count {
            let depth = ((size_list[i] - min_depth) as f32 * depth_inv).floor() as i32;
            let depth = depth.clamp(0, size16 as i32 - 1);
            size_list[i] = depth;
            counts0[depth as usize] += 1;
        }
        let mut starts0 = vec![0_u32; size16];
        // store the cumulative count of elements
        for i in 1..size16 {
            starts0[i] = starts0[i - 1] + counts0[i - 1];
        }

        let mut depth_index = vec![0_u32; self.splat_count];
        for i in 0..self.splat_count {
            let depth = size_list[i] as usize;
            let j = starts0[depth] as usize;
            depth_index[j] = i as u32;
            starts0[depth] += 1;
        }
        depth_index.reverse(); // FIXME

        (depth_index, raw_depth)
    }

    pub fn sort_merged(
        view_proj_z: Vec3,
        scene_vec: Vec<&Self>,
        scene_offset: Vec<Vec3>,
    ) -> Vec<(usize, usize)> {
        // calculates the depth for each splat based on the view projection matrix
        // and updates sizeList with the calculated depths.
        let mut max_depth = i32::MIN;
        let mut min_depth = i32::MAX;
        let mut full_splat_count: usize = 0;
        let mut size_list: Vec<i32> = Vec::new();
        let mut splat_displ: Vec<usize> = vec![0];
        for scene_id in 0..scene_vec.len() {
            let f_buffer: &[f32] = transmute_slice::<_, f32>(scene_vec[scene_id].buffer.as_slice());
            let mut local_size_list: Vec<i32> = (0..scene_vec[scene_id].splat_count)
                .map(|i| {
                    let index_f = 8 * i as usize;
                    let depth = ((view_proj_z.x
                        * (f_buffer[index_f + 0] + scene_offset[scene_id].x)
                        + view_proj_z.y * (f_buffer[index_f + 1] + scene_offset[scene_id].y)
                        + view_proj_z.z * (f_buffer[index_f + 2] + scene_offset[scene_id].z))
                        * 4096.0) as i32;
                    if depth > max_depth {
                        max_depth = depth;
                    }
                    if depth < min_depth {
                        min_depth = depth;
                    }
                    depth
                })
                .collect();
            size_list.append(&mut local_size_list);
            full_splat_count += scene_vec[scene_id].splat_count;
            splat_displ.push(full_splat_count);
        }
        //log!("Scene::sort(): max_depth={:?}, min_depth={:?}", max_depth, min_depth);

        let size16: usize = 256 * 256; // 65,536
        let depth_inv = (size16 - 1) as f32 / (max_depth - min_depth) as f32;

        let mut counts0 = vec![0_u32; size16];
        // count the occurrences of each depth
        for i in 0..full_splat_count {
            let depth = ((size_list[i] - min_depth) as f32 * depth_inv).floor() as i32;
            let depth = depth.clamp(0, size16 as i32 - 1);
            size_list[i] = depth;
            counts0[depth as usize] += 1;
        }
        let mut starts0 = vec![0_u32; size16];
        // store the cumulative count of elements
        for i in 1..size16 {
            starts0[i] = starts0[i - 1] + counts0[i - 1];
        }

        let mut depth_index: Vec<(usize, usize)> = vec![(0, 0); full_splat_count];
        for scene_id in 0..scene_vec.len() {
            for i in splat_displ[scene_id]..splat_displ[scene_id + 1] {
                let depth = size_list[i] as usize;
                let j = starts0[depth] as usize;
                // depth_index[j] = (i - splat_displ[scene_id]) as u32 + scene_index_offset[scene_id];
                depth_index[j] = (scene_id, i - splat_displ[scene_id]);
                starts0[depth] += 1;
            }
        }
        depth_index.reverse(); // FIXME

        depth_index
    }

    pub fn sort_raw_depth_vec(raw_depth_vec: Vec<&Vec<i32>>) -> Vec<(usize, usize)> {
        let mut full_splat_count: usize = 0;
        let mut size_list: Vec<i32> = Vec::new();
        let mut splat_displ: Vec<usize> = vec![0];
        for scene_id in 0..raw_depth_vec.len() {
            size_list.extend(raw_depth_vec[scene_id]);
            full_splat_count += raw_depth_vec[scene_id].len();
            splat_displ.push(full_splat_count);
        }
        //log!("Scene::sort(): max_depth={:?}, min_depth={:?}", max_depth, min_depth);
        let min_depth = *size_list.iter().min().unwrap();
        let max_depth = *size_list.iter().max().unwrap();

        let size16: usize = 256 * 256; // 65,536
        let depth_inv = (size16 - 1) as f32 / (max_depth - min_depth) as f32;

        let mut counts0 = vec![0_u32; size16];
        // count the occurrences of each depth
        for i in 0..full_splat_count {
            let depth = ((size_list[i] - min_depth) as f32 * depth_inv).floor() as i32;
            let depth = depth.clamp(0, size16 as i32 - 1);
            size_list[i] = depth;
            counts0[depth as usize] += 1;
        }
        let mut starts0 = vec![0_u32; size16];
        // store the cumulative count of elements
        for i in 1..size16 {
            starts0[i] = starts0[i - 1] + counts0[i - 1];
        }

        let mut depth_index: Vec<(usize, usize)> = vec![(0, 0); full_splat_count];
        for scene_id in 0..raw_depth_vec.len() {
            for i in splat_displ[scene_id]..splat_displ[scene_id + 1] {
                let depth = size_list[i] as usize;
                let j = starts0[depth] as usize;
                // depth_index[j] = (i - splat_displ[scene_id]) as u32 + scene_index_offset[scene_id];
                depth_index[j] = (scene_id, i - splat_displ[scene_id]);
                starts0[depth] += 1;
            }
        }
        depth_index.reverse(); // FIXME

        depth_index
    }

    /// Sorts the splats based on their depth using 16-bit single-pass counting sort
    pub fn sort2(scene: &Self, view_proj: &[f32], bus: &mut Bus<Vec<u32>>, n_threads: usize) {
        if scene.buffer.is_empty() {
            return;
        }
        let f_buffer: &[f32] = transmute_slice::<_, f32>(scene.buffer.as_slice());

        {
            let mut mutex = scene.prev_vp.lock().unwrap();
            if (*mutex).is_empty() {
                (*mutex).push(view_proj[2]);
                (*mutex).push(view_proj[6]);
                (*mutex).push(view_proj[10]);
            } else {
                let dot = (*mutex)[0] * view_proj[2]
                    + (*mutex)[1] * view_proj[6]
                    + (*mutex)[2] * view_proj[10];
                if (dot - 1.0).abs() < 0.01 {
                    return;
                }
            }
        }

        // calculates the depth for each splat based on the view projection matrix
        // and updates sizeList with the calculated depths.
        let mut max_depth = i32::MIN;
        let mut min_depth = i32::MAX;
        /*
        let mut size_list = vec![0_i32; scene.splat_count];
        for i in 0..scene.splat_count {
            let index_f = 8*i as usize;
            let depth = (
                (
                    view_proj[2] * f_buffer[index_f + 0] +
                    view_proj[6] * f_buffer[index_f + 1] +
                    view_proj[10] * f_buffer[index_f + 2]
                ) * 4096.0
            ) as i32;
            size_list[i] = depth;
            if depth > max_depth { max_depth = depth; }
            if depth < min_depth { min_depth = depth; }
        }
        */
        let size_list: Vec<i32> = (0..scene.splat_count)
            .map(|i| {
                let index_f = 8 * i as usize;
                let depth = ((view_proj[2] * f_buffer[index_f + 0]
                    + view_proj[6] * f_buffer[index_f + 1]
                    + view_proj[10] * f_buffer[index_f + 2])
                    * 4096.0) as i32;
                if depth > max_depth {
                    max_depth = depth;
                }
                if depth < min_depth {
                    min_depth = depth;
                }
                depth
            })
            .collect();
        let mut size_list = size_list;
        //log!("Scene::sort(): max_depth={:?}, min_depth={:?}", max_depth, min_depth);

        let size16: usize = 256 * 256; // 65,536
        let depth_inv = (size16 - 1) as f32 / (max_depth - min_depth) as f32;

        let mut counts0 = vec![0_u32; size16];
        // count the occurrences of each depth
        for i in 0..scene.splat_count {
            let depth = ((size_list[i] - min_depth) as f32 * depth_inv).floor() as i32;
            let depth = depth.clamp(0, size16 as i32 - 1);
            size_list[i] = depth;
            counts0[depth as usize] += 1;
        }
        let mut starts0 = vec![0_u32; size16];
        // store the cumulative count of elements
        for i in 1..size16 {
            starts0[i] = starts0[i - 1] + counts0[i - 1];
        }

        let mut depth_index = vec![0_u32; scene.splat_count];
        for i in 0..scene.splat_count {
            let depth = size_list[i] as usize;
            let j = starts0[depth] as usize;
            depth_index[j] = i as u32;
            starts0[depth] += 1;
        }
        depth_index.reverse(); // FIXME

        //////////////////////////////////
        // no cloning is happening for the single-consumer case
        let _ = bus.try_broadcast(depth_index);
        //////////////////////////////////

        {
            let mut mutex = scene.prev_vp.lock().unwrap();
            (*mutex)[0] = view_proj[2];
            (*mutex)[1] = view_proj[6];
            (*mutex)[2] = view_proj[10];
        }
    }

    pub fn merge(&mut self, scene: &Scene) {
        self.buffer.extend_from_slice(scene.buffer.as_slice());
        if self.source_row_indices.len() != self.splat_count {
            self.source_row_indices = (0..self.splat_count as u32).collect();
        }
        if scene.source_row_indices.len() == scene.splat_count {
            self.source_row_indices
                .extend_from_slice(scene.source_row_indices.as_slice());
        } else {
            self.source_row_indices
                .extend((0..scene.splat_count).map(|idx| idx as u32));
        }

        match (&mut self.orig_means, &scene.orig_means) {
            (Some(dst), Some(src)) => dst.extend_from_slice(src.as_slice()),
            (None, Some(src)) if self.splat_count == 0 => {
                self.orig_means = Some(src.clone());
            }
            (Some(_), None) | (None, Some(_)) => {
                self.orig_means = None;
            }
            (None, None) => {}
        }

        match (&mut self.orig_quats, &scene.orig_quats) {
            (Some(dst), Some(src)) => dst.extend_from_slice(src.as_slice()),
            (None, Some(src)) if self.splat_count == 0 => {
                self.orig_quats = Some(src.clone());
            }
            (Some(_), None) | (None, Some(_)) => {
                self.orig_quats = None;
            }
            (None, None) => {}
        }

        self.splat_count += scene.splat_count;
    }

    pub fn translate(&mut self, offset: Vec3) {
        let row_length = 3 * 4 + 3 * 4 + 4 + 4; // 32bytes
        for i in 0..self.splat_count {
            let start = i * row_length;
            let end = start + 3 * 4;
            {
                // read 3x f32
                let position: &mut [f32] =
                    transmute_slice_mut::<_, f32>(&mut self.buffer[start..end]);
                position[0] += offset.x;
                position[1] += offset.y;
                position[2] += offset.z;
            }
        }
    }

    pub fn copy_from(&mut self, scene: &Scene) {
        self.splat_count = scene.splat_count;
        self.buffer = scene.buffer.clone();
        self.tex_data = scene.tex_data.clone();
        self.tex_width = scene.tex_width;
        self.tex_height = scene.tex_height;
        self.orig_means = scene.orig_means.clone();
        self.orig_quats = scene.orig_quats.clone();
        self.source_row_indices = scene.source_row_indices.clone();
    }

    pub fn compute_aabb_and_center(&self) -> ((Vec3, Vec3), Vec3) {
        let mut aabb: Option<(Vec3, Vec3)> = None;
        let mut avg_center = Vec3::zero();
        let row_length = 3 * 4 + 3 * 4 + 4 + 4; // 32bytes
        for i in 0..self.splat_count {
            let start = i * row_length;
            let end = start + 3 * 4;
            {
                // read 3x f32
                let position: &[f32] = transmute_slice::<_, f32>(&self.buffer[start..end]);
                let position = vec3(position[0], position[1], position[2]);
                avg_center += position;
                if let Some(aabb_ref) = aabb.as_mut() {
                    aabb_ref.0 = vec3(
                        aabb_ref.0.x.min(position.x),
                        aabb_ref.0.y.min(position.y),
                        aabb_ref.0.z.min(position.z),
                    );
                    aabb_ref.1 = vec3(
                        aabb_ref.1.x.max(position.x),
                        aabb_ref.1.y.max(position.y),
                        aabb_ref.1.z.max(position.z),
                    );
                } else {
                    aabb = Some((position, position));
                }
            }
        }
        avg_center /= self.splat_count as f32;

        (aabb.unwrap(), avg_center)
    }

    pub fn compute_scale_sum(&self) -> f32 {
        let mut scale_sum: f32 = 0.0;
        let f_buffer: &[f32] = transmute_slice::<_, f32>(self.buffer.as_slice());
        for i in 0..self.splat_count {
            scale_sum += f_buffer[8 * i + 3];
            scale_sum += f_buffer[8 * i + 4];
            scale_sum += f_buffer[8 * i + 5];
        }

        scale_sum
    }
}
impl Clone for Scene {
    fn clone(&self) -> Self {
        Self {
            splat_count: self.splat_count,
            buffer: self.buffer.clone(),
            tex_data: self.tex_data.clone(),
            tex_width: self.tex_width,
            tex_height: self.tex_height,
            orig_means: self.orig_means.clone(),
            orig_quats: self.orig_quats.clone(),
            source_row_indices: self.source_row_indices.clone(),
            prev_vp: Mutex::new(Vec::<f32>::new()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_test_ply(
        props: &[&str],
        rows: &[HashMap<&str, f32>],
        truncate_payload_bytes: usize,
    ) -> Vec<u8> {
        let mut out = Vec::<u8>::new();
        out.extend_from_slice(b"ply\n");
        out.extend_from_slice(b"format binary_little_endian 1.0\n");
        out.extend_from_slice(format!("element vertex {}\n", rows.len()).as_bytes());
        for p in props {
            out.extend_from_slice(format!("property float {}\n", p).as_bytes());
        }
        out.extend_from_slice(b"end_header\n");

        for row in rows {
            for p in props {
                let v = *row.get(p).unwrap_or(&0.0);
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        if truncate_payload_bytes > 0 && out.len() > truncate_payload_bytes {
            out.truncate(out.len() - truncate_payload_bytes);
        }
        out
    }

    #[test]
    fn ply_parser_reads_properties_by_name_and_keeps_sorted_alignment() {
        let props = vec![
            "rot_1", "x", "f_rest_0", "scale_1", "opacity", "orig_z", "rot_0", "y", "f_dc_2",
            "scale_2", "rot_3", "orig_y", "scale_0", "f_dc_0", "z", "rot_2", "f_dc_1", "orig_x",
        ];

        let row_a = HashMap::from([
            ("x", 1.0),
            ("y", 2.0),
            ("z", 3.0),
            ("f_dc_0", 0.10),
            ("f_dc_1", 0.20),
            ("f_dc_2", 0.30),
            ("f_rest_0", 0.01),
            ("opacity", 2.0),
            ("scale_0", 1.0),
            ("scale_1", 1.0),
            ("scale_2", 1.0),
            ("rot_0", 0.1),
            ("rot_1", 0.2),
            ("rot_2", 0.3),
            ("rot_3", 0.4),
            ("orig_x", 10.0),
            ("orig_y", 11.0),
            ("orig_z", 12.0),
        ]);
        let row_b = HashMap::from([
            ("x", -1.0),
            ("y", -2.0),
            ("z", -3.0),
            ("f_dc_0", 0.40),
            ("f_dc_1", 0.50),
            ("f_dc_2", 0.60),
            ("f_rest_0", 0.02),
            ("opacity", -1.0),
            ("scale_0", 0.0),
            ("scale_1", 0.0),
            ("scale_2", 0.0),
            ("rot_0", 0.5),
            ("rot_1", 0.6),
            ("rot_2", 0.7),
            ("rot_3", 0.8),
            ("orig_x", 20.0),
            ("orig_y", 21.0),
            ("orig_z", 22.0),
        ]);

        let ply = build_test_ply(props.as_slice(), &[row_a, row_b], 0);
        let (header, mut cursor) = Scene::parse_file_header(ply).unwrap();
        let mut scene = Scene::new();
        scene.load(&mut cursor, &header).unwrap();

        assert_eq!(scene.splat_count, 2);
        assert_eq!(scene.source_row_indices, vec![0, 1]);
        let f_buffer: &[f32] = transmute_slice(scene.buffer.as_slice());
        // First row should be row_a (larger size*opacity)
        assert!((f_buffer[0] - 1.0).abs() < 1e-6);
        assert!((f_buffer[1] - 2.0).abs() < 1e-6);
        assert!((f_buffer[2] - 3.0).abs() < 1e-6);

        let orig_means = scene.orig_means.as_ref().unwrap();
        assert_eq!(orig_means.len(), 2);
        assert!((orig_means[0][0] - 10.0).abs() < 1e-6);
        assert!((orig_means[0][1] - 11.0).abs() < 1e-6);
        assert!((orig_means[0][2] - 12.0).abs() < 1e-6);

        let orig_quats = scene.orig_quats.as_ref().unwrap();
        assert_eq!(orig_quats.len(), 2);
        assert!((orig_quats[0][0] - 0.1).abs() < 1e-6);
        assert!((orig_quats[0][1] - 0.2).abs() < 1e-6);
        assert!((orig_quats[0][2] - 0.3).abs() < 1e-6);
        assert!((orig_quats[0][3] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn ply_parser_records_source_rows_after_size_sorting() {
        let props = vec![
            "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "f_rest_0", "opacity", "scale_0",
            "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
        ];
        let row_small = HashMap::from([
            ("x", 0.0),
            ("y", 0.0),
            ("z", 0.0),
            ("f_dc_0", 0.0),
            ("f_dc_1", 0.0),
            ("f_dc_2", 0.0),
            ("f_rest_0", 0.0),
            ("opacity", 0.0),
            ("scale_0", 0.0),
            ("scale_1", 0.0),
            ("scale_2", 0.0),
            ("rot_0", 0.0),
            ("rot_1", 0.0),
            ("rot_2", 0.0),
            ("rot_3", 1.0),
        ]);
        let row_large = HashMap::from([
            ("x", 1.0),
            ("y", 0.0),
            ("z", 0.0),
            ("f_dc_0", 0.0),
            ("f_dc_1", 0.0),
            ("f_dc_2", 0.0),
            ("f_rest_0", 0.0),
            ("opacity", 10.0),
            ("scale_0", 1.0),
            ("scale_1", 1.0),
            ("scale_2", 1.0),
            ("rot_0", 0.0),
            ("rot_1", 0.0),
            ("rot_2", 0.0),
            ("rot_3", 1.0),
        ]);
        let row_mid = HashMap::from([
            ("x", 2.0),
            ("y", 0.0),
            ("z", 0.0),
            ("f_dc_0", 0.0),
            ("f_dc_1", 0.0),
            ("f_dc_2", 0.0),
            ("f_rest_0", 0.0),
            ("opacity", 5.0),
            ("scale_0", 0.0),
            ("scale_1", 0.0),
            ("scale_2", 0.0),
            ("rot_0", 0.0),
            ("rot_1", 0.0),
            ("rot_2", 0.0),
            ("rot_3", 1.0),
        ]);

        let ply = build_test_ply(props.as_slice(), &[row_small, row_large, row_mid], 0);
        let (header, mut cursor) = Scene::parse_file_header(ply).unwrap();
        let mut scene = Scene::new();
        scene.load(&mut cursor, &header).unwrap();

        assert_eq!(scene.source_row_indices, vec![1, 2, 0]);
    }

    #[test]
    fn scene_merge_preserves_source_row_mappings() {
        let mut a = Scene::new();
        a.splat_count = 2;
        a.source_row_indices = vec![3, 1];
        let mut b = Scene::new();
        b.splat_count = 1;
        b.source_row_indices = vec![2];

        let mut merged = Scene::new();
        merged.merge(&a);
        merged.merge(&b);

        assert_eq!(merged.splat_count, 3);
        assert_eq!(merged.source_row_indices, vec![3, 1, 2]);
    }

    #[test]
    fn ply_parser_rejects_missing_required_property() {
        let props = vec![
            "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "f_rest_0", "opacity", "scale_0",
            "scale_1", // scale_2 missing
            "rot_0", "rot_1", "rot_2", "rot_3",
        ];
        let row = HashMap::from([
            ("x", 0.0),
            ("y", 0.0),
            ("z", 0.0),
            ("f_dc_0", 0.0),
            ("f_dc_1", 0.0),
            ("f_dc_2", 0.0),
            ("f_rest_0", 0.0),
            ("opacity", 0.0),
            ("scale_0", 0.0),
            ("scale_1", 0.0),
            ("rot_0", 0.0),
            ("rot_1", 0.0),
            ("rot_2", 0.0),
            ("rot_3", 1.0),
        ]);
        let ply = build_test_ply(props.as_slice(), &[row], 0);
        let (header, mut cursor) = Scene::parse_file_header(ply).unwrap();
        let mut scene = Scene::new();
        let err = scene.load(&mut cursor, &header).unwrap_err();
        assert!(err.contains("scale_2"));
    }

    #[test]
    fn ply_parser_rejects_truncated_payload() {
        let props = vec![
            "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "f_rest_0", "opacity", "scale_0",
            "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
        ];
        let row = HashMap::from([
            ("x", 0.0),
            ("y", 0.0),
            ("z", 0.0),
            ("f_dc_0", 0.0),
            ("f_dc_1", 0.0),
            ("f_dc_2", 0.0),
            ("f_rest_0", 0.0),
            ("opacity", 0.0),
            ("scale_0", 0.0),
            ("scale_1", 0.0),
            ("scale_2", 0.0),
            ("rot_0", 0.0),
            ("rot_1", 0.0),
            ("rot_2", 0.0),
            ("rot_3", 1.0),
        ]);
        let ply = build_test_ply(props.as_slice(), &[row], 8);
        let (header, mut cursor) = Scene::parse_file_header(ply).unwrap();
        let mut scene = Scene::new();
        let err = scene.load(&mut cursor, &header).unwrap_err();
        assert!(err.contains("failed to read payload"));
    }
}

/// Loads a .ply or .splat file and returns a [Scene]
pub async fn load_scene() -> Scene {
    /*
    A WebAssembly page has a constant size of 65,536 bytes (or 64KB).
    Therefore, the maximum range that a WASM module can address,
    as WASM currently only allows 32-bit addressing, is 2^16 * 64KB = 4GB.
    */
    let mut scene = Scene::new();

    let file = rfd::AsyncFileDialog::new()
        .add_filter("3DGS model", &["ply", "splat", "spz"])
        .pick_file()
        .await;
    if let Some(f) = file.as_ref() {
        if f.file_name().contains(".ply") {
            let bytes = f.read().await;
            let (header, mut cursor) = match Scene::parse_file_header(bytes) {
                Ok((h, c)) => (h, c),
                Err(e) => {
                    log!("load_scene(): ERROR: {}", e);
                    unreachable!();
                }
            };
            scene.splat_count = header.splat_count;
            if let Err(e) = scene.load(&mut cursor, &header) {
                log!("load_scene(): ERROR loading PLY: {}", e);
                unreachable!();
            }
        } else if f.file_name().contains(".splat") {
            scene.buffer = f.read().await;
            scene.splat_count = scene.buffer.len() / 32; // 32bytes per splat
        } else {
            unreachable!();
        }
    }

    scene.generate_texture();

    log!("load_scene(): scene.splat_count={}", scene.splat_count);

    scene
}

/// Loads multiple .ply or .splat file and returns a [Vec<Vec<Scene>>] with shape [n_lod, n_tile]
pub async fn load_scene_vec() -> Vec<Vec<Scene>> {
    /*
    A WebAssembly page has a constant size of 65,536 bytes (or 64KB).
    Therefore, the maximum range that a WASM module can address,
    as WASM currently only allows 32-bit addressing, is 2^16 * 64KB = 4GB.
    */

    let file_vec = rfd::AsyncFileDialog::new()
        .set_title("Upload Tiles")
        .add_filter("3DGS model", &["ply", "splat", "spz"])
        .pick_files()
        .await;

    if file_vec.is_none() {
        return Vec::new();
    }

    let mut file_vec = file_vec.unwrap();
    let re = Regex::new(r"lod(\d+)_tile_(\d+)").unwrap();
    file_vec.sort_by_key(|s| {
        let filename = s.file_name();
        let caps = re.captures(filename.as_str()).unwrap();
        let strs = (caps.get(1).unwrap().as_str(), caps.get(2).unwrap().as_str());
        let nums = (
            strs.0.parse::<i32>().unwrap(),
            strs.1.parse::<i32>().unwrap(),
        );

        nums
    });
    let first_filename = file_vec.first().unwrap().file_name();
    let caps = re.captures(first_filename.as_str()).unwrap();
    let strs = (caps.get(1).unwrap().as_str(), caps.get(2).unwrap().as_str());
    let first_nums = (
        strs.0.parse::<i32>().unwrap(),
        strs.1.parse::<i32>().unwrap(),
    );
    let last_filename = file_vec.last().unwrap().file_name();
    let caps = re.captures(last_filename.as_str()).unwrap();
    let strs = (caps.get(1).unwrap().as_str(), caps.get(2).unwrap().as_str());
    let last_nums = (
        strs.0.parse::<i32>().unwrap(),
        strs.1.parse::<i32>().unwrap(),
    );

    let n_lod = last_nums.0 as usize - first_nums.0 as usize + 1;
    let n_tile = last_nums.1 as usize + 1;

    let mut scene_vec: Vec<Vec<Scene>> = Vec::new();

    for i in 0..n_lod {
        let mut lod_vec: Vec<Scene> = Vec::new();
        for j in 0..n_tile {
            let f = &file_vec[i * n_tile + j];
            let mut scene = Scene::new();

            if f.file_name().contains(".ply") {
                let bytes = f.read().await;
                let (header, mut cursor) = match Scene::parse_file_header(bytes) {
                    Ok((h, c)) => (h, c),
                    Err(e) => {
                        log!("load_scene(): ERROR: {}", e);
                        unreachable!();
                    }
                };
                scene.splat_count = header.splat_count;
                if let Err(e) = scene.load(&mut cursor, &header) {
                    log!("load_scene_vec(): ERROR loading PLY: {}", e);
                    unreachable!();
                }
            } else if f.file_name().contains(".splat") {
                scene.buffer = f.read().await;
                scene.splat_count = scene.buffer.len() / 32; // 32bytes per splat
            } else {
                unreachable!();
            }

            scene.generate_texture();

            log!("load_scene(): {}", f.file_name());
            log!("load_scene(): scene.splat_count={}", scene.splat_count);

            lod_vec.push(scene);
        }
        scene_vec.push(lod_vec);
    }

    scene_vec
}

pub struct SceneZipData {
    pub scene_vec: Vec<Vec<Scene>>,
    pub deformation_weights: Option<Vec<u8>>,
    pub catmull_rom_motion: Option<Arc<CatmullRomMotionSet>>,
}

pub async fn load_scene_zip() -> SceneZipData {
    /*
    A WebAssembly page has a constant size of 65,536 bytes (or 64KB).
    Therefore, the maximum range that a WASM module can address,
    as WASM currently only allows 32-bit addressing, is 2^16 * 64KB = 4GB.
    */

    let file_zip = rfd::AsyncFileDialog::new()
        .set_title("Upload Tiles (.zip)")
        .add_filter("Tiles", &["zip"])
        .pick_file()
        .await;

    if file_zip.is_none() {
        return SceneZipData {
            scene_vec: Vec::new(),
            deformation_weights: None,
            catmull_rom_motion: None,
        };
    }
    let file_zip = file_zip.unwrap().read().await;
    let file_cursor = Cursor::new(file_zip);
    let mut archive = zip::ZipArchive::new(file_cursor).unwrap();

    // Extract zip
    struct SceneFileEntry {
        index: usize,
        filename: String,
        lod_id: usize,
        tile_id: usize,
    }
    let re = Regex::new(r"tile(\d+)_lod(\d+)").unwrap();
    let mut file_vec: Vec<SceneFileEntry> = Vec::new();
    let mut deformation_weights_index: Option<usize> = None;
    let mut catmull_rom_meta_index: Option<usize> = None;
    let mut catmull_rom_entries: Vec<CatmullRomMotionZipEntry> = Vec::new();
    for i in 0..archive.len() {
        let file = archive.by_index(i).unwrap();
        let filename = file
            .enclosed_name()
            .unwrap()
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        let filename_lower = filename.to_ascii_lowercase();
        if filename_lower == "deformation_weights.bin" {
            deformation_weights_index = Some(i);
            continue;
        }
        if filename_lower == CATMULL_ROM_META_FILENAME {
            catmull_rom_meta_index = Some(i);
            continue;
        }
        if let Some((tile_id, lod_id)) = detect_motion_file(filename_lower.as_str()) {
            catmull_rom_entries.push(CatmullRomMotionZipEntry {
                index: i,
                filename,
                tile_id,
                lod_id,
            });
            continue;
        }
        if !(filename_lower.ends_with(".ply") || filename_lower.ends_with(".splat")) {
            continue;
        }
        let opt_caps = re.captures(filename.as_str());
        if let Some(caps) = opt_caps {
            let strs = (caps.get(1).unwrap().as_str(), caps.get(2).unwrap().as_str());
            let tile_id = strs.0.parse::<usize>().unwrap();
            let lod_id = strs.1.parse::<usize>().unwrap();
            let entry = SceneFileEntry {
                index: i,
                filename,
                lod_id,
                tile_id,
            };
            file_vec.push(entry);
        }
    }

    let deformation_weights = if let Some(index) = deformation_weights_index {
        let mut file = archive.by_index(index).unwrap();
        let mut bytes = vec![0_u8; file.size() as usize];
        file.read_exact(&mut bytes.as_mut_slice())
            .expect("Error loading deformation_weights.bin");
        log!(
            "load_scene_zip(): deformation_weights.bin loaded ({} bytes)",
            bytes.len()
        );
        Some(bytes)
    } else {
        log!("load_scene_zip(): deformation_weights.bin not found in zip.");
        None
    };

    if file_vec.is_empty() {
        return SceneZipData {
            scene_vec: Vec::new(),
            deformation_weights,
            catmull_rom_motion: None,
        };
    }

    file_vec.sort_by_key(|e| (e.lod_id, e.tile_id));
    let first_entry = file_vec.first().unwrap();
    let last_entry = file_vec.last().unwrap();

    let n_lod = last_entry.lod_id - first_entry.lod_id + 1;
    let n_tile = last_entry.tile_id as usize + 1;

    let mut scene_vec: Vec<Vec<Scene>> = Vec::with_capacity(n_lod);

    for i in 0..n_lod {
        let mut lod_vec: Vec<Scene> = Vec::with_capacity(n_tile);
        for j in 0..n_tile {
            let file_entry = &file_vec[i * n_tile + j];
            let mut scene = Scene::new();

            if file_entry.filename.contains(".ply") {
                let mut file = archive.by_index(file_entry.index).unwrap();
                let mut bytes = vec![0_u8; file.size() as usize];
                file.read_exact(&mut bytes.as_mut_slice())
                    .expect(format!("Error loading file: {}", file_entry.filename).as_str());
                let (header, mut cursor) = match Scene::parse_file_header(bytes) {
                    Ok((h, c)) => (h, c),
                    Err(e) => {
                        log!("load_scene(): ERROR: {}", e);
                        unreachable!();
                    }
                };
                scene.splat_count = header.splat_count;
                if let Err(e) = scene.load(&mut cursor, &header) {
                    log!("load_scene_zip(): ERROR loading PLY: {}", e);
                    unreachable!();
                }
            } else if file_entry.filename.contains(".splat") {
                let mut file = archive.by_index(file_entry.index).unwrap();
                let mut bytes = vec![0_u8; file.size() as usize];
                file.read_exact(&mut bytes.as_mut_slice())
                    .expect(format!("Error loading file: {}", file_entry.filename).as_str());
                scene.buffer = bytes;
                scene.splat_count = scene.buffer.len() / 32; // 32bytes per splat
            } else {
                unreachable!();
            }

            // scene.generate_texture();

            log!("load_scene(): {}", file_entry.filename);
            log!("load_scene(): scene.splat_count={}", scene.splat_count);

            lod_vec.push(scene);
        }
        scene_vec.push(lod_vec);
    }

    let catmull_rom_motion = if let Some(meta_index) = catmull_rom_meta_index {
        log!(
            "load_scene_zip(): detected Catmull-Rom motion meta and {} tile/LOD motion files.",
            catmull_rom_entries.len()
        );
        match load_catmull_rom_motion_from_zip(
            &mut archive,
            meta_index,
            catmull_rom_entries.as_slice(),
            scene_vec.as_slice(),
        ) {
            Ok(motion) => motion,
            Err(err) => {
                log!(
                    "load_scene_zip(): failed to load Catmull-Rom motion: {}",
                    err
                );
                None
            }
        }
    } else {
        if !catmull_rom_entries.is_empty() {
            log!(
                "load_scene_zip(): Catmull-Rom tile motion files found, but motion_catmull_rom_meta.pt is missing."
            );
        }
        None
    };

    SceneZipData {
        scene_vec,
        deformation_weights,
        catmull_rom_motion,
    }
}

/// Merges a vec of scenes into one
pub fn merge_scene(scene_vec: &Vec<Scene>) -> Scene {
    let mut new_scene = Scene::new();

    for scene in scene_vec {
        new_scene.merge(scene);
    }

    new_scene.generate_texture();

    log!(
        "merge_scene(): new_scene.splat_count={}",
        new_scene.splat_count
    );

    new_scene
}

pub fn translate_scene(scene: &Scene, offset: Vec3, gen_tex: bool) -> Scene {
    let mut new_scene = scene.clone();

    new_scene.translate(offset);

    if gen_tex {
        new_scene.generate_texture();
    }

    new_scene
}
