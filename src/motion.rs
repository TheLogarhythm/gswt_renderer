pub const MOTION_KNOT_COUNT: usize = 6;
pub const MOTION_ORIENTATION_COUNT: usize = 2;
pub const MOTION_EDGE_COLOR_COUNT: usize = 2;
pub const MOTION_GROUP_COUNT: usize = MOTION_ORIENTATION_COUNT * MOTION_EDGE_COLOR_COUNT;
pub const MOTION_SPLINE_FAMILY_COUNT: usize = 3;
pub const MOTION_PACKED_KNOT_COUNT: usize =
    MOTION_GROUP_COUNT * MOTION_SPLINE_FAMILY_COUNT * MOTION_KNOT_COUNT;

#[derive(Clone, Debug, PartialEq)]
pub struct MotionEditConfig {
    pub enabled: bool,
    pub amplitude: f32,
    pub edge_band: f32,
    pub wave_phase_span: f32,
    pub detail_amplitude: f32,
    pub spline_groups: [MotionSplineGroup; MOTION_GROUP_COUNT],
}

impl Default for MotionEditConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            amplitude: 0.0,
            edge_band: 0.12,
            wave_phase_span: 0.25,
            detail_amplitude: 0.0,
            spline_groups: [MotionSplineGroup::zero(); MOTION_GROUP_COUNT],
        }
    }
}

impl MotionEditConfig {
    pub fn zero_motion(&mut self) {
        self.spline_groups = [MotionSplineGroup::zero(); MOTION_GROUP_COUNT];
    }

    pub fn load_wave_preset(&mut self) {
        self.enabled = true;
        self.amplitude = 0.08;
        self.edge_band = 0.12;
        self.wave_phase_span = 0.35;
        self.detail_amplitude = 0.35;

        self.spline_groups = [
            MotionSplineGroup::wave_preset([0.0, 0.0, 1.0], 1.0),
            MotionSplineGroup::wave_preset([0.0, 0.0, 1.0], -1.0),
            MotionSplineGroup::wave_preset([0.0, 0.0, 1.0], 0.7),
            MotionSplineGroup::wave_preset([0.0, 0.0, 1.0], -0.7),
        ];
    }

    pub fn group(&self, orientation: MotionEdgeOrientation, edge_color: u32) -> &MotionSplineGroup {
        &self.spline_groups[motion_group_index(orientation, edge_color)]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionSplineGroup {
    pub base: VectorSpline6,
    pub wave: VectorSpline6,
    pub detail: VectorSpline6,
}

impl MotionSplineGroup {
    pub const fn zero() -> Self {
        Self {
            base: VectorSpline6::zero(),
            wave: VectorSpline6::zero(),
            detail: VectorSpline6::zero(),
        }
    }

    pub fn is_zero(&self) -> bool {
        self.base.is_zero() && self.wave.is_zero() && self.detail.is_zero()
    }

    fn wave_preset(axis: [f32; 3], sign: f32) -> Self {
        Self {
            base: VectorSpline6::from_axis(axis, 0.4 * sign),
            wave: VectorSpline6::from_axis(axis, 0.8 * sign),
            detail: VectorSpline6::from_axis([axis[1], axis[0], axis[2]], 0.35 * sign),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VectorSpline6 {
    pub knots: [[f32; 3]; MOTION_KNOT_COUNT],
}

impl VectorSpline6 {
    pub const fn zero() -> Self {
        Self {
            knots: [[0.0; 3]; MOTION_KNOT_COUNT],
        }
    }

    pub fn is_zero(&self) -> bool {
        self.knots.iter().all(|knot| knot == &[0.0; 3])
    }

    pub fn from_axis(axis: [f32; 3], amplitude: f32) -> Self {
        Self {
            knots: [
                [0.0, 0.0, 0.0],
                scale3(axis, amplitude),
                scale3(axis, 1.35 * amplitude),
                scale3(axis, 0.15 * amplitude),
                scale3(axis, -0.75 * amplitude),
                scale3(axis, -0.2 * amplitude),
            ],
        }
    }

    pub fn sample_periodic(&self, phase: f32) -> [f32; 3] {
        let scaled_phase = phase.rem_euclid(1.0) * MOTION_KNOT_COUNT as f32;
        let segment = scaled_phase.floor() as usize;
        let t = scaled_phase - segment as f32;

        let p0 = self.knots[(segment + MOTION_KNOT_COUNT - 1) % MOTION_KNOT_COUNT];
        let p1 = self.knots[segment % MOTION_KNOT_COUNT];
        let p2 = self.knots[(segment + 1) % MOTION_KNOT_COUNT];
        let p3 = self.knots[(segment + 2) % MOTION_KNOT_COUNT];

        catmull_rom_vec3(p0, p1, p2, p3, t)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MotionEdgeOrientation {
    Vertical,
    Horizontal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeSide {
    West,
    North,
    East,
    South,
}

impl EdgeSide {
    pub const fn orientation(self) -> MotionEdgeOrientation {
        match self {
            Self::West | Self::East => MotionEdgeOrientation::Vertical,
            Self::North | Self::South => MotionEdgeOrientation::Horizontal,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CanonicalEdgeCoordinate {
    pub s: f32,
    pub distance: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WangEdgeColors {
    pub west: u32,
    pub north: u32,
    pub east: u32,
    pub south: u32,
}

pub fn wang_edge_colors(tile_id: u32) -> WangEdgeColors {
    WangEdgeColors {
        west: tile_id / 8 % 2,
        north: tile_id / 4 % 2,
        east: tile_id / 2 % 2,
        south: tile_id % 2,
    }
}

pub fn motion_group_index(orientation: MotionEdgeOrientation, edge_color: u32) -> usize {
    let orientation_index = match orientation {
        MotionEdgeOrientation::Vertical => 0,
        MotionEdgeOrientation::Horizontal => 1,
    };
    orientation_index * MOTION_EDGE_COLOR_COUNT + (edge_color as usize % MOTION_EDGE_COLOR_COUNT)
}

pub fn canonical_edge_coordinate(side: EdgeSide, local_uv: [f32; 2]) -> CanonicalEdgeCoordinate {
    let u = local_uv[0].clamp(0.0, 1.0);
    let v = local_uv[1].clamp(0.0, 1.0);

    match side {
        EdgeSide::West => CanonicalEdgeCoordinate { s: v, distance: u },
        EdgeSide::East => CanonicalEdgeCoordinate {
            s: v,
            distance: 1.0 - u,
        },
        EdgeSide::South => CanonicalEdgeCoordinate { s: u, distance: v },
        EdgeSide::North => CanonicalEdgeCoordinate {
            s: u,
            distance: 1.0 - v,
        },
    }
}

pub fn pack_motion_spline_knots(config: &MotionEditConfig) -> [[f32; 4]; MOTION_PACKED_KNOT_COUNT] {
    let mut packed = [[0.0; 4]; MOTION_PACKED_KNOT_COUNT];
    let mut index = 0;

    for group in &config.spline_groups {
        for spline in [&group.base, &group.wave, &group.detail] {
            for knot in spline.knots {
                packed[index] = [knot[0], knot[1], knot[2], 0.0];
                index += 1;
            }
        }
    }

    packed
}

fn catmull_rom_vec3(p0: [f32; 3], p1: [f32; 3], p2: [f32; 3], p3: [f32; 3], t: f32) -> [f32; 3] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        catmull_rom_scalar(p0[0], p1[0], p2[0], p3[0], t, t2, t3),
        catmull_rom_scalar(p0[1], p1[1], p2[1], p3[1], t, t2, t3),
        catmull_rom_scalar(p0[2], p1[2], p2[2], p3[2], t, t2, t3),
    ]
}

fn catmull_rom_scalar(p0: f32, p1: f32, p2: f32, p3: f32, t: f32, t2: f32, t3: f32) -> f32 {
    0.5 * (2.0 * p1
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

fn scale3(v: [f32; 3], scale: f32) -> [f32; 3] {
    [v[0] * scale, v[1] * scale, v[2] * scale]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_motion_edit_config_preserves_existing_rendering() {
        let config = MotionEditConfig::default();

        assert!(!config.enabled);
        assert_eq!(config.amplitude, 0.0);
        assert_eq!(config.edge_band, 0.12);
        assert_eq!(config.wave_phase_span, 0.25);
        assert_eq!(config.detail_amplitude, 0.0);
        assert!(config.spline_groups.iter().all(MotionSplineGroup::is_zero));
    }

    #[test]
    fn zero_motion_clears_all_spline_groups() {
        let mut config = MotionEditConfig::default();
        config.load_wave_preset();

        config.zero_motion();

        assert!(config.spline_groups.iter().all(MotionSplineGroup::is_zero));
    }

    #[test]
    fn wave_preset_creates_base_wave_and_detail_motion() {
        let mut config = MotionEditConfig::default();

        config.load_wave_preset();

        assert!(
            config
                .spline_groups
                .iter()
                .any(|group| !group.base.is_zero())
        );
        assert!(
            config
                .spline_groups
                .iter()
                .any(|group| !group.wave.is_zero())
        );
        assert!(
            config
                .spline_groups
                .iter()
                .any(|group| !group.detail.is_zero())
        );
    }

    #[test]
    fn periodic_catmull_rom_wraps_six_knots() {
        let spline = VectorSpline6 {
            knots: [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
        };

        assert_eq!(spline.sample_periodic(0.0), [0.0, 0.0, 0.0]);
        assert_eq!(spline.sample_periodic(1.0 / 6.0), [1.0, 0.0, 0.0]);
        assert_eq!(spline.sample_periodic(2.0 / 6.0), [1.0, 1.0, 0.0]);
        assert_eq!(spline.sample_periodic(3.0 / 6.0), [0.0, 1.0, 0.0]);
        assert_eq!(spline.sample_periodic(4.0 / 6.0), [-1.0, 1.0, 0.0]);
        assert_eq!(spline.sample_periodic(5.0 / 6.0), [-1.0, 0.0, 0.0]);
        assert_eq!(spline.sample_periodic(1.0), spline.sample_periodic(0.0));
    }

    #[test]
    fn wang_edge_colors_follow_tile_id_bit_layout() {
        let colors = wang_edge_colors(0b1011);

        assert_eq!(
            colors,
            WangEdgeColors {
                west: 1,
                north: 0,
                east: 1,
                south: 1,
            }
        );
    }

    #[test]
    fn canonical_edge_coordinates_match_across_neighboring_tiles() {
        let west_of_right_tile = canonical_edge_coordinate(EdgeSide::West, [0.0, 0.25]);
        let east_of_left_tile = canonical_edge_coordinate(EdgeSide::East, [1.0, 0.25]);

        assert_eq!(west_of_right_tile.s, east_of_left_tile.s);
        assert_eq!(west_of_right_tile.distance, 0.0);
        assert_eq!(east_of_left_tile.distance, 0.0);
    }

    #[test]
    fn edge_distance_is_normalized_for_each_side() {
        assert_eq!(
            canonical_edge_coordinate(EdgeSide::West, [0.2, 0.5]).distance,
            0.2
        );
        assert_eq!(
            canonical_edge_coordinate(EdgeSide::East, [0.2, 0.5]).distance,
            0.8
        );
        assert_eq!(
            canonical_edge_coordinate(EdgeSide::South, [0.5, 0.2]).distance,
            0.2
        );
        assert_eq!(
            canonical_edge_coordinate(EdgeSide::North, [0.5, 0.2]).distance,
            0.8
        );
    }
}
