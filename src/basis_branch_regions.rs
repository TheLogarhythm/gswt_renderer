#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BasisGraphBranchDomain {
    GlobalBasis,
    FbmRegions,
}

impl BasisGraphBranchDomain {
    pub const ALL: [Self; 2] = [Self::GlobalBasis, Self::FbmRegions];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::GlobalBasis => "Global",
            Self::FbmRegions => "Spatial Variation",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BasisGraphRegionConfig {
    pub domain: BasisGraphBranchDomain,
    pub region_count: u32,
    pub region_size_world: f32,
    pub octaves: u32,
    pub warp_strength: f32,
    pub seed: u32,
}

impl Default for BasisGraphRegionConfig {
    fn default() -> Self {
        Self {
            domain: BasisGraphBranchDomain::GlobalBasis,
            region_count: 8,
            region_size_world: 8.0,
            octaves: 2,
            warp_strength: 0.35,
            seed: 13,
        }
    }
}

impl BasisGraphRegionConfig {
    pub const MAX_REGION_COUNT: u32 = 64;

    pub fn sanitized(self) -> Self {
        Self {
            domain: self.domain,
            region_count: self.region_count.clamp(1, Self::MAX_REGION_COUNT),
            region_size_world: if self.region_size_world.is_finite() {
                self.region_size_world.max(1e-3)
            } else {
                Self::default().region_size_world
            },
            octaves: self.octaves.clamp(1, 4),
            warp_strength: if self.warp_strength.is_finite() {
                self.warp_strength.clamp(0.0, 4.0)
            } else {
                Self::default().warp_strength
            },
            seed: self.seed,
        }
    }

    pub fn effective_region_count(self) -> u32 {
        let config = self.sanitized();
        match config.domain {
            BasisGraphBranchDomain::GlobalBasis => 1,
            BasisGraphBranchDomain::FbmRegions => config.region_count,
        }
    }
}

pub fn graph_state_index(
    region_id: usize,
    basis_id: usize,
    global_basis_count: usize,
) -> Option<usize> {
    if global_basis_count == 0 || basis_id >= global_basis_count {
        return None;
    }
    region_id
        .checked_mul(global_basis_count)
        .and_then(|base| base.checked_add(basis_id))
}

pub fn assign_branch_regions(base_means: &[[f32; 3]], config: BasisGraphRegionConfig) -> Vec<u32> {
    let config = config.sanitized();
    if config.domain == BasisGraphBranchDomain::GlobalBasis {
        return vec![0; base_means.len()];
    }
    let count = config.region_count.max(1);
    base_means
        .iter()
        .map(|mean| {
            let inv_size = 1.0 / config.region_size_world;
            let p = [mean[0] * inv_size, mean[1] * inv_size, mean[2] * inv_size];
            let n = fbm_noise3(p, config.octaves, config.warp_strength, config.seed);
            let n01 = (n * 0.5 + 0.5).clamp(0.0, 0.999_999);
            (n01 * count as f32).floor() as u32
        })
        .collect()
}

pub fn fbm_noise3(p: [f32; 3], octaves: u32, warp_strength: f32, seed: u32) -> f32 {
    let octaves = octaves.clamp(1, 4);
    let warp_strength = warp_strength.clamp(0.0, 4.0);
    let warp = [
        value_noise3(
            [p[0] * 0.5 + 11.3, p[1] * 0.5 - 7.1, p[2] * 0.5 + 3.7],
            seed ^ 0xA511_E9B3,
        ),
        value_noise3(
            [p[0] * 0.5 - 5.9, p[1] * 0.5 + 13.5, p[2] * 0.5 - 2.1],
            seed ^ 0x63D8_35F1,
        ),
        value_noise3(
            [p[0] * 0.5 + 2.4, p[1] * 0.5 + 8.2, p[2] * 0.5 - 10.6],
            seed ^ 0xB529_7A4D,
        ),
    ];
    let mut q = [
        p[0] + warp[0] * warp_strength,
        p[1] + warp[1] * warp_strength,
        p[2] + warp[2] * warp_strength,
    ];
    let mut amp = 0.5;
    let mut sum = 0.0;
    let mut norm = 0.0;
    for octave in 0..octaves {
        sum += amp * value_noise3(q, seed.wrapping_add(octave.wrapping_mul(1013)));
        norm += amp;
        q = [q[0] * 2.0, q[1] * 2.0, q[2] * 2.0];
        amp *= 0.5;
    }
    if norm > 0.0 {
        (sum / norm).clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

fn value_noise3(p: [f32; 3], seed: u32) -> f32 {
    let ix = p[0].floor() as i32;
    let iy = p[1].floor() as i32;
    let iz = p[2].floor() as i32;
    let fx = p[0] - ix as f32;
    let fy = p[1] - iy as f32;
    let fz = p[2] - iz as f32;
    let ux = fade(fx);
    let uy = fade(fy);
    let uz = fade(fz);

    let mut corners = [[[0.0_f32; 2]; 2]; 2];
    for dx in 0..2 {
        for dy in 0..2 {
            for dz in 0..2 {
                corners[dx][dy][dz] =
                    hash_noise(ix + dx as i32, iy + dy as i32, iz + dz as i32, seed);
            }
        }
    }

    let x00 = lerp(corners[0][0][0], corners[1][0][0], ux);
    let x10 = lerp(corners[0][1][0], corners[1][1][0], ux);
    let x01 = lerp(corners[0][0][1], corners[1][0][1], ux);
    let x11 = lerp(corners[0][1][1], corners[1][1][1], ux);
    let y0 = lerp(x00, x10, uy);
    let y1 = lerp(x01, x11, uy);
    lerp(y0, y1, uz)
}

fn fade(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn hash_noise(x: i32, y: i32, z: i32, seed: u32) -> f32 {
    let mut h = seed
        ^ (x as u32).wrapping_mul(0x8DA6_B343)
        ^ (y as u32).wrapping_mul(0xD816_3841)
        ^ (z as u32).wrapping_mul(0xCB1A_B31F);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7FEB_352D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846C_A68B);
    h ^= h >> 16;
    (h as f32 / u32::MAX as f32) * 2.0 - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_region_config_uses_global_basis_domain() {
        let config = BasisGraphRegionConfig::default();

        assert_eq!(config.domain, BasisGraphBranchDomain::GlobalBasis);
        assert_eq!(config.effective_region_count(), 1);
    }

    #[test]
    fn fbm_noise_is_deterministic_for_same_position_and_seed() {
        let a = fbm_noise3([0.25, -0.5, 1.75], 2, 0.35, 13);
        let b = fbm_noise3([0.25, -0.5, 1.75], 2, 0.35, 13);

        assert!((a - b).abs() < 1e-6);
    }

    #[test]
    fn assigned_region_ids_are_bounded_and_stable() {
        let config = BasisGraphRegionConfig {
            domain: BasisGraphBranchDomain::FbmRegions,
            region_count: 8,
            region_size_world: 4.0,
            octaves: 2,
            warp_strength: 0.35,
            seed: 13,
        };
        let means = vec![[0.0, 0.0, 0.0], [2.0, 1.0, -0.5], [9.0, 0.25, 3.0]];

        let a = assign_branch_regions(means.as_slice(), config);
        let b = assign_branch_regions(means.as_slice(), config);

        assert_eq!(a, b);
        assert_eq!(a.len(), means.len());
        assert!(a.iter().all(|region| *region < config.region_count));
    }

    #[test]
    fn changing_seed_or_region_size_can_change_assignment() {
        let base = BasisGraphRegionConfig {
            domain: BasisGraphBranchDomain::FbmRegions,
            region_count: 16,
            region_size_world: 4.0,
            octaves: 2,
            warp_strength: 0.35,
            seed: 13,
        };
        let means = vec![[1.5, 2.0, 3.0], [5.0, 2.5, -1.0], [8.0, -3.0, 0.5]];

        let seed_changed = assign_branch_regions(
            means.as_slice(),
            BasisGraphRegionConfig { seed: 99, ..base },
        );
        let size_changed = assign_branch_regions(
            means.as_slice(),
            BasisGraphRegionConfig {
                region_size_world: 1.25,
                ..base
            },
        );
        let original = assign_branch_regions(means.as_slice(), base);

        assert!(seed_changed != original || size_changed != original);
    }

    #[test]
    fn graph_state_index_is_row_major_by_region_then_basis() {
        assert_eq!(graph_state_index(0, 2, 4), Some(2));
        assert_eq!(graph_state_index(1, 0, 4), Some(4));
        assert_eq!(graph_state_index(2, 3, 4), Some(11));
        assert_eq!(graph_state_index(0, 4, 4), None);
        assert_eq!(graph_state_index(1, 0, 0), None);
    }
}
