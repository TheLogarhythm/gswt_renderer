use crate::basis_bank_motion::basis_bank_delta;

pub const DEFAULT_BASIS_EDIT_PACKED: [f32; 4] = [0.0, 1.0, 0.0, 1.0];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BasisKnotEditPlane {
    XY,
    XZ,
    YZ,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BasisKnotEditState {
    original_knots: Vec<f32>,
    edited_knots: Vec<f32>,
    basis_count: usize,
    knot_count: usize,
    editable_knot_count: usize,
}

impl BasisKnotEditState {
    pub fn new(
        original_knots: Vec<f32>,
        basis_count: usize,
        knot_count: usize,
        editable_knot_count: usize,
    ) -> Self {
        let editable_knot_count = editable_knot_count.min(knot_count);
        Self {
            edited_knots: original_knots.clone(),
            original_knots,
            basis_count,
            knot_count,
            editable_knot_count,
        }
    }

    pub fn matches_shape(&self, basis_count: usize, knot_count: usize) -> bool {
        self.basis_count() == basis_count
            && self.knot_count() == knot_count
            && self.edited_knots.len() == basis_count.saturating_mul(knot_count).saturating_mul(3)
    }

    pub fn basis_count(&self) -> usize {
        self.basis_count
    }

    pub fn knot_count(&self) -> usize {
        self.knot_count
    }

    pub fn editable_knot_count(&self) -> usize {
        self.editable_knot_count
    }

    pub fn edited_knots(&self) -> &[f32] {
        self.edited_knots.as_slice()
    }

    pub fn original_knots(&self) -> &[f32] {
        self.original_knots.as_slice()
    }

    pub fn knot(&self, basis_id: usize, knot: usize) -> Option<[f32; 3]> {
        let base = self.knot_base(basis_id, knot)?;
        Some([
            self.edited_knots[base],
            self.edited_knots[base + 1],
            self.edited_knots[base + 2],
        ])
    }

    pub fn knot_is_edited(&self, basis_id: usize, knot: usize) -> bool {
        let Some(base) = self.knot_base(basis_id, knot) else {
            return false;
        };
        self.edited_knots[base..base + 3] != self.original_knots[base..base + 3]
    }

    pub fn basis_is_edited(&self, basis_id: usize) -> bool {
        if basis_id >= self.basis_count {
            return false;
        }
        let start = basis_id * self.knot_count * 3;
        let end = start + self.knot_count * 3;
        self.edited_knots[start..end] != self.original_knots[start..end]
    }

    pub fn set_knot(&mut self, basis_id: usize, knot: usize, point: [f32; 3]) -> bool {
        if knot >= self.editable_knot_count || !point.iter().all(|v| v.is_finite()) {
            return false;
        }
        let Some(base) = self.knot_base(basis_id, knot) else {
            return false;
        };
        self.edited_knots[base..base + 3].copy_from_slice(&point);
        true
    }

    pub fn reset_knot(&mut self, basis_id: usize, knot: usize) -> bool {
        if knot >= self.editable_knot_count {
            return false;
        }
        let Some(base) = self.knot_base(basis_id, knot) else {
            return false;
        };
        self.edited_knots[base..base + 3].copy_from_slice(&self.original_knots[base..base + 3]);
        true
    }

    pub fn reset_basis(&mut self, basis_id: usize) -> bool {
        if basis_id >= self.basis_count {
            return false;
        }
        let start = basis_id * self.knot_count * 3;
        let end = start + self.knot_count * 3;
        self.edited_knots[start..end].copy_from_slice(&self.original_knots[start..end]);
        true
    }

    pub fn reset_all(&mut self) {
        self.edited_knots.copy_from_slice(&self.original_knots);
    }

    fn knot_base(&self, basis_id: usize, knot: usize) -> Option<usize> {
        if basis_id >= self.basis_count || knot >= self.knot_count {
            return None;
        }
        Some((basis_id * self.knot_count + knot) * 3)
    }
}

pub fn apply_basis_knot_plane_delta(
    point: [f32; 3],
    plane: BasisKnotEditPlane,
    delta: [f32; 2],
) -> Option<[f32; 3]> {
    if !point.iter().all(|v| v.is_finite()) || !delta.iter().all(|v| v.is_finite()) {
        return None;
    }
    let edited = match plane {
        BasisKnotEditPlane::XY => [point[0] + delta[0], point[1] + delta[1], point[2]],
        BasisKnotEditPlane::XZ => [point[0] + delta[0], point[1], point[2] + delta[1]],
        BasisKnotEditPlane::YZ => [point[0], point[1] + delta[0], point[2] + delta[1]],
    };
    Some(edited)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BasisEditOverride {
    pub enabled: bool,
    pub amplitude_scale: f32,
    pub phase_offset: f32,
    pub time_scale: f32,
}

impl Default for BasisEditOverride {
    fn default() -> Self {
        Self {
            enabled: false,
            amplitude_scale: 1.0,
            phase_offset: 0.0,
            time_scale: 1.0,
        }
    }
}

impl BasisEditOverride {
    pub fn edited_time(self, time01: f32) -> f32 {
        if self.enabled {
            time01 * self.time_scale + self.phase_offset
        } else {
            time01
        }
    }

    pub fn edited_delta(self, delta: [f32; 3]) -> [f32; 3] {
        if self.enabled {
            [
                delta[0] * self.amplitude_scale,
                delta[1] * self.amplitude_scale,
                delta[2] * self.amplitude_scale,
            ]
        } else {
            delta
        }
    }

    pub fn packed(self) -> [f32; 4] {
        [
            if self.enabled { 1.0 } else { 0.0 },
            self.amplitude_scale,
            self.phase_offset,
            self.time_scale,
        ]
    }
}

pub fn edited_basis_bank_delta(
    basis_knots: &[f32],
    basis_count: usize,
    knot_count: usize,
    basis_id: usize,
    time01: f32,
    edit: BasisEditOverride,
) -> [f32; 3] {
    let delta = basis_bank_delta(
        basis_knots,
        basis_count,
        knot_count,
        basis_id,
        edit.edited_time(time01),
    );
    edit.edited_delta(delta)
}

pub fn pack_basis_edit_overrides(
    edits: &[BasisEditOverride],
    global_basis_count: usize,
) -> Vec<[f32; 4]> {
    let mut packed = vec![DEFAULT_BASIS_EDIT_PACKED; global_basis_count];
    for (dst, edit) in packed.iter_mut().zip(edits.iter()) {
        *dst = edit.packed();
    }
    packed
}

pub fn resize_basis_edit_overrides(edits: &mut Vec<BasisEditOverride>, global_basis_count: usize) {
    edits.resize(global_basis_count, BasisEditOverride::default());
}

pub fn reset_basis_edit(edits: &mut [BasisEditOverride], basis_id: usize) {
    if let Some(edit) = edits.get_mut(basis_id) {
        *edit = BasisEditOverride::default();
    }
}

pub fn reset_all_basis_edits(edits: &mut [BasisEditOverride]) {
    edits.fill(BasisEditOverride::default());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn single_basis_knots() -> Vec<f32> {
        vec![
            0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, //
            2.0, 0.0, 0.0, //
            3.0, 0.0, 0.0, //
        ]
    }

    #[test]
    fn default_edit_override_samples_identically_to_original_basis() {
        let knots = single_basis_knots();
        let edit = BasisEditOverride::default();

        assert_eq!(
            edited_basis_bank_delta(&knots, 1, 4, 0, 0.25, edit),
            [1.0, 0.0, 0.0]
        );
    }

    #[test]
    fn enabled_amplitude_scale_multiplies_sampled_displacement() {
        let knots = single_basis_knots();
        let edit = BasisEditOverride {
            enabled: true,
            amplitude_scale: 2.0,
            phase_offset: 0.0,
            time_scale: 1.0,
        };

        assert_eq!(
            edited_basis_bank_delta(&knots, 1, 4, 0, 0.25, edit),
            [2.0, 0.0, 0.0]
        );
    }

    #[test]
    fn phase_offset_shifts_sample_time_and_wraps_periodically() {
        let knots = single_basis_knots();
        let edit = BasisEditOverride {
            enabled: true,
            amplitude_scale: 1.0,
            phase_offset: 0.25,
            time_scale: 1.0,
        };

        assert_eq!(
            edited_basis_bank_delta(&knots, 1, 4, 0, 1.0, edit),
            [1.0, 0.0, 0.0]
        );
    }

    #[test]
    fn time_scale_changes_sample_time_and_wraps_periodically() {
        let knots = single_basis_knots();
        let edit = BasisEditOverride {
            enabled: true,
            amplitude_scale: 1.0,
            phase_offset: 0.0,
            time_scale: 2.0,
        };

        assert_eq!(
            edited_basis_bank_delta(&knots, 1, 4, 0, 0.625, edit),
            [1.0, 0.0, 0.0]
        );
    }

    #[test]
    fn edit_buffer_packing_emits_one_vec4_per_global_basis_with_defaults() {
        let edits = vec![
            BasisEditOverride::default(),
            BasisEditOverride {
                enabled: true,
                amplitude_scale: 0.5,
                phase_offset: -0.25,
                time_scale: 1.5,
            },
        ];

        let packed = pack_basis_edit_overrides(&edits, 3);

        assert_eq!(packed.len(), 3);
        assert_eq!(packed[0], [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(packed[1], [1.0, 0.5, -0.25, 1.5]);
        assert_eq!(packed[2], [0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn reset_helpers_restore_selected_or_all_edits_to_defaults() {
        let edited = BasisEditOverride {
            enabled: true,
            amplitude_scale: 0.5,
            phase_offset: 0.25,
            time_scale: 2.0,
        };
        let mut edits = vec![edited; 3];

        reset_basis_edit(&mut edits, 1);
        assert_eq!(edits[0], edited);
        assert_eq!(edits[1], BasisEditOverride::default());
        assert_eq!(edits[2], edited);

        reset_all_basis_edits(&mut edits);
        assert!(
            edits
                .iter()
                .all(|edit| *edit == BasisEditOverride::default())
        );
    }

    #[test]
    fn plane_drag_updates_only_projected_components() {
        let point = [1.0, 2.0, 3.0];

        assert_eq!(
            apply_basis_knot_plane_delta(point, BasisKnotEditPlane::XY, [0.5, -1.0]).unwrap(),
            [1.5, 1.0, 3.0]
        );
        assert_eq!(
            apply_basis_knot_plane_delta(point, BasisKnotEditPlane::XZ, [0.5, -1.0]).unwrap(),
            [1.5, 2.0, 2.0]
        );
        assert_eq!(
            apply_basis_knot_plane_delta(point, BasisKnotEditPlane::YZ, [0.5, -1.0]).unwrap(),
            [1.0, 2.5, 2.0]
        );
    }

    #[test]
    fn plane_drag_rejects_non_finite_values() {
        assert!(
            apply_basis_knot_plane_delta([1.0, 2.0, 3.0], BasisKnotEditPlane::XY, [f32::NAN, 0.0],)
                .is_none()
        );
    }

    #[test]
    fn knot_edit_state_edits_source_and_closure_knots() {
        let original = vec![
            0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, //
            2.0, 0.0, 0.0, //
            99.0, 99.0, 99.0, //
        ];
        let mut edits = BasisKnotEditState::new(original, 1, 4, 4);

        assert!(edits.set_knot(0, 1, [1.0, 2.0, 3.0]));
        assert_eq!(edits.knot(0, 1), Some([1.0, 2.0, 3.0]));
        assert!(edits.set_knot(0, 3, [4.0, 5.0, 6.0]));
        assert_eq!(edits.knot(0, 3), Some([4.0, 5.0, 6.0]));
    }

    #[test]
    fn knot_edit_reset_helpers_restore_original_knots() {
        let original = vec![
            0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, //
            2.0, 0.0, 0.0, //
            10.0, 0.0, 0.0, //
            11.0, 0.0, 0.0, //
            12.0, 0.0, 0.0, //
        ];
        let mut edits = BasisKnotEditState::new(original.clone(), 2, 3, 3);

        assert!(edits.set_knot(0, 1, [5.0, 5.0, 5.0]));
        assert!(edits.set_knot(1, 1, [6.0, 6.0, 6.0]));
        edits.reset_knot(0, 1);
        assert_eq!(edits.knot(0, 1), Some([1.0, 0.0, 0.0]));
        assert_eq!(edits.knot(1, 1), Some([6.0, 6.0, 6.0]));

        edits.reset_basis(1);
        assert_eq!(edits.knot(1, 1), Some([11.0, 0.0, 0.0]));

        assert!(edits.set_knot(0, 2, [7.0, 7.0, 7.0]));
        edits.reset_all();
        assert_eq!(edits.edited_knots(), original.as_slice());
    }

    #[test]
    fn edited_knot_sampling_uses_modified_geometry() {
        let original = single_basis_knots();
        let mut edits = BasisKnotEditState::new(original, 1, 4, 4);

        assert_eq!(
            basis_bank_delta(edits.edited_knots(), 1, 4, 0, 0.25),
            [1.0, 0.0, 0.0]
        );
        assert!(edits.set_knot(0, 1, [1.0, 2.0, 0.0]));
        assert_eq!(
            basis_bank_delta(edits.edited_knots(), 1, 4, 0, 0.25),
            [1.0, 2.0, 0.0]
        );
    }

    #[test]
    fn edited_closure_knot_changes_sampling_near_loop_boundary() {
        let original = single_basis_knots();
        let mut edits = BasisKnotEditState::new(original, 1, 4, 4);
        let before = basis_bank_delta(edits.edited_knots(), 1, 4, 0, 0.875);

        assert!(edits.set_knot(0, 3, [3.0, 4.0, 0.0]));
        let after = basis_bank_delta(edits.edited_knots(), 1, 4, 0, 0.875);

        assert_ne!(before, after);
        assert!(after[1] > before[1]);

        edits.reset_knot(0, 3);
        assert_eq!(
            basis_bank_delta(edits.edited_knots(), 1, 4, 0, 0.875),
            before
        );
    }
}
