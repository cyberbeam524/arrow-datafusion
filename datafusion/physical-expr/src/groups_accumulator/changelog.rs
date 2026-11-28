// Example state helper for invertible aggregates (COUNT / SUM / AVG) with +1 / -1 weights.
// Wire into your accumulator or changelog state module in `datafusion/physical-expr/src/`.
use arrow::array::{ArrayRef, Float64Array, Int64Array};
use datafusion_common::{Result, exec_err, plan_err};
use std::mem::size_of;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangelogWeight {
    Insert = 1,
    Retract = -1,
}

impl ChangelogWeight {
    pub fn try_from_i8(w: i8) -> Result<Self> {
        match w {
            1 => Ok(Self::Insert),
            -1 => Ok(Self::Retract),
            other => plan_err!("Invalid changelog weight {other}; expected +1 or -1"),
        }
    }

    pub const fn as_i64(self) -> i64 {
        self as i64
    }
}

#[derive(Debug, Clone, Default)]
pub struct InvertibleGroupState {
    pub counts: Vec<i64>,
    pub sums: Vec<i64>,
}

impl InvertibleGroupState {
    pub fn resize_to(&mut self, num_groups: usize) {
        if self.counts.len() < num_groups {
            self.counts.resize(num_groups, 0);
            self.sums.resize(num_groups, 0);
        }
    }

    pub fn apply_row(
        &mut self,
        group_idx: usize,
        weight: ChangelogWeight,
        val: Option<i64>,
    ) -> Result<()> {
        let sign = weight.as_i64();
        if let Some(v) = val {
            let next_count = self.counts[group_idx] + sign;
            if next_count < 0 {
                return exec_err!(
                    "Changelog retraction underflow for group {group_idx}: count would be {next_count}"
                );
            }
            self.counts[group_idx] = next_count;
            self.sums[group_idx] += sign * v;
        }
        Ok(())
    }

    pub fn active_group_indices(&self) -> Vec<usize> {
        self.counts
            .iter()
            .enumerate()
            .filter_map(|(idx, &cnt)| (cnt > 0).then_some(idx))
            .collect()
    }

    pub fn evaluate_outputs(&self) -> (ArrayRef, ArrayRef, ArrayRef) {
        let count_arr: Vec<Option<i64>> =
            self.counts.iter().map(|&c| (c > 0).then_some(c)).collect();
        let sum_arr: Vec<Option<i64>> = self
            .counts
            .iter()
            .zip(&self.sums)
            .map(|(&c, &s)| (c > 0).then_some(s))
            .collect();
        let avg_arr: Vec<Option<f64>> = self
            .counts
            .iter()
            .zip(&self.sums)
            .map(|(&c, &s)| (c > 0).then(|| s as f64 / c as f64))
            .collect();

        (
            Arc::new(Int64Array::from(count_arr)),
            Arc::new(Int64Array::from(sum_arr)),
            Arc::new(Float64Array::from(avg_arr)),
        )
    }

    pub fn size_bytes(&self) -> usize {
        self.counts.capacity() * size_of::<i64>()
            + self.sums.capacity() * size_of::<i64>()
    }
}
