use std::sync::Arc;

use crate::aggregates::group_values::GroupValues;
use crate::aggregates::group_values::new_group_values;
use crate::aggregates::order::GroupClustering;
use arrow::array::{Array, ArrayRef, AsArray, Int8Array, RecordBatch, UInt32Array};
use arrow::compute;
use arrow::datatypes::{Int64Type, SchemaRef};
use datafusion_common::{Result, exec_err, not_impl_err};
use datafusion_expr::GroupSelection;
use datafusion_physical_expr::groups_accumulator::changelog::{
    ChangelogWeight, InvertibleGroupState,
};

/// Incremental changelog aggregation engine over `+1` (insert/after) and `-1`
/// (retract/before) micro-batches using dense `GroupValues` indices and
/// non-destructive `GroupSelection` reads.
pub struct ChangelogAggEngine {
    pub group_values: Box<dyn GroupValues>,
    pub state: InvertibleGroupState,
}

impl ChangelogAggEngine {
    pub fn new(group_values: Box<dyn GroupValues>) -> Self {
        Self {
            group_values,
            state: InvertibleGroupState::default(),
        }
    }

    /// Expects columns: `[group_key (Utf8), val (Int64 nullable), weight (Int8: +1 / -1)]`.
    pub fn apply_changelog_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        let keys: ArrayRef = Arc::new(batch.column(0).as_string::<i32>().clone());
        let vals = batch.column(1).as_primitive::<Int64Type>();
        let weights = match batch.column(2).as_any().downcast_ref::<Int8Array>() {
            Some(arr) => arr,
            None => return exec_err!("weight column must be Int8"),
        };

        let mut group_ids = Vec::with_capacity(batch.num_rows());
        self.group_values.intern(&[keys], &mut group_ids)?;
        self.state.resize_to(self.group_values.len());

        for (row, &gid) in group_ids.iter().enumerate() {
            let weight = ChangelogWeight::try_from_i8(weights.value(row))?;
            let val = (!vals.is_null(row)).then(|| vals.value(row));
            self.state.apply_row(gid, weight, val)?;
        }

        Ok(())
    }

    /// Emits a snapshot of currently active (`count > 0`) groups without
    /// rebuilding or shifting interned `GroupValues` slots.
    pub fn snapshot_active(
        &mut self,
        out_schema: SchemaRef,
    ) -> Result<Option<RecordBatch>> {
        let active_indices = self.state.active_group_indices();
        if active_indices.is_empty() {
            return Ok(None);
        }

        let selection =
            GroupSelection::try_from_indices(&active_indices, self.group_values.len())?;
        let key_arrays = if self.group_values.supports_values_preserving() {
            self.group_values.values_preserving(selection)?
        } else {
            return not_impl_err!(
                "GroupValues implementation does not support values_preserving"
            );
        };

        let key_col: ArrayRef = match key_arrays.into_iter().next() {
            Some(col) => col,
            None => return exec_err!("expected interned key column"),
        };

        let (all_counts, all_sums, all_avgs) = self.state.evaluate_outputs();
        let idx_array = UInt32Array::from(
            active_indices.iter().map(|&i| i as u32).collect::<Vec<_>>(),
        );

        let active_counts = compute::take(all_counts.as_ref(), &idx_array, None)?;
        let active_sums = compute::take(all_sums.as_ref(), &idx_array, None)?;
        let active_avgs = compute::take(all_avgs.as_ref(), &idx_array, None)?;

        let batch = RecordBatch::try_new(
            out_schema,
            vec![key_col, active_counts, active_sums, active_avgs],
        )?;
        Ok(Some(batch))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};

    fn make_changelog_batch(rows: &[(&str, Option<i64>, i8)]) -> Result<RecordBatch> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("val", DataType::Int64, true),
            Field::new("weight", DataType::Int8, false),
        ]));

        let keys: Vec<&str> = rows.iter().map(|(k, _, _)| *k).collect();
        let vals: Vec<Option<i64>> = rows.iter().map(|(_, v, _)| *v).collect();
        let weights: Vec<i8> = rows.iter().map(|(_, _, w)| *w).collect();

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(keys)),
                Arc::new(Int64Array::from(vals)),
                Arc::new(Int8Array::from(weights)),
            ],
        )
        .map_err(Into::into)
    }

    fn make_out_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("count", DataType::Int64, true),
            Field::new("sum", DataType::Int64, true),
            Field::new("avg", DataType::Float64, true),
        ]))
    }

    #[test]
    fn test_changelog_insert_and_retract() -> Result<()> {
        // Requires a `GroupValues` implementation that supports `values_preserving`
        // (or substitute your crate's `GroupValues` constructor).
        let gv_schema =
            Arc::new(Schema::new(vec![Field::new("key", DataType::Utf8, false)]));
        let group_values = new_group_values(gv_schema, &GroupClustering::None)?;
        assert!(group_values.supports_values_preserving());

        let mut engine = ChangelogAggEngine::new(group_values);
        let out_schema = make_out_schema();

        // Insert +1 rows across three groups: a (10, 20), b (5), c (100)
        let batch1 = make_changelog_batch(&[
            ("a", Some(10), 1),
            ("a", Some(20), 1),
            ("b", Some(5), 1),
            ("c", Some(100), 1),
        ])?;
        engine.apply_changelog_batch(&batch1)?;

        // Retract one row from "a" (-1 for 10) and fully retract all rows for "b" (-1 for 5)
        let batch2 = make_changelog_batch(&[("a", Some(10), -1), ("b", Some(5), -1)])?;
        engine.apply_changelog_batch(&batch2)?;

        let snapshot = engine
            .snapshot_active(out_schema)?
            .expect("expected active groups a and c after retracting b");

        let out_keys = snapshot.column(0).as_string::<i32>();
        let out_counts = snapshot.column(1).as_primitive::<Int64Type>();
        let out_sums = snapshot.column(2).as_primitive::<Int64Type>();
        let out_avgs = snapshot
            .column(3)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("expected Float64Array for avg");

        assert_eq!(snapshot.num_rows(), 2);
        assert_eq!(out_keys.value(0), "a");
        assert_eq!(out_counts.value(0), 1);
        assert_eq!(out_sums.value(0), 20);
        assert!((out_avgs.value(0) - 20.0).abs() < 1e-12);

        assert_eq!(out_keys.value(1), "c");
        assert_eq!(out_counts.value(1), 1);
        assert_eq!(out_sums.value(1), 100);
        assert!((out_avgs.value(1) - 100.0).abs() < 1e-12);

        // Retracting more than inserted should return an execution error
        let underflow_batch = make_changelog_batch(&[("b", Some(5), -1)])?;
        assert!(engine.apply_changelog_batch(&underflow_batch).is_err());

        Ok(())
    }
}
