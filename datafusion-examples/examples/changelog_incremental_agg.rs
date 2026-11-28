use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{
    Array, ArrayRef, AsArray, Float64Array, Int8Array, Int64Array, RecordBatch, StringArray,
};
use arrow::datatypes::{DataType, Field, Int64Type, Schema, SchemaRef};
use datafusion::common::Result;
use datafusion::datasource::MemTable;
use datafusion::execution::context::SessionContext;
use datafusion::logical_expr::{EmitTo, GroupSelection};
use datafusion::physical_plan::aggregates::group_values::{
    GroupValues, new_group_values,
};
use datafusion::physical_plan::aggregates::order::GroupClustering;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GroupAggState {
    count: i64,
    sum: i64,
    active: bool,
}

impl GroupAggState {
    fn new() -> Self {
        Self {
            count: 0,
            sum: 0,
            active: false,
        }
    }

    fn apply(&mut self, weight: i8, val: Option<i64>) {
        let w = i64::from(weight);
        if let Some(v) = val {
            self.count += w;
            self.sum += w * v;
        }
        self.active = self.count > 0;
    }

    fn avg(&self) -> Option<f64> {
        if self.count <= 0 {
            None
        } else {
            Some(self.sum as f64 / self.count as f64)
        }
    }
}

struct IncrementalChangelogAgg {
    key_schema: SchemaRef,
    group_values: Box<dyn GroupValues>,
    states: Vec<GroupAggState>,
    dirty: Vec<bool>,
}

impl IncrementalChangelogAgg {
    fn try_new() -> Result<Self> {
        let key_schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Utf8, true)]));
        let group_values = new_group_values(Arc::clone(&key_schema), &GroupClustering::None)?;
        Ok(Self {
            key_schema,
            group_values,
            states: Vec::new(),
            dirty: Vec::new(),
        })
    }

    fn apply_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        let keys = batch.column(0).as_string::<i32>();
        let vals = batch.column(1).as_primitive::<Int64Type>();
        let weights = batch
            .column(2)
            .as_any()
            .downcast_ref::<Int8Array>()
            .expect("weight column must be Int8");

        let mut group_ids = Vec::with_capacity(batch.num_rows());
        let key_cols: Vec<ArrayRef> = vec![Arc::new(keys.clone())];
        self.group_values.intern(&key_cols, &mut group_ids)?;

        let total_groups = self.group_values.len();
        if self.states.len() < total_groups {
            self.states.resize(total_groups, GroupAggState::new());
            self.dirty.resize(total_groups, false);
        }

        for row in 0..batch.num_rows() {
            let gid = group_ids[row];
            let w = weights.value(row);
            let v = if vals.is_null(row) {
                None
            } else {
                Some(vals.value(row))
            };
            self.states[gid].apply(w, v);
            self.dirty[gid] = true;
        }

        Ok(())
    }

    fn active_group_count(&self) -> usize {
        self.states.iter().filter(|s| s.active).count()
    }

    fn snapshot_active(&mut self) -> Result<BTreeMap<String, (i64, i64, Option<f64>)>> {
        let active_indices: Vec<usize> = self
            .states
            .iter()
            .enumerate()
            .filter_map(|(idx, s)| s.active.then_some(idx))
            .collect();

        let mut out = BTreeMap::new();
        if active_indices.is_empty() {
            return Ok(out);
        }

        let sel = GroupSelection::try_from_indices(&active_indices, self.group_values.len())?;
        let key_arrays = if self.group_values.supports_values_preserving() {
            self.group_values.values_preserving(sel)?
        } else {
            let all_cols = self.group_values.emit(EmitTo::All)?;
            let mut groups = Vec::new();
            let _ = self.group_values.intern(&all_cols, &mut groups);
            all_cols
        };

        let key_arr = key_arrays[0].as_string::<i32>();
        for (pos, &gid) in active_indices.iter().enumerate() {
            let k = key_arr.value(pos).to_string();
            let st = self.states[gid];
            out.insert(k, (st.count, st.sum, st.avg()));
        }
        Ok(out)
    }
}

fn make_changelog_batch(
    schema: SchemaRef,
    rows: &[(&str, Option<i64>, i8)],
) -> Result<RecordBatch> {
    let keys: Vec<&str> = rows.iter().map(|(k, _, _)| *k).collect();
    let vals: Vec<Option<i64>> = rows.iter().map(|(_, v, _)| *v).collect();
    let weights: Vec<i8> = rows.iter().map(|(_, _, w)| *w).collect();

    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(keys)),
            Arc::new(Int64Array::from(vals)),
            Arc::new(Int8Array::from(weights)),
        ],
    )?)
}

fn build_grouped_batch(
    key_schema: SchemaRef,
    keys: Vec<String>,
    vals: Vec<Option<i64>>,
) -> Result<RecordBatch> {
    Ok(RecordBatch::try_new(
        key_schema,
        vec![
            Arc::new(StringArray::from(keys)),
            Arc::new(Int64Array::from(vals)),
        ],
    )?)
}

async fn run_batch_sql_reference(
    materialized: &HashMap<String, Vec<Option<i64>>>,
) -> Result<BTreeMap<String, (i64, i64, Option<f64>)>> {
    let mut keys = Vec::new();
    let mut vals = Vec::new();
    for (k, list) in materialized {
        for v in list {
            keys.push(k.as_str());
            vals.push(*v);
        }
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new("val", DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(keys)),
            Arc::new(Int64Array::from(vals)),
        ],
    )?;

    let ctx = SessionContext::new();
    let table = MemTable::try_new(schema, vec![vec![batch]])?;
    ctx.register_table("t", Arc::new(table))?;

    let df = ctx
        .sql(
            "SELECT key, COUNT(val) AS cnt, SUM(val) AS total, AVG(val) AS avg_val \
             FROM t GROUP BY key ORDER BY key",
        )
        .await?;
    let batches = df.collect().await?;

    let mut result = BTreeMap::new();
    for b in batches {
        let k_col = b.column(0).as_string::<i32>();
        let cnt_col = b.column(1).as_primitive::<Int64Type>();
        let sum_col = b.column(2).as_primitive::<Int64Type>();
        let avg_col = b
            .column(3)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        for r in 0..b.num_rows() {
            let k = k_col.value(r).to_string();
            let cnt = cnt_col.value(r);
            if cnt > 0 {
                let sum = if sum_col.is_null(r) {
                    0
                } else {
                    sum_col.value(r)
                };
                let avg = if avg_col.is_null(r) {
                    None
                } else {
                    Some(avg_col.value(r))
                };
                result.insert(k, (cnt, sum, avg));
            }
        }
    }
    Ok(result)
}

#[derive(Clone, Copy)]
struct SweepCase {
    name: &'static str,
    num_groups: usize,
    initial_rows_per_group: usize,
    delta_pairs_per_batch: usize,
    retract_share_pct: usize, // e.g. 10 (mostly updates) vs 50 (heavy retractions)
    micro_batches: usize,
}

async fn run_sweep_case(
    cl_schema: SchemaRef,
    base_schema: SchemaRef,
    case: SweepCase,
) -> Result<()> {
    let total_initial = case.num_groups * case.initial_rows_per_group;
    let mut agg = IncrementalChangelogAgg::try_new()?;
    let mut keys = Vec::with_capacity(total_initial);
    let mut vals = Vec::with_capacity(total_initial);
    let mut weights = Vec::with_capacity(total_initial);

    let group_names: Vec<String> = (0..case.num_groups).map(|g| format!("grp_{g}")).collect();

    for g in &group_names {
        for r in 0..case.initial_rows_per_group {
            keys.push(g.clone());
            vals.push(Some((r as i64) + 1));
            weights.push(1i8);
        }
    }

    let init_batch = RecordBatch::try_new(
        Arc::clone(&cl_schema),
        vec![
            Arc::new(StringArray::from(keys.clone())),
            Arc::new(Int64Array::from(vals.clone())),
            Arc::new(Int8Array::from(weights)),
        ],
    )?;
    agg.apply_batch(&init_batch)?;

    // Build delta batch (mix of -1/+1 updates and -1 retractions)
    let mut delta_keys: Vec<&str> = Vec::new();
    let mut delta_vals: Vec<Option<i64>> = Vec::new();
    let mut delta_weights: Vec<i8> = Vec::new();

    for i in 0..case.delta_pairs_per_batch {
        let g = group_names[i % case.num_groups].as_str();
        let is_retract_only =
            ((i * 100) / case.delta_pairs_per_batch.max(1)) < case.retract_share_pct;
        if is_retract_only {
            delta_keys.push(g);
            delta_vals.push(Some(1));
            delta_weights.push(-1i8);
        } else {
            delta_keys.push(g);
            delta_vals.push(Some(1));
            delta_weights.push(-1i8);
            delta_keys.push(g);
            delta_vals.push(Some(999));
            delta_weights.push(1i8);
        }
    }

    let delta_batch = RecordBatch::try_new(
        Arc::clone(&cl_schema),
        vec![
            Arc::new(StringArray::from(delta_keys)),
            Arc::new(Int64Array::from(delta_vals)),
            Arc::new(Int8Array::from(delta_weights)),
        ],
    )?;

    // 1. Incremental changelog updates across N micro-batches
    let t0 = Instant::now();
    for _ in 0..case.micro_batches {
        agg.apply_batch(&delta_batch)?;
    }
    let inc_elapsed = t0.elapsed();

    // 2. Materialize table once outside the timed loop so we measure GROUP BY
    //    recomputation rather than Vec::remove / StringArray / MemTable setup.
    let full_batch = build_grouped_batch(Arc::clone(&base_schema), keys, vals)?;
    let ctx = SessionContext::new();
    let table = MemTable::try_new(Arc::clone(&base_schema), vec![vec![full_batch]])?;
    ctx.register_table("t", Arc::new(table))?;
    let sql = "SELECT key, COUNT(val), SUM(val), AVG(val) FROM t GROUP BY key";
    let logical_plan = ctx.sql(sql).await?.logical_plan().clone();
    let state = ctx.state();
    let task_ctx = ctx.task_ctx();

    let t1 = Instant::now();
    for _ in 0..case.micro_batches {
        let phys_plan = state.create_physical_plan(&logical_plan).await?;
        let _ = datafusion::physical_plan::collect(phys_plan, Arc::clone(&task_ctx)).await?;
    }
    let df_recompute_elapsed = t1.elapsed();

    // Verify active group count after retracting 50% of groups completely
    let groups_to_clear = case.num_groups / 2;
    let mut clear_rows = Vec::new();
    for g_idx in 0..groups_to_clear {
        let g = group_names[g_idx].as_str();
        for r in 0..case.initial_rows_per_group {
            clear_rows.push((g, Some((r as i64) + 1), -1i8));
        }
    }
    if !clear_rows.is_empty() {
        let clear_batch = make_changelog_batch(Arc::clone(&cl_schema), &clear_rows)?;
        agg.apply_batch(&clear_batch)?;
    }

    let _ = agg.dirty.len();
    println!(
        "[{:<22}] groups={:<5} init={:<6} delta={:<3} retract%={:<2} | inc={:<10?} df_recompute={:<10?} | active={}/{}",
        case.name,
        case.num_groups,
        total_initial,
        delta_batch.num_rows(),
        case.retract_share_pct,
        inc_elapsed,
        df_recompute_elapsed,
        agg.active_group_count(),
        agg.group_values.len(),
    );

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = &IncrementalChangelogAgg::try_new()?.key_schema;
    let cl_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new("val", DataType::Int64, true),
        Field::new("weight", DataType::Int8, false),
    ]));
    let base_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new("val", DataType::Int64, true),
    ]));

    // 1. Correctness check against DataFusion SQL GROUP BY reference
    let mut agg = IncrementalChangelogAgg::try_new()?;
    let mut base_rows: HashMap<String, Vec<Option<i64>>> = HashMap::new();

    let steps: Vec<Vec<(&str, Option<i64>, i8)>> = vec![
        vec![("a", Some(10), 1), ("a", Some(20), 1), ("b", Some(5), 1)],
        vec![("a", Some(10), -1), ("a", Some(30), 1), ("b", Some(5), -1)],
        vec![("a", Some(20), -1), ("a", Some(30), -1), ("c", Some(100), 1)],
    ];

    for (step_idx, step_rows) in steps.iter().enumerate() {
        let batch = make_changelog_batch(Arc::clone(&cl_schema), step_rows)?;
        agg.apply_batch(&batch)?;

        for (k, v, w) in step_rows {
            let entry = base_rows.entry((*k).to_string()).or_default();
            if *w > 0 {
                entry.push(*v);
            } else if let Some(pos) = entry.iter().position(|x| x == v) {
                entry.remove(pos);
            }
        }

        let inc_snap = agg.snapshot_active()?;
        let sql_snap = run_batch_sql_reference(&base_rows).await?;
        assert_eq!(
            inc_snap, sql_snap,
            "Mismatch at step {step_idx}: incremental={inc_snap:?}, sql={sql_snap:?}"
        );
    }
    println!(
        "Correctness passed (active groups after retractions: {})",
        agg.active_group_count()
    );

    // 2. Sweep across group counts, delta sizes, and retraction shares
    let cases = [
        SweepCase {
            name: "small_delta_low_grp",
            num_groups: 100,
            initial_rows_per_group: 100, // 10k rows
            delta_pairs_per_batch: 50,
            retract_share_pct: 10,
            micro_batches: 50,
        },
        SweepCase {
            name: "small_delta_high_grp",
            num_groups: 5_000,
            initial_rows_per_group: 20, // 100k rows
            delta_pairs_per_batch: 250,
            retract_share_pct: 10,
            micro_batches: 50,
        },
        SweepCase {
            name: "heavy_retract_med_grp",
            num_groups: 1_000,
            initial_rows_per_group: 100, // 100k rows
            delta_pairs_per_batch: 500,
            retract_share_pct: 50,
            micro_batches: 50,
        },
    ];

    for case in cases {
        run_sweep_case(Arc::clone(&cl_schema), Arc::clone(&base_schema), case).await?;
    }

    Ok(())
}