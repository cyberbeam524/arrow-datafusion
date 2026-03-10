// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Benchmarks for row_number + unnest + group by memory usage

use arrow::array::{ArrayRef, Int32Array, ListArray, StringArray};
use arrow::datatypes::{DataType, Field, Int32Type, Schema};
use arrow::record_batch::RecordBatch;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use datafusion::prelude::*;
use std::sync::Arc;
use tokio::runtime::Runtime;

fn create_test_batch(num_rows: usize, list_size: usize) -> RecordBatch {
    let metadata: Vec<String> = (0..num_rows).map(|i| format!("row_{}", i)).collect();
    let metadata = StringArray::from(metadata);
    
    let mut values_a_builder = vec![];
    let mut values_b_builder = vec![];
    let mut values_c_builder = vec![];
    
    for i in 0..num_rows {
        let start = (i * list_size) as i32;
        let values: Vec<i32> = (start..start + list_size as i32).collect();
        values_a_builder.push(Some(values.clone()));
        values_b_builder.push(Some(values.clone()));
        values_c_builder.push(Some(values));
    }
    
    let values_a = ListArray::from_iter_primitive::<Int32Type, _, _>(values_a_builder);
    let values_b = ListArray::from_iter_primitive::<Int32Type, _, _>(values_b_builder);
    let values_c = ListArray::from_iter_primitive::<Int32Type, _, _>(values_c_builder);
    
    let schema = Arc::new(Schema::new(vec![
        Field::new("metadata", DataType::Utf8, false),
        Field::new("values_a", DataType::new_list(DataType::Int32, false), false),
        Field::new("values_b", DataType::new_list(DataType::Int32, false), false),
        Field::new("values_c", DataType::new_list(DataType::Int32, false), false),
    ]));
    
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(metadata) as ArrayRef,
            Arc::new(values_a) as ArrayRef,
            Arc::new(values_b) as ArrayRef,
            Arc::new(values_c) as ArrayRef,
        ],
    )
    .unwrap()
}

async fn run_unnest_group_by_query(
    ctx: &SessionContext,
    batch: RecordBatch,
) -> Vec<RecordBatch> {
    ctx.deregister_table("test_data").ok();
    ctx.register_batch("test_data", batch).unwrap();
    
    let sql = r#"
        WITH indexed AS (
            SELECT 
                ROW_NUMBER() OVER () as row_idx,
                metadata,
                values_a,
                values_b,
                values_c
            FROM test_data
        ),
        unnested AS (
            SELECT 
                row_idx,
                metadata,
                unnest(values_a) as val_a,
                unnest(values_b) as val_b,
                unnest(values_c) as val_c
            FROM indexed
        ),
        transformed AS (
            SELECT
                row_idx,
                metadata,
                val_a,
                val_b,
                val_c,
                CASE WHEN val_c > 100 THEN val_a * val_b ELSE val_a + val_b END AS val_d
            FROM unnested
        )
        SELECT
            row_idx,
            metadata,
            array_agg(val_a) AS values_a,
            array_agg(val_b) AS values_b,
            array_agg(val_c) AS values_c,
            array_agg(val_d) AS values_d
        FROM transformed
        GROUP BY row_idx, metadata
        ORDER BY row_idx
    "#;
    
    let df = ctx.sql(sql).await.unwrap();
    df.collect().await.unwrap()
}

fn unnest_memory_benchmark(c: &mut Criterion) {
    let runtime = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("unnest_memory");
    group.sample_size(10);
    
    // Test with different sizes to show memory scaling
    for (num_rows, list_size) in &[
        (10, 100),
        (100, 100),
        (1000, 100),
    ] {
        let batch = create_test_batch(*num_rows, *list_size);
        let total_intermediate_rows = num_rows * list_size;
        
        group.bench_with_input(
            BenchmarkId::new(
                "row_number_unnest_group",
                format!("{}rows_{}list_{}intermediate", num_rows, list_size, total_intermediate_rows),
            ),
            &batch,
            |b, batch| {
                b.iter(|| {
                    let ctx = SessionContext::new();
                    runtime.block_on(async {
                        black_box(run_unnest_group_by_query(&ctx, batch.clone()).await)
                    })
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(benches, unnest_memory_benchmark);
criterion_main!(benches);
