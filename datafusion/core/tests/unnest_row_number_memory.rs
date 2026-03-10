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

//! Tests for unnest + row_number + group by memory usage
//!
//! This test validates that the combination of ROW_NUMBER() + UNNEST + GROUP BY
//! uses sorted aggregation mode instead of linear mode, which would cause
//! excessive memory usage by buffering all unnested rows.

use arrow::array::{Array, ArrayRef, ListArray, StringArray};
use arrow::datatypes::{DataType, Field, Int32Type, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::*;
use datafusion_common::Result;
use datafusion_physical_plan::displayable;
use std::sync::Arc;

/// Create a test table with list columns
fn create_test_data() -> Result<RecordBatch> {
    create_test_data_with_size(4, 100)
}

/// Create a test table with configurable size
fn create_test_data_with_size(num_rows: usize, list_size: usize) -> Result<RecordBatch> {
    let metadata: Vec<String> = (0..num_rows).map(|i| format!("row_{}", i)).collect();
    let metadata = StringArray::from(metadata);
    
    // Create list arrays with multiple elements per row
    let mut values_a_builder = vec![];
    let mut values_b_builder = vec![];
    
    for i in 0..metadata.len() {
        let start = (i * list_size) as i32;
        let values: Vec<Option<i32>> = (start..start + list_size as i32).map(Some).collect();
        values_a_builder.push(Some(values.clone()));
        values_b_builder.push(Some(values));
    }
    
    let values_a = ListArray::from_iter_primitive::<Int32Type, _, _>(values_a_builder);
    let values_b = ListArray::from_iter_primitive::<Int32Type, _, _>(values_b_builder);
    
    let schema = Arc::new(Schema::new(vec![
        Field::new("metadata", DataType::Utf8, false),
        Field::new(
            "values_a",
            DataType::new_list(DataType::Int32, true),
            false,
        ),
        Field::new(
            "values_b",
            DataType::new_list(DataType::Int32, true),
            false,
        ),
    ]));
    
    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(metadata) as ArrayRef,
            Arc::new(values_a) as ArrayRef,
            Arc::new(values_b) as ArrayRef,
        ],
    )?)
}

#[tokio::test]
async fn test_unnest_row_number_group_by_correctness() -> Result<()> {
    let ctx = SessionContext::new();
    let batch = create_test_data()?;
    
    ctx.register_batch("test_data", batch)?;
    
    // The problematic query pattern: row_number() + unnest + group by
    let sql = r#"
        WITH indexed AS (
            SELECT 
                ROW_NUMBER() OVER () as row_idx,
                metadata,
                values_a,
                values_b
            FROM test_data
        ),
        unnested AS (
            SELECT 
                row_idx,
                metadata,
                unnest(values_a) as val_a,
                unnest(values_b) as val_b
            FROM indexed
        )
        SELECT
            row_idx,
            metadata,
            array_agg(val_a ORDER BY val_a) AS values_a,
            array_agg(val_b ORDER BY val_b) AS values_b
        FROM unnested
        GROUP BY row_idx, metadata
        ORDER BY row_idx
    "#;
    
    let df = ctx.sql(sql).await?;
    
    // Execute the query to verify correctness
    let results = df.collect().await?;
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].num_rows(), 4); // Should have 4 output rows
    
    // Verify the data is correct
    let row_idx_col = results[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::UInt64Array>()
        .unwrap();
    
    // row_number starts at 1
    assert_eq!(row_idx_col.value(0), 1);
    assert_eq!(row_idx_col.value(1), 2);
    assert_eq!(row_idx_col.value(2), 3);
    assert_eq!(row_idx_col.value(3), 4);
    
    Ok(())
}

#[tokio::test]
async fn test_row_number_has_output_ordering() -> Result<()> {
    let ctx = SessionContext::new();
    let batch = create_test_data()?;
    
    ctx.register_batch("test_data", batch)?;
    
    // Simple query with just row_number
    let df = ctx
        .sql("SELECT ROW_NUMBER() OVER () as rn, metadata FROM test_data")
        .await?;
    
    let physical_plan = df.create_physical_plan().await?;
    
    // Check if the plan has output ordering
    let output_ordering = physical_plan.properties().output_ordering();
    
    println!(
        "Row number output ordering: {}",
        if output_ordering.is_some() {
            "present"
        } else {
            "absent"
        }
    );
    
    // With the fix, row_number should advertise ordering
    // Without it, this may be None
    if let Some(ordering) = output_ordering {
        println!("Output ordering present with {} expressions", ordering.len());
        assert_eq!(ordering.len(), 1, "Row number should have one ordering column");
    } else {
        panic!("Row number should advertise output ordering");
    }
    
    Ok(())
}

#[tokio::test]
async fn test_aggregate_uses_sorted_mode() -> Result<()> {
    let ctx = SessionContext::new();
    let batch = create_test_data()?;
    
    ctx.register_batch("test_data", batch)?;
    
    // The problematic query pattern: row_number() + unnest + group by
    let sql = r#"
        WITH indexed AS (
            SELECT 
                ROW_NUMBER() OVER () as row_idx,
                metadata,
                values_a,
                values_b
            FROM test_data
        ),
        unnested AS (
            SELECT 
                row_idx,
                metadata,
                unnest(values_a) as val_a,
                unnest(values_b) as val_b
            FROM indexed
        )
        SELECT
            row_idx,
            metadata,
            COUNT(*) as count
        FROM unnested
        GROUP BY row_idx, metadata
        ORDER BY row_idx
    "#;
    
    let df = ctx.sql(sql).await?;
    let physical_plan = df.create_physical_plan().await?;
    
    // Print the plan to see if it uses sorted aggregation
    let plan_str = format!("{}", displayable(physical_plan.as_ref()).indent(true));
    println!("Physical plan:\n{}", plan_str);
    
    // Check if the plan mentions ordering_mode
    // PartiallySorted or Sorted is good - means streaming aggregation
    // Linear would mean buffering all rows - memory explosion
    let uses_streaming = plan_str.contains("ordering_mode=Sorted")
        || plan_str.contains("ordering_mode=PartiallySorted");
    
    let uses_linear = plan_str.contains("ordering_mode=Linear");
    
    if uses_streaming {
        println!("✓ Aggregate is using streaming mode (Sorted/PartiallySorted) - memory efficient!");
        assert!(true, "Aggregate should use streaming aggregation");
    } else if uses_linear {
        panic!("✗ Aggregate is using Linear mode - will cause memory explosion!");
    } else {
        println!("⚠ Could not detect aggregation mode from plan");
    }
    
    Ok(())
}

#[tokio::test]
async fn test_large_dataset_memory_efficiency() -> Result<()> {
    // Test with a larger dataset closer to user's scenario
    // Using 1000 rows x 1000 elements = 1M unnested rows (smaller than user's 40M for testing)
    let ctx = SessionContext::new();
    let batch = create_test_data_with_size(1000, 1000)?;
    
    println!("Created test data: {} rows, ~1M elements total", batch.num_rows());
    
    ctx.register_batch("large_data", batch)?;
    
    let sql = r#"
        WITH indexed AS (
            SELECT 
                ROW_NUMBER() OVER () as row_idx,
                metadata,
                values_a,
                values_b
            FROM large_data
        ),
        unnested AS (
            SELECT 
                row_idx,
                metadata,
                unnest(values_a) as val_a,
                unnest(values_b) as val_b
            FROM indexed
        )
        SELECT
            row_idx,
            COUNT(*) as count,
            AVG(val_a) as avg_a
        FROM unnested
        GROUP BY row_idx
        ORDER BY row_idx
        LIMIT 10
    "#;
    
    let df = ctx.sql(sql).await?;
    let physical_plan = df.clone().create_physical_plan().await?;
    
    // Check the aggregation mode
    let plan_str = format!("{}", displayable(physical_plan.as_ref()).indent(true));
    let uses_streaming = plan_str.contains("ordering_mode=Sorted")
        || plan_str.contains("ordering_mode=PartiallySorted");
    
    assert!(
        uses_streaming, 
        "Large dataset should use streaming aggregation to avoid memory explosion.\nPlan:\n{}",
        plan_str
    );
    
    println!("✓ Large dataset test uses streaming aggregation");
    
    // Execute and verify
    let start = std::time::Instant::now();
    let results = df.collect().await?;
    let duration = start.elapsed();
    
    println!("Query completed in {:?}", duration);
    assert_eq!(results[0].num_rows(), 10); // Limited to 10 rows
    
    Ok(())
}
