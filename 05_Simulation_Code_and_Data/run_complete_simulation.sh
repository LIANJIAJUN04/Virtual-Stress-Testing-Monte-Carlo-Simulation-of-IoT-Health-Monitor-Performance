#!/bin/bash
echo "=== STARTING COMPLETE 1000-RUN SIMULATION ==="
echo "Start time: $(date)"

# Run Controlled environment batches (5 batches)
for batch in {0..4}; do
    echo -e "\n=== Batch $((batch+1))/10: Controlled $batch ==="
    python3 run_batch_integrated.py Controlled $batch
done

# Run Residential environment batches (5 batches)
for batch in {0..4}; do
    echo -e "\n=== Batch $((batch+6))/10: Residential $batch ==="
    python3 run_batch_integrated.py Residential $batch
done

echo -e "\n=== SIMULATION COMPLETE ==="
echo "End time: $(date)"
echo "Files created:"
ls -lh full_spec_results/
