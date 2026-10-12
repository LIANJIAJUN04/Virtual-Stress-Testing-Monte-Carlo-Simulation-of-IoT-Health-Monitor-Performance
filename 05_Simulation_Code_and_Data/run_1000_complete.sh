# Create complete run script

#!/bin/bash
echo "Starting 1000-run simulation..."
echo "This will take about 1-2 minutes"
echo "Start: $(date)"

# Run 5 batches of Controlled (100 runs each = 500 total)
for i in {0..4}; do
    echo -e "\n--- Controlled Batch $i ---"
    python3 run_batch_integrated.py Controlled $i
done

# Run 5 batches of Residential (100 runs each = 500 total)
for i in {0..4}; do
    echo -e "\n--- Residential Batch $i ---"
    python3 run_batch_integrated.py Residential $i
done

echo -e "\n1000-run simulation complete!"
echo "End: $(date)"
echo "Files created:"
ls -lh full_spec_results/*.parquet | wc -l
echo "Total files"


chmod +x run_1000_complete.sh

# Run it (will take 1-2 minutes)
./run_1000_complete.sh
