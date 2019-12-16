for run_id in 1 2 3 4 5
do
    python -u main.py 0 0 1 6 "meld_only" $run_id > "meld_only_${run_id}.out"
done