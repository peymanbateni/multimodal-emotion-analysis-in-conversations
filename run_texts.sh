for run_id in 1 2 3 4 5
do
    python -u main.py 1 0 0 7 "text_only" $run_id > "text_only_${run_id}.out"
done
