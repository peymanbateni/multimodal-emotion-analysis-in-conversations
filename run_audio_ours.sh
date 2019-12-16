for run_id in 1 2 3 4 5
do
    python -u main.py 0 1 0 3 "audio_our_only" $run_id > "audio_our_only_${run_id}.out"
done