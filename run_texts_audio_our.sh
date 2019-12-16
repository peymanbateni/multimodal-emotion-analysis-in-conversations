for run_id in 1 2 3 4 5
do
    python -u main.py 1 1 0 5 "text_audio_our" $run_id > "text_audio_our_${run_id}.out"
done