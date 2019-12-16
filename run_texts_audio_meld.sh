for run_id in 1 2 3 4 5
do
    python -u main.py 1 0 1 3 "text_audio_meld" $run_id > "text_audio_meld_${run_id}.out"
done