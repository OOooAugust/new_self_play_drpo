# run_all.sh
#!/bin/bash

echo "Start training A"
python /home/kyle/Documents/lab/drpo/DRPO_scripts/hh/sft_CompletionOnly.py > SFT_CompletionOnly.log 2>&1


echo "Start training B"
python /home/kyle/Documents/lab/drpo/DRPO_scripts/hh/reaward_model.py > reaward_model.log 2>&1
