#!/bin/bash

# Step 1: Convert GloVe embeddings to pickle format
python -m utils.pickle_glove --input glove/glove.840B.300d.txt --output pt/pt

# Step 2: Preprocess the training data
python -m BlindGRU.preprocess --input_dir ./dataset --output_dir preprocess/BlindGRU

# Step 3: Train the model
python -m BlindGRU.train --input_dir preprocess/BlindGRU --save_dir save_dir/BlindGRU --glove_pt pt/pt

# Step 4: Predict answers for the test set
python -m BlindGRU.predict --input_dir preprocess/BlindGRU --save_dir save_dir/BlindGRU

# Execute the script using:
# ./run_BlindGRU.sh