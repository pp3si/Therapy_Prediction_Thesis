from project_functions import run_LSTM

#Best params in grid search:
'''	out_folder: N/A
	width: 3
	hidden_size: 30
	num_layers: 1
	dropout: 0.0
	batch_size: 16
	learning_rate: 0.001
	cv: N/A (we use 1 because we're not grid-searching)
	num_epochs: 1000'''

#IMPORTANT: Replace the below values with the grid search's best found hyperparameters!
run_LSTM([3], "PLACEHOLDER_FOLDER_3", 30, 1, 0.0, 16, 0.001, 1, 1000, datafile="OQ_lists_new.pkl")