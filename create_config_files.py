width_options = [1,2,3,4,5,6,7,8,9,10,20]
out_folder = "PLACEHOLDER_FOLDER_1"
hidden_size_options = [30,40]
num_layers_options = [1,2]
dropout_options = [0.0, 0.2]
batch_size_options = [16]
learning_rate_options = [0.001, 0.005, 0.0005]
cross_validation_options = [3]
num_epochs_options = [1000]

from itertools import product

config_number = 1

for options_tuple in\
        product(width_options, hidden_size_options, num_layers_options, dropout_options, batch_size_options,
                learning_rate_options, cross_validation_options, num_epochs_options):
    with open(f"PLACEHOLDER_FOLDER_0/config_{config_number}.in", 'w') as file:
        file.write(out_folder+"\n")
        for param in options_tuple:
            file.write(str(param)+"\n") #This relies heaviliy on getting the order right
            #out_folder, width, hidden_size, num_layers, dropout, batch_size, learning_rate, cv, num_epochs
    
    config_number += 1