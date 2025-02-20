import argparse
from project_functions import run_LSTM


parser = argparse.ArgumentParser()
parser.add_argument("--infile", type=str)
parser.add_argument("--datafile", type=str, default="OQ_lists_new.pkl")

args = parser.parse_args()

infile = args.infile
datafile = args.datafile

#out_folder, width, hidden_size, num_layers, dropout, batch_size, learning_rate, cv, num_epochs: Matches create_config_files
types_list = [str, int, int, int, float, int, float, int, int]
filepath = "/PLACEHOLDER_FOLDER_0/"+infile
print(filepath)

#Will look from its local folder to a subfolder called config_files for the given filename
with open(filepath, 'r') as file:
    lines = file.readlines()
    args_list = [types_list[i](line.strip()) for i, line in enumerate(lines)]
    out_folder, width, hidden_size, num_layers, dropout, batch_size, learning_rate, cv, num_epochs = args_list

names_list = ["out_folder", "width", "hidden_size", "num_layers", "dropout", "batch_size", "learning_rate", "cv", "num_epochs"]
for i, name in enumerate(names_list):
    print(f"{name} = {args_list[i]}")

#Separate folder into width and then config file (minus .in)
final_out_folder = out_folder+f"/window_{width}/{infile[:-3]}"

run_LSTM(widths=[width], out_folder=final_out_folder, hidden_size=hidden_size,
         num_layers=num_layers, dropout=dropout, batch_size=batch_size,
         lr=learning_rate, cv=cv, num_epochs=num_epochs, datafile=datafile)