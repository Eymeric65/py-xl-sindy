import json
import glob
import numpy as np
import pandas as pd

# Get list of all JSON files in the "result" folder
json_files = glob.glob("result/*.json")

# List to hold each JSON file's scalar data (one record per file)
records = []
# List to hold each file's solution_norm vector (as a NumPy array)
solution_norms = []

id=0

ideal_solutions_norm=[]

for filename in json_files:
    with open(filename, "r") as f:
        data = json.load(f)
    
    # Extract scalar values from the JSON. Many numeric values are stored as strings,
    # so we convert them as needed.
    experiment_folder = data["input"]["experiment_folder"]
    exploration_vol = float(data["result"]["exploration_volumes"])
    rmse_model = float(data["result"]["RMSE_model"])
    rmse_acceleration = float(data["result"]["RMSE_acceleration"])
    sparsity_diff = float(data["result"]["sparsity_difference"])
    sparsity_diff_perc = float(data["result"]["sparsity_difference_percentage"])
    max_time = float(data["input"]["max_time"])
    forces_period = float(data["input"]["forces_period"])
    forces_period_shift = float(data["input"]["forces_period_shift"])
    coordinate_number = int(data["environment"]["coordinate_number"])
    catalog_len = int(data["environment"]["catalog_len"])

    forces_input = data["input"]["forces_scale_vector"]
    forces_input = np.array(forces_input,dtype=float)

    max_forces = np.sum(forces_input)
    
    # Build a record (dictionary) of scalar inputs for this file
    record = {
        "filename": filename,
        "experiment_folder": experiment_folder,
        "exploration_volumes": exploration_vol,
        "RMSE_model": rmse_model,
        "RMSE_acceleration": rmse_acceleration,
        "sparsity_difference": sparsity_diff,
        "sparsity_difference_percentage": sparsity_diff_perc,
        "max_time": max_time,
        "forces_period": forces_period,
        "forces_period_shift": forces_period_shift,
        "coordinate_number": coordinate_number,
        "catalog_len": catalog_len,
        "id":id,
        "max_forces":max_forces
    }
    records.append(record)
    id+=1
    
    # Convert the solution_norm_nn string (which looks like a NumPy array) to an actual array
    sol_norm_str = data["result"]["solution_norm_nn"]
    sol_norm_array = np.fromstring(sol_norm_str.strip("[]"), sep=" ")

    ideal_solutions_norm_str = data["result"]["ideal_solution_norm_nn"]
    ideal_solutions_norm_sarray = np.fromstring(ideal_solutions_norm_str.strip("[]"), sep=" ")

    solution_norms.append(sol_norm_array[ideal_solutions_norm_sarray!=0]) # Keep only non null term
    ideal_solutions_norm.append(ideal_solutions_norm_sarray[ideal_solutions_norm_sarray!=0])

# Create a pandas DataFrame from the records
df = pd.DataFrame(records)