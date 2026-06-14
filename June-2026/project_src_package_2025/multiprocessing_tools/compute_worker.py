

def compute_and_send(computation_name, param_dict):
    from multiprocessing_tools import computation_router
    from system_configuration import file_paths as fp
    import os
    import json

    output_dir = fp.json_output
    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, f"result.json")

    try:
        result = computation_router.run_selected_computation(computation_name, param_dict)
        output = {"status": "ok", "result": result}
    except Exception as e:
        output = {"status": "error", "message": str(e)}

    with open(result_file, "w") as f:
        json.dump(output, f)
        # time.sleep(1)

