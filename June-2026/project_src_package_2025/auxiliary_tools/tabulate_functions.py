from . import datetime, pd, os


def produce_csv_from_xy(data_array, xlab, ylab, file_path, file_name):

    """

    Extracts numerical data from an array with dictionary elements of the form:
    {"W": {w_param}, "MFPT": {mfpt}} or {V: {v_param}, MFPT: {mfpt}},
    see solve_mfpt_multi_process() in project_src_package_2025/launch_functions/launch.py, for
    implementation details of this data structure.

    Numerical data is stored to a csv file which is then transferred to a specified directory.
    (In most cases data should be transferred to data_output/heatmaps)

    :param data_array: the array containing dictionary elements, {"W": {w_param}, "MFPT": {mfpt}} or {V: {v_param}, MFPT: {mfpt}}.
    :param xlab: (str) the label of the x coordinate (in most cases this should be W or V)
    :param ylab: (str) the label of the y coordinate  (in most cases this should be MFPT)
    :param file_path: (str) destination of the newly created csv file
    :param file_name: (str) name of the file
    :return: void

    Please keep in mind that file, path, or directory naming styles differ across systems, which may lead to errors.

    If separators are needed within file naming, please limit usage to underscores: example_of_naming.filetype.
    """

    x_values = []
    y_values = []
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    full_file_name = f'{file_name}_{current_time}.csv'

    for entry in data_array:
        for item in entry:
            if xlab in item:
                x_values.append(float(item.split(':')[1]))
            if ylab in item:
                y_values.append(float(item.split(':')[1]))

    df = pd.DataFrame({
        xlab: x_values,
        ylab: y_values
    })

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    df.to_csv(os.path.join(file_path, full_file_name), sep=',', index=False)


def create_directory(filepath, directory_name):
    """

    Creates a directory to store data to.

    :param filepath: (str) destination/path of the directory
    :param directory_name: (str) desired name of the directory
    :return: void
    """
    directory_path = os.path.join(filepath, directory_name)

    try:
        os.makedirs(directory_path, exist_ok=True)
        print(f"Directory '{directory_name}' created-successfully at {filepath}.")
        return directory_path
    except Exception as e:
        print(f"An error occurred while creating the directory: {e}")