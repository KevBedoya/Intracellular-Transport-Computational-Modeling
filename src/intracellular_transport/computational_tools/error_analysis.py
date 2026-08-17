from . import pd
import os


def compute_percent_diff(csv_path1: str,
                         csv_path2: str,
                         output_dir: str,
                         x_label: str = "X",
                         y1_label: str = "Y1",
                         y2_label: str = "Y2",
                         diff_label: str = "%Diff",
                         output_filename: str = "percent_diff.csv"):
    """
    Computes percent differences between second columns of two CSVs
    and writes result to a .csv file in a user-supplied output directory.

    Args:
        csv_path1 (str): Path to the first CSV (reference).
        csv_path2 (str): Path to the second CSV (comparison).
        output_dir (str): Directory to save the output file.
        x_label (str): Custom label for the independent variable.
        y1_label (str): Custom label for file1’s dependent variable.
        y2_label (str): Custom label for file2’s dependent variable.
        diff_label (str): Label for percent difference.
        output_filename (str): Optional output file name (must end in .csv).
    """

    # Validate file name
    if not output_filename.endswith(".csv"):
        raise ValueError("Output file name must end with .csv")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct full output file path
    output_path = os.path.join(output_dir, output_filename)

    # Read input files
    df1 = pd.read_csv(csv_path1)
    df2 = pd.read_csv(csv_path2)

    if df1.shape != df2.shape or df1.shape[1] < 2:
        raise ValueError("Input CSVs must have the same shape and at least two columns.")

    x = df1.iloc[:, 0]
    y1 = df1.iloc[:, 1]
    y2 = df2.iloc[:, 1]

    # Compute percent difference, handle division-by-zero as NA
    percent_diff = (abs(y1 - y2) / abs(y1)) * 100
    percent_diff.replace([float('inf'), -float('inf')], pd.NA, inplace=True)

    result_df = pd.DataFrame({
        x_label: x,
        y1_label: y1,
        y2_label: y2,
        diff_label: percent_diff
    })

    # Save result
    result_df.to_csv(output_path, index=False)
    print(f"Saved percent difference CSV to:\n{output_path}")
