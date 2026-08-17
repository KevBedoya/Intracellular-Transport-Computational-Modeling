def format_windows_path(path: str) -> str:
    """
    Ensures a given file path is correctly formatted for Windows usage
    in a Python string, using double backslashes between directories.

    Args:
        path (str): The input file path.

    Returns:
        str: A Windows-compatible file path using double backslashes.
    """
    # Normalize to use forward slashes first to avoid mixed slashes
    normalized = path.replace('\\', '/')

    # Split the path into parts and rejoin with double backslashes
    parts = normalized.split('/')
    formatted_path = '\\\\'.join(parts)

    return formatted_path


# Example usage:
if __name__ == "__main__":
    # Modify this path to test various inputs
    input_path = "C:/Users/kevin/Documents/my_file.txt"
    windows_path = format_windows_path(input_path)
    print("Formatted Windows Path:", windows_path)
