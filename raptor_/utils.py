


def get_value_by_key(list_of_dicts, search_key):
    """
    Get value for a specific key from a list of dictionaries with unique keys.

    Args:
        list_of_dicts (list): List containing dictionaries with unique keys
        search_key (str): Key to search for

    Returns:
        Any: Value corresponding to the search_key

    Example:
        >>> data = [{"name": "Alice"}, {"age": 25}, {"city": "New York"}]
        >>> get_value_by_key(data, "name")
        'Alice'
    """
    try:
        # Verify input is a list
        if not isinstance(list_of_dicts, list):
            raise ValueError("Input must be a list of dictionaries")

        # Look for the key in each dictionary
        for d in list_of_dicts:
            if search_key in d:
                return d[search_key]

        return None

    except Exception as e:
        raise Exception(f"An error occurred: {str(e)}")