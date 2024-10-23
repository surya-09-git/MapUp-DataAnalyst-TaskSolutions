from typing import Dict, List
import pandas as pd
import re
from itertools import permutations
import numpy as np  # Import numpy for numerical calculations


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    reversed_list = []
    for i in range(0, len(lst), n):
        group = lst[i:i + n]
        # Manually reverse the group and extend the result
        for j in range(len(group) - 1, -1, -1):
            reversed_list.append(group[j])
    return reversed_list


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    length_dict = {}
    for string in lst:
        length = len(string)
        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(string)
    return length_dict


def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    """
    flat_dict = {}

    def flatten(current_dict, parent_key=''):
        for k, v in current_dict.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                flatten(v, new_key)
            else:
                flat_dict[new_key] = v

    flatten(nested_dict)
    return flat_dict


def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    """
    return [list(p) for p in set(permutations(nums))]


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    """
    date_patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b',  # dd-mm-yyyy
        r'\b\d{2}/\d{2}/\d{4}\b',  # mm/dd/yyyy
        r'\b\d{4}\.\d{2}\.\d{2}\b'  # yyyy.mm.dd
    ]

    matches = []
    for pattern in date_patterns:
        matches.extend(re.findall(pattern, text))
    return matches


def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    """
    # Assuming the polyline is a simple string format; you might need to decode it first.
    points = polyline_str.split(';')  # Example: split by semicolon
    latitudes = []
    longitudes = []
    distances = []

    for i, point in enumerate(points):
        lat, lon = map(float, point.split(','))
        latitudes.append(lat)
        longitudes.append(lon)
        if i > 0:
            # Calculate distance between points (using Euclidean distance)
            dist = np.sqrt((latitudes[i] - latitudes[i - 1]) ** 2 + (longitudes[i] - longitudes[i - 1]) ** 2)
            distances.append(dist)
        else:
            distances.append(0)  # No distance for the first point

    df = pd.DataFrame({
        'latitude': latitudes,
        'longitude': longitudes,
        'distance': distances
    })

    return df


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    """
    n = len(matrix)
    rotated_matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]

    for i in range(n):
        for j in range(n):
            rotated_matrix[i][j] *= (i + j)

    return rotated_matrix


def time_check(df: pd.DataFrame) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether 
    the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period.
    """
    completeness_series = df.groupby(['id', 'id_2']).apply(
        lambda group: (group['timestamp'].max() - group['timestamp'].min()).total_seconds() >= 604800
    )
    return completeness_series


# Test reverse_by_n_elements
print(reverse_by_n_elements([1, 2, 3, 4, 5, 6, 7, 8], 3))  # Output: [3, 2, 1, 6, 5, 4, 8, 7]

# Test group_by_length
print(group_by_length(["a", "bb", "ccc", "d", "ee"]))  # Output: {1: ['a', 'd'], 2: ['bb', 'ee'], 3: ['ccc']}

# Test flatten_dict
print(flatten_dict({'a': {'b': 1, 'c': 2}, 'd': 3}))  # Output: {'a.b': 1, 'a.c': 2, 'd': 3}

# Test unique_permutations
print(unique_permutations([1, 1, 2]))  # Output: [[1, 1, 2], [1, 2, 1], [2, 1, 1]]

# Test find_all_dates
print(find_all_dates("Today's date is 20-10-2024 and tomorrow will be 21/10/2024."))  # Output: ['20-10-2024', '21/10/2024']

# Test polyline_to_dataframe
print(polyline_to_dataframe("12.34,56.78;90.12,34.56"))  # Sample input

# Test rotate_and_multiply_matrix
print(rotate_and_multiply_matrix([[1, 2], [3, 4]]))  # Sample input

# Test time_check (make sure you have a proper DataFrame with 'timestamp' column)
# Example DataFrame
# df = pd.DataFrame({
#     'id': [1, 1, 1, 2, 2],
#     'id_2': [1, 1, 1, 1, 1],
#     'timestamp': pd.to_datetime(['2024-10-20 00:00:00', '2024-10-20 01:00:00', 
#                                   '2024-10-20 23:59:59', '2024-10-21 00:00:00', 
#                                   '2024-10-27 00:00:00'])
# })
# print(time_check(df))
