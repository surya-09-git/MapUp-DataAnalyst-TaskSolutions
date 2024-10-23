import pandas as pd

def calculate_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    distance_matrix = df.pivot(index='id_start', columns='id_end', values='distance').fillna(0)
    return distance_matrix

def unroll_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    unrolled_df = df.stack().reset_index(name='distance')
    unrolled_df.columns = ['id_start', 'id_end', 'distance']
    return unrolled_df

def find_ids_within_ten_percentage_threshold(df: pd.DataFrame, reference_id: int) -> pd.DataFrame:
    avg_distance = df.groupby('id_start')['distance'].mean().reset_index()
    
    # Check if reference_id exists in avg_distance
    if reference_id not in avg_distance['id_start'].values:
        print(f"Warning: Reference ID {reference_id} not found in average distance data.")
        return pd.DataFrame(columns=['id_start', 'distance'])  # Return empty DataFrame with proper columns

    reference_distance = avg_distance.loc[avg_distance['id_start'] == reference_id, 'distance'].values[0]
    threshold = 0.10 * reference_distance
    valid_ids = avg_distance[(avg_distance['distance'] >= (reference_distance - threshold)) & 
                             (avg_distance['distance'] <= (reference_distance + threshold))]
    
    return valid_ids

def calculate_toll_rate(df: pd.DataFrame) -> pd.DataFrame:
    toll_rates = {
        'car': 0.10,
        'truck': 0.20,
        'bus': 0.15
    }

    if 'vehicle_type' in df.columns:
        df['toll_rate'] = df['vehicle_type'].map(toll_rates)
        df['toll_cost'] = df['distance'] * df['toll_rate']
    else:
        df['toll_cost'] = df['distance'] * 0.10  # Default rate if vehicle_type is missing

    return df[['id_start', 'id_end', 'toll_cost']]

def calculate_time_based_toll_rates(df: pd.DataFrame) -> pd.DataFrame:
    time_based_rates = {
        '00:00-06:00': 0.05,
        '06:00-12:00': 0.10,
        '12:00-18:00': 0.15,
        '18:00-24:00': 0.20
    }

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['time'] = df['timestamp'].dt.strftime('%H:%M')
        
        for time_interval, rate in time_based_rates.items():
            start_time, end_time = time_interval.split('-')
            df.loc[(df['time'] >= start_time) & (df['time'] < end_time), 'time_toll_cost'] = df['distance'] * rate
        return df[['id_start', 'id_end', 'time_toll_cost']].fillna(0)
    else:
        df['time_toll_cost'] = df['distance'] * 0.10  # Default rate if timestamp is missing
        return df[['id_start', 'id_end', 'time_toll_cost']]

# Main execution block
if __name__ == "__main__":
    file_path = 'datasets/dataset-2.csv'
    df = pd.read_csv(file_path)

    print("Loaded DataFrame:")
    print(df.head())
    print("Columns in DataFrame:")
    print(df.columns)

    required_columns = ['id_start', 'id_end', 'distance', 'vehicle_type', 'timestamp']
    if not all(col in df.columns for col in required_columns):
        print(f"Warning: Input DataFrame is missing some columns. Found columns: {df.columns.tolist()}")

    distance_matrix = calculate_distance_matrix(df)
    print("Distance Matrix:")
    print(distance_matrix)

    unrolled_df = unroll_distance_matrix(distance_matrix)
    print("\nUnrolled DataFrame:")
    print(unrolled_df)

    reference_id = 1  # Make sure this ID exists in the unrolled DataFrame
    ids_within_threshold = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id)
    print(f"\nIDs within 10% of the average distance of reference ID {reference_id}:")
    print(ids_within_threshold)

    toll_rates_df = calculate_toll_rate(unrolled_df)
    print("\nToll Rates DataFrame:")
    print(toll_rates_df)

    time_based_toll_df = calculate_time_based_toll_rates(unrolled_df)
    print("\nTime-Based Toll Rates DataFrame:")
    print(time_based_toll_df)
