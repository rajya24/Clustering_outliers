import pandas as pd
 
def remove_outliers_iqr(df, column_name):
    """
    Removes outliers from a DataFrame based on the IQR method for a specific column.
 
    Parameters:
    df : pd.DataFrame - The input dataframe.
    column_name : str - The name of the column for which outliers need to be removed.
 
    Returns:
    pd.DataFrame - DataFrame with outliers removed.
    """
 
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
 
    # Calculate Interquartile Range (IQR)
    IQR = Q3 - Q1
 
    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
 
    # Filter the DataFrame to remove rows that contain outliers
    filtered_df = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]
    return filtered_df
 
# Example usage:
# df_cleaned = remove_outliers_iqr(df, 'column_name')
