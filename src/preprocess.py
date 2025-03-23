# src/preprocess.py

import pandas as pd


def load_and_clean_data(data_path):
    # Load the data from Excel
    try:
        releases_df = pd.read_excel(data_path, sheet_name='Releases 2000-2020', engine='openpyxl')
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    # Extract only the relevant contaminants
    contaminants_of_interest = [
        'Sulphur dioxide',
        'Nitrogen oxides (expressed as nitrogen dioxide)',
        'Volatile Organic Compounds (VOCs)',
        'PM10 - Particulate Matter <= 10 Micrometers',
        'PM2.5 - Particulate Matter <= 2.5 Micrometers',
        'Carbon monoxide'
    ]
    releases_df = releases_df[
        releases_df['Substance Name (English) / Nom de substance (Anglais)'].isin(contaminants_of_interest)]

    # Feature engineering for releases
    releases_df['Air_Releases'] = releases_df[['Release to Air - Fugitive', 'Release to Air - Other Non-Point ',
                                               'Release to Air - Spills ',
                                               'Release to Air - Storage / Handling ']].sum(axis=1)
    releases_df['Land_Releases'] = releases_df[['Releases to Land - Leaks', 'Releases to Land - Other ',
                                                'Releases to Land - Spills ']].sum(axis=1)
    releases_df['Water_Releases'] = releases_df[['Releases to Water Bodies - Direct Discharges ',
                                                 'Releases to Water Bodies - Leaks',
                                                 'Releases to Water Bodies - Spills ']].sum(axis=1)

    releases_df['Total_Releases'] = releases_df[
        ['Air_Releases', 'Land_Releases', 'Water_Releases', 'Sum of release to all media (<1tonne)']].sum(axis=1)

    # Drop unnecessary columns
    releases_df.drop([
    'Release to Air - Fugitive', 'Release to Air - Other Non-Point ', 'Release to Air - Road dust ',
    'Release to Air - Spills ', 'Release to Air - Stack / Point ', 'Release to Air - Storage / Handling ',
    'Releases to Land - Leaks', 'Releases to Land - Other ', 'Releases to Land - Spills ',
    'Releases to Water Bodies - Direct Discharges ', 'Releases to Water Bodies - Leaks',
    'Releases to Water Bodies - Spills ', 'Sum of release to all media (<1tonne)',
], axis=1, inplace=True, errors='ignore')  # Add errors='ignore' to prevent errors if columns are not found


    # Convert 'Number of employees' to numeric and handle missing values
    releases_df['Number of employees'] = pd.to_numeric(releases_df['Number of employees'], errors='coerce')
    releases_df['Number of employees'].fillna(releases_df['Number of employees'].mean(), inplace=True)

    # Select relevant columns
    releases_filtered = releases_df[['Reporting_Year / Année', 'Substance Name (English) / Nom de substance (Anglais)',
                                    'Number of employees',
                                    'Air_Releases', 'Land_Releases', 'Water_Releases',
                                    'Total_Releases']]
    # Melt the DataFrame to long format
    releases_long = releases_filtered.melt(
        id_vars=['Reporting_Year / Année', 'Substance Name (English) / Nom de substance (Anglais)',
                 'Number of employees'],
        var_name='Release_Type',
        value_name='Release'
    )
    # Convert the year to datetime format
    releases_long['Reporting_Year / Année'] = pd.to_datetime(releases_long['Reporting_Year / Année'], format='%Y')

    # Create multiple lag features (previous 1, 2, and 3 years)
    releases_long['Lag_1'] = releases_long.groupby(['Substance Name (English) / Nom de substance (Anglais)', 'Release_Type', 'Number of employees'])['Release'].shift(1)
    releases_long['Lag_2'] = releases_long.groupby(['Substance Name (English) / Nom de substance (Anglais)', 'Release_Type', 'Number of employees'])['Release'].shift(2)
    releases_long['Lag_3'] = releases_long.groupby(['Substance Name (English) / Nom de substance (Anglais)', 'Release_Type', 'Number of employees'])['Release'].shift(3)

    # Drop rows where any of the lag values are missing (optional)
    releases_long.dropna(subset=['Lag_1', 'Lag_2', 'Lag_3'], inplace=True)

    # Define the number of bins (3 bins: Low, Medium, High)
    num_bins = 3

    # Group by 'Substance Name', 'Reporting Year', 'Number of Employees'
    releases_long['Release_Category'] = releases_long.groupby(
        ['Substance Name (English) / Nom de substance (Anglais)',
         'Reporting_Year / Année',
         'Number of employees',
         ]
    )['Release'].transform(
        lambda x: pd.qcut(x.rank(method='first'), num_bins, labels=['Low', 'Medium', 'High'])
    )
    # Create a dictionary to store each substance's data separately
    substance_data = {}

    # Filter the dataset for each substance and store it in the dictionary
    for substance in contaminants_of_interest:
        substance_data[substance] = releases_long[releases_long['Substance Name (English) / Nom de substance (Anglais)'] == substance]
        print(f"Substance: {substance}")
        print(substance_data[substance].head())


# Add the function call at the end of the script
if __name__ == "__main__":
    data_path = '../data/raw/mydata.xlsx'  # Update this path if necessary
    load_and_clean_data(data_path)
