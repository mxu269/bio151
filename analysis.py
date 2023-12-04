import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# Path to the CSV file
avg_file = 'Combined Sleep Data Fa23 average.csv'
# Function to load the CSV file into a pandas DataFrame
def load_csv_to_dataframe(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        return str(e)

def remove_outliers(df, column_indices):
    try:
        # Convert column indices to actual column names
        column_names = [df.columns[i] for i in column_indices]

        # Initialize filtered DataFrame
        filtered_df = df.copy()

        # Loop through each column and remove outliers
        for column in column_names:
            q1, q3 = filtered_df[column].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            filtered_df = filtered_df[filtered_df[column].between(lower_bound, upper_bound)]

        return filtered_df
    except Exception as e:
        return str(e)
    

def run_anova_Q4(df, idp_idx, dep_idx, xlabel, ylabel, title):
    # Define column indices
    independent_idx, dependent_idx = idp_idx, dep_idx

    # Clean the DataFrame
    cleaned_df = remove_outliers(df, [independent_idx, dependent_idx])

    # Calculate quartile values for caffeine consumption
    Q1, Q2, Q3 = cleaned_df[df.columns[independent_idx]].quantile([0.25, 0.5, 0.75])

    # Define quartile ranges with formatted numbers
    quartile_ranges = [
        f'x < {Q1:.2f}',
        f'{Q1:.2f} ≤ x < {Q2:.2f}',
        f'{Q2:.2f} ≤ x < {Q3:.2f}',
        f'x ≥ {Q3:.2f}'
    ]

    # Assign each row to a quartile group
    cleaned_df['Quartile Group'] = pd.cut(cleaned_df[df.columns[independent_idx]], bins=[-float('inf'), Q1, Q2, Q3, float('inf')], labels=quartile_ranges, include_lowest=True)

    # Group by quartile
    groups = cleaned_df.groupby('Quartile Group')

    # Perform ANOVA test
    anova_result = stats.f_oneway(*[group[df.columns[dependent_idx]] for name, group in groups])

    # Print the result of ANOVA test
    print("ANOVA Test Result:", anova_result)

    # Create a bar graph with error bars and labels
    plt.figure(figsize=(10, 6))

    # Calculate mean and standard deviation
    means = groups[df.columns[dependent_idx]].mean()
    std_devs = groups[df.columns[dependent_idx]].std()

    # Specific colors for each bar
    colors = ['#F7DFD3', '#E2C3C8', '#AFAFC7', '#5F7DAF']  # Light Blue, Soft Pink, Mint Green, Lavender

    # Plot bars with error bars
    bars = plt.bar(range(4), means, yerr=std_devs, color=colors, capsize=5)

    # Add mean and std labels to each bar
    for bar, mean, std in zip(bars, means, std_devs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), 
                 f'Mean: {mean:.2f}\nSTD: {std:.2f}', 
                 ha='center', va='bottom')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(range(4), quartile_ranges, rotation='horizontal')

    plt.savefig(title+".png")

def main():
    # Test the function to load the CSV file
    avg_df = load_csv_to_dataframe(avg_file)
    
    run_anova_Q4(avg_df, 12, 4, "Caffeine Consumption (mg)", "Weekly Average Sleep Duration (min)", "Caffeine Consumption vs. Weekly Average Sleep Duration")
    run_anova_Q4(avg_df, 4, 34, "Weekly Average Sleep Duration (min)", "Memory Test A Score - penalized (sec)", "Weekly Average Sleep Duration vs. Memory Test A Score - penalized")
    run_anova_Q4(avg_df, 4, 35, "Weekly Average Sleep Duration (min)", "Memory Test B Score - penalized (sec)", "Weekly Average Sleep Duration vs. Memory Test B Score - penalized")
    run_anova_Q4(avg_df, 4, 15, "Weekly Average Sleep Duration (min)", "Mean Reaction Time (ms)", "Weekly Average Sleep Duration vs. Mean Reaction Time")
    run_anova_Q4(avg_df, 4, 10, "Weekly Average Sleep Duration (min)", "24Hr Stress Level (0-10)", "Weekly Average Sleep Duration vs. 24Hr Stress Level")


if __name__ == "__main__":
    main()