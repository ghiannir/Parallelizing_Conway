import pandas as pd


def clean_data(path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(path + ".csv",
                     usecols=[1, 2, 3],
                     names=["iterations", "dimensions", "time"],
                     dtype={"iterations": int, "dimensions": int, "time": float})

    print(df)
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates(subset=["iterations", "dimensions"], keep="first")

    df_cleaned = df_cleaned[df['iterations'] <= 500]
    df_cleaned = df_cleaned[df['dimensions'] <= 1000]
    df_cleaned = df_cleaned[df['time'] > 0]

    # Write the cleaned data back to the file
    df_cleaned.to_csv(path + "_cleaned.csv", index=False)

    print("Duplicates removed and file rewritten.")


if __name__ == "__main__":
    file_paths = ["stats_cuda", "stats_sequential", "stats_openmp"]
    for file_path in file_paths:
        clean_data(file_path)
