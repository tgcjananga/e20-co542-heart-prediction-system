import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define paths
BASE_DIR = os.path.dirname(__file__)  # Get the directory of the script
DATA_PATH = os.path.join(BASE_DIR, "../data/heart_disease.csv")  
VISUALS_DIR = os.path.join(BASE_DIR, "static/visuals")

# Ensure the visuals directory exists
os.makedirs(VISUALS_DIR, exist_ok=True)

def generate_visualizations():
    # Check if dataset exists
    if not os.path.exists(DATA_PATH):
        return "Error: Dataset file not found!"

    # Load dataset
    df = pd.read_csv(DATA_PATH)
    
    # Drop duplicates
    df = df.drop_duplicates()

    # Create and save visualizations
    visualizations = []

    # Heatmap: Correlation Matrix
    plt.figure(figsize=(16, 6))
    sns.heatmap(df.corr(), annot=True)
    heatmap_path = os.path.join(VISUALS_DIR, "heatmap.png")
    plt.title("Correlation Heatmap of Dataset Features")
    plt.savefig(heatmap_path)
    plt.close()
    visualizations.append({
        'image': heatmap_path,
        'description': "This heatmap shows the correlation between the different features in the dataset. High correlation indicates strong relationships between features."
    })

    # Heart Disease Count: Distribution of Target Variable
    plt.figure()
    sns.countplot(x='target', data=df)
    target_dist_path = os.path.join(VISUALS_DIR, "target_distribution.png")
    plt.title("Distribution of Heart Disease (Target Variable)")
    plt.savefig(target_dist_path)
    plt.close()
    visualizations.append({
        'image': target_dist_path,
        'description': "This plot shows the distribution of the target variable. The target represents whether or not a person has heart disease (1 = disease, 0 = no disease)."
    })

    # Gender Distribution: Distribution of Gender
    plt.figure()
    sns.countplot(x='sex', data=df)
    gender_dist_path = os.path.join(VISUALS_DIR, "gender_distribution.png")
    plt.title("Gender Distribution in the Dataset")
    plt.savefig(gender_dist_path)
    plt.close()
    visualizations.append({
        'image': gender_dist_path,
        'description': "This plot shows the distribution of gender in the dataset. It indicates the number of male and female patients."
    })

    # Heart Disease by Gender: Disease Distribution by Gender
    plt.figure()
    sns.countplot(data=df, x='sex', hue='target')
    plt.xticks([0, 1], ['Female', 'Male'])
    plt.legend(labels=["No Disease", "Disease"])
    heart_gender_path = os.path.join(VISUALS_DIR, "heart_disease_by_gender.png")
    plt.title("Heart Disease Distribution by Gender")
    plt.savefig(heart_gender_path)
    plt.close()
    visualizations.append({
        'image': heart_gender_path,
        'description': "This plot shows the relationship between gender and heart disease. It breaks down the disease distribution by gender."
    })

    # Return descriptions and visualization paths
    return visualizations  # This returns the visualizations with their descriptions

if __name__ == "__main__":
    visualizations = generate_visualizations()  # Run to check output
    for vis in visualizations:
        print(f"Image Path: {vis['image']}")
        print(f"Description: {vis['description']}")
        print()
