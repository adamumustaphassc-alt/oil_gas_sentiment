import os
import pandas as pd
from transformers import pipeline

def main():
    print("Initializing environment...")
    # Get absolute paths relative to this script
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, "data", "raw_reports.csv")
    output_path = os.path.join(base_dir, "data", "processed_results.csv")

    # 1. Load the mock dataset
    if not os.path.exists(input_path):
        print(f"Error: Could not find input file at {input_path}")
        return
        
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)

    # 2. Initialize the Hugging Face sentiment pipeline
    print("Loading transformer model 'xaqren/sentiment_analysis'...")
    # This automatically uses GPU if available, otherwise defaults to CPU
    pipe = pipeline("text-classification", model="xaqren/sentiment_analysis")

    # 3. Define mapping for model labels
    label_mapping = {
        "LABEL_0": "Negative / Risk",
        "LABEL_2": "Positive / Clear"
    }

    # 4. Process text rows
    print("Running model inference on technical reports...")
    sentiments = []
    confidence_scores = []
    clean_labels = []

    for text in df["Technical_Report"]:
        # Safe string conversion handling
        prediction = pipe(str(text))[0]
        raw_label = prediction['label']
        score = prediction['score']
        
        sentiments.append(raw_label)
        confidence_scores.append(round(score, 4))
        clean_labels.append(label_mapping.get(raw_label, "Unknown"))

    # 5. Append results back to DataFrame columns
    df["Transformer_Sentiment"] = sentiments
    df["Confidence_Score"] = confidence_scores
    df["Sentiment_Clean"] = clean_labels

    # 6. Save final outputs
    df.to_csv(output_path, index=False)
    print(f"\nSuccess! Final predictions saved to: {output_path}")
    
    # Preview results in console
    print("\n--- Previewing Output Data ---")
    print(df[["Facility", "Sentiment_Clean", "Confidence_Score"]])

if __name__ == "__main__":
    main()