from transformers import pipeline

pipe = pipeline("text-classification", model="xaqren/sentiment_analysis")

# Test a clear positive and clear negative statement
print("Positive Test:", pipe("This operation is running perfectly and safely."))
print("Negative Test:", pipe("Critical failure and massive disaster reported."))