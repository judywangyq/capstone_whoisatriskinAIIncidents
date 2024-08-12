import tkinter as tk
from tkinter import ttk
import capstone_custom_model as cm  # Import custom model
import capstone_data_processing as dp  # Import data processing module

def predict_impact(text, time):
    # Use the data processing module to preprocess the input text
    processed_text = dp.clean_text(text)
    processed_text = dp.remove_stop_words(processed_text)
    
    # Transform the text using the same TF-IDF vectorizer and model as in training
    tfidf_transformed = dp.vectorize_text([processed_text], dp.X_train_tfidf)[0]
    
    # Predict using the custom model
    y_pred = cm.custom_clf.predict(tfidf_transformed)
    
    # Map prediction to the target labels
    demographics = ["Business", "Consumers", "General Public", "Government", "Minorities", "Workers"]
    result = demographics[y_pred[0]]
    
    # Placeholder keywords (replace with your model's output if available)
    keywords = ["privacy", "security", "data"]

    # Update the GUI
    output_label.config(text=f"The incident description is likely to affect: {result}")
    keywords_label.config(text=f"Top 3 Keywords for this Demographic: {', '.join(keywords)}")

# Create the main window
root = tk.Tk()
root.title("Incident Impact Predictor")

# Create and place widgets
text_label = ttk.Label(root, text="What's Happening:")
text_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")

text_input = tk.Text(root, width=50, height=10)
text_input.grid(row=1, column=0, padx=10, pady=10, columnspan=2)

time_label = ttk.Label(root, text="Time Frame:")
time_label.grid(row=2, column=0, padx=10, pady=10, sticky="w")

time_selection = ttk.Combobox(root, values=['Today', 'Last Week', 'Last Month', 'Last Year'])
time_selection.set('Last Week')
time_selection.grid(row=2, column=1, padx=10, pady=10)

submit_button = ttk.Button(root, text="Submit Incident", command=lambda: predict_impact(text_input.get("1.0", tk.END), time_selection.get()))
submit_button.grid(row=3, column=0, padx=10, pady=10, columnspan=2)

output_label = ttk.Label(root, text="The incident description is likely to affect: Consumers")
output_label.grid(row=4, column=0, padx=10, pady=10, columnspan=2)

keywords_label = ttk.Label(root, text="Top 3 Keywords for this Demographic: user, video, psychological")
keywords_label.grid(row=5, column=0, padx=10, pady=10, columnspan=2)

# Start the GUI event loop
root.mainloop()
