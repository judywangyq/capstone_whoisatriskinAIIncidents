import tkinter as tk
from tkinter import ttk
import capstone_data_processing as dp
from capstone_custom_model import CustomClassifier, build_custom_model

# Build and train the custom model
models, class_model_map = build_custom_model()
custom_clf = CustomClassifier(models=models, class_model_map=class_model_map)
custom_clf.fit(dp.X_train_tfidf_resampled, dp.y_train_resampled)

def predict_impact(text, time):
    # Preprocess the input text
    processed_text = dp.clean_text(text)
    processed_text = dp.remove_stop_words(processed_text)
    
    # Ensure the processed text is passed as a list of strings
    tfidf = dp.vectorize_text([processed_text], dp.X_train_tfidf)
    
    # Predict using the custom model
    y_pred = custom_clf.predict(tfidf)
    
    # Map prediction to the target labels
    demographics = ["Business", "Consumers", "General Public", "Government", "Minorities", "Workers"]
    result = demographics[y_pred[0]]
    
    # Placeholder keywords
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
