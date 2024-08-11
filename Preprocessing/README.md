<h2>Preprocessing</h2>
<p>The files in this folder are scripts designed to perform feature extraction on scaled and Laplacian-filtered CSV data, and then save the results in JSON and JSONL formats.</p>
<ul>
  <li><strong>feature_extraction.py</strong>: This script calculates the Power Spectral Density (PSD) based on the input data and selected columns, and uses it to compute the <strong>alpha:delta power ratio</strong>, <strong>theta:alpha power ratio</strong>, and <strong>delta:theta power ratio.</strong></li>
  <li><strong>csv_to_json_4o.py</strong>: This script contains a function that transforms GPT's role, extracted features, and label values into a JSON format suitable for training GPT-4o (referencing the extract_features function from feature_extraction.py).</li>
  <li><strong>csv_to_json_davinci.py</strong>: This script includes a function that transforms GPT's role, extracted features, and label values into a JSON format suitable for training davinci-002 (referencing the extract_features function from feature_extraction.py).</li>
  <li><strong>preprocessing.py</strong>: This script provides a pipeline for loading data, extracting features, and saving the processed data in JSON and JSONL formats (referencing load_eeg_data from feature_extraction.py, csv_to_json from csv_to_json_4o.py, and json_to_jsonl).</li>
</ul>

<img width=100% alt="preprocessing" src="https://github.com/user-attachments/assets/1f334d41-e110-4053-8dcf-7c6693431b7d">
