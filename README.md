> This project was presented as a poster at the Cyberworlds 2024 international conference held in Yamanashi, Japan.<br><p>https://cyberworlds2024.yamanashi-u.jp/accepted-papers/</p>
<br>

<h2>Fine-tuned LLM for Electroencephalography(EEG) data classification</h2>
<p>This is LLM Fine tuning model that classifies four movements (left hand, right hand, tongue, foot) from EEG.</p>
<ul>
  <li>LLM performs its own classification operations based on EEG data.</li>
  <li>We trained gpt-4o model utilizing fine-tuning for better performance.</li>
</ul>
<img width="927" alt="figure1" src="https://github.com/user-attachments/assets/18f66f60-1b33-45af-9929-befd06ef0d55">

<br>
<h2>Requirements</h2>
<p><strong>Python>=3.8,   openai>=1.30.2,   mne>=1.6.1</strong><br>You can install all libraries entering the code: </p>
<blockquote>!pip install -r requirements.txt</blockquote>
<br>
<h2>Data</h2>
<p><strong>Data description</strong> : https://www.bbci.de/competition/iii/desc_IIIa.pdf</p>
  <ul>
    <li>cued motor imagery (multi-class) with 4 classes (left hand, right hand, foot, tongue) three subjects (ranging from quite good to fair performance)</li>
    <li>EEG, 60 channels, 60 trials per class</li>
    <li>performance measure: kappa-coefficient</li>
  </ul>
<br>
<p><strong>Download</strong> : BBCI Competition III (https://www.bbci.de/competition/iii/download/index.html?agree=yes&submit=Submit)</p>

<br>
<h2>Features</h2>
<p>1) <strong>Power spectral density (PSD)</strong> is computed in 2Hz steps from 4Hz to 36Hz.</p>
<p>For feature selection, <strong>Fisher Ratio</strong> is utilized.</p>
<table style="border-collapse: collapse; width: 100%; data-ke-align="alignLeft">
  <tbody>
        <tr>
            <td style="width: 50%;"><img width="500" alt="fr_label1" src="https://github.com/user-attachments/assets/619a317b-f0d0-4600-9fe7-14da35aec61a"></td>
            <td style="width: 50%;"><img width="500" alt="fr_label2" src="https://github.com/user-attachments/assets/ccb4fc57-7db1-4902-b231-6400043e33ef"></td>
            <td style="width: 50%;"><img width="500" alt="fr_label3" src="https://github.com/user-attachments/assets/9bc6ef09-93ac-4f9a-82c7-c173cacb798e"></td>
            <td style="width: 50%;"><img width="500" alt="fr_label4" src="https://github.com/user-attachments/assets/528634b7-558e-4704-97b2-38dd7b427eea"></td>
        </tr>
  </tbody>
</table>
<p>2) <strong>Common spatial pattern (CSP)</strong> is used to extract spatial features that maximize discriminability between classes.</p>

<br>
<h2>Evaluation</h2>
<p>To compare fine-tuned LLM classifier's performance with traditional ML models, we additionally trained <strong>SVM</strong>, <strong>RF</strong> and <strong>MLP</strong> in the same data and same preprocessing method.</p>
<p>Performance metrics: <strong>Accuracy, F1 score, ROC-AUC</strong></p>
<img width=60% src="https://github.com/user-attachments/assets/b7c0047f-2f4f-44f4-8d0f-796bc7e5ef3c">
<ul>
  <li>Although the performance of the GPT-4o-based supervised learning model slightly lagged behind traditional machine learning models, this project is significant in demonstrating the potential of utilizing LLMs as supervised learning models.</li>
  <li>Furthermore, it highlights the expectation that as LLM performance continues to improve, the capabilities of LLM-based supervised learning models will also advance.</li>
</ul>
