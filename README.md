<h2>Fine-tuned LLM for Electroencephalography(EEG) data classification</h2>
<p>This is LLM Fine tuning model that classifies four movements (left arm, right arm, tongue, foot) from EEG.</p>
<ul>
  <li>LLM performs its own classification operations based on EEG data.</li>
  <li>We trained gpt-4o model utilizing fine-tuning for better performance.</li>
</ul>
<img width="927" alt="figure1" src="https://github.com/user-attachments/assets/9803f163-de78-4ad8-9341-41bc65a46a91">

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
    <li>performance measure: kappa-coefficient (we use Accuracy, F1-score, and ROC-AUC instead of this)</li>
  </ul>
<br>
<p><strong>Download</strong> : BBCI Competition III (https://www.bbci.de/competition/iii/download/index.html?agree=yes&submit=Submit)</p>

<br>
<h2>Features</h2>
<p>For feature selection and extraction, <strong>Fisher Ratio</strong> is utilized.</p>
<table style="border-collapse: collapse; width: 100%; data-ke-align="alignLeft">
  <tbody>
        <tr>
            <td style="width: 50%;"><img width="500" alt="fr_label1" src="https://github.com/user-attachments/assets/657684c1-7cf2-41eb-a10f-e4d1f0ca2159"></td>
            <td style="width: 50%;"><img width="500" alt="fr_label2" src="https://github.com/user-attachments/assets/baf30b29-2513-4078-a01d-ab3bf524ef17"></td>
            <td style="width: 50%;"><img width="500" alt="fr_label3" src="https://github.com/user-attachments/assets/f63b1310-513e-4335-b64e-6c1014964eb9"></td>
            <td style="width: 50%;"><img width="500" alt="fr_label4" src="https://github.com/user-attachments/assets/ac46f692-9e85-4d32-bffc-7f73ac26b44e"></td>
        </tr>
  </tbody>
</table>

<br>
<h2>Evaluation</h2>
<p>To compare fine-tuned LLM classifier's performance with traditional ML models, we additionally trained <strong>SVM</strong>, <strong>RF</strong> and <strong>MLP</strong> in the same data and same preprocessing method.</p>
<p>Performance metrics:
<ul>
  <li><strong>Accuracy</strong></li>
  <li><strong>F1 score</strong></li>
  <li><strong>ROC-AUC</strong></li>
</ul>
<p>(Metrics plot will be here)</p>
</p>
