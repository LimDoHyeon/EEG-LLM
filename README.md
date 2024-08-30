<h2>Fine-tuned LLM for Electroencephalography(EEG) data classification</h2>
<p>This is LLM Fine tuning model that classifies four movements (left arm, right arm, tongue, foot) from EEG.</p>
<ul>
  <li>LLM performs its own classification operations based on EEG data.</li>
  <li>We trained gpt-4o model utilizing fine-tuning for better performance.</li>
</ul>
<img width="927" alt="figure1" src="https://github.com/user-attachments/assets/5e44d117-c761-4bcb-8ea0-a7a612ddc33a">

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
<p>For feature selection and extraction, <strong>Fisher Ratio</strong> is utilized.</p>
<table style="border-collapse: collapse; width: 100%; data-ke-align="alignLeft">
  <tbody>
        <tr>
            <td style="width: 50%;"><img width="499" alt="fr_label1" src="https://github.com/user-attachments/assets/d731b564-65e8-4f13-b28b-b6391eb267eb"></td>
            <td style="width: 50%;"><img width="501" alt="fr_label2" src="https://github.com/user-attachments/assets/0c621691-6dd3-4c89-9113-d0c7b0e3bf70"></td>
            <td style="width: 50%;"><img width="496" alt="fr_label3" src="https://github.com/user-attachments/assets/0479cd0c-5a48-4c4c-b763-0c06d253c81c"></td>
            <td style="width: 50%;"><img width="493" alt="fr_label4" src="https://github.com/user-attachments/assets/a825d381-3506-47b1-8c5b-82522fd770d3"></td>
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
  <li><strong>Cohen's Kappa Coefficient</strong></li>
</ul>
<p>(Metrics plot will be here)</p>
</p>
