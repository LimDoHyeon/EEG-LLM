<h2>Fine-tuned LLM for Electroencephalography(EEG) data classification</h2>
<p>This is LLM Fine tuning model that classifies four movements (left arm, right arm, tongue, foot) from EEG.</p>
<ul>
  <li>LLM performs its own classification operations based on EEG data.</li>
  <li>We trained gpt-4o model utilizing fine-tuning for better performance.</li>
</ul>
<img width="952" alt="EEG-GPT pipiline" src="https://github.com/LimDoHyeon/EEG-GPT/assets/94499717/fa86e248-801a-49b1-a46c-9c28d38610c8">

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
<p>The following four features were used: </p>
<ul><strong>
  <li>alpha:delta power ratio</li>
  <li>theta:alpha power ratio</li>
  <li>delta:theta power ratio</li>
</ul></strong>
<p>And for feature selection, <strong>Fisher Ratio</strong> is utilized.</p>
<p>(PSD plot here)</p>


<br>
<h2>Evaluation</h2>
<p>Performance metrics:
<ul>
  <li><strong>Accuracy</strong></li>
  <li><strong>F1 score</strong></li>
  <li><strong>Cohen's Kappa Coefficient</strong></li>
</ul>
<p>(Metrics plot here)</p>
</p>

<br>
<h2>Reference</h2>
<p>- Jonathan W.Kim et al. EEG-GPT: EXPLORING CAPABILITIES OF LARGE LANGUAGE MODELS FOR EEG CLASSIFICATION AND INTERPRETATION. 2024. arXiv:2401.18006 [q-bio.QM]</p>
