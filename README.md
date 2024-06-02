<h2>Fine-tuned GPT for Electroencephalography(EEG) data</h2>
<p>This is LLM Fine tuning model that classifies four movements (left arm, right arm, tongue, foot) from EEG.<br>Fine-tuning and In-context learning methods are used for extracting useful insights.</p>
<br>
<h2>Requirements</h2>
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
<p>Download : BBCI Competition III (https://www.bbci.de/competition/iii/download/index.html?agree=yes&submit=Submit)</p>

<br>
<h2>Features</h2>
<p>The following four features were used: </p>
<ul>
  <li>alpha:delta power ratio</li>
  <li>theta:alpha power ratio</li>
  <li>delta:theta power ratio</li>
  <li>fisher ratio</li>
</ul>


<br>
<h2>Evaluation</h2>
<br>
<h2>Reference</h2>
<p>- Jonathan W.Kim et al. EEG-GPT: EXPLORING CAPABILITIES OF LARGE LANGUAGE MODELS FOR EEG CLASSIFICATION AND INTERPRETATION. 2024. arXiv:2401.18006 [q-bio.QM]</p>
