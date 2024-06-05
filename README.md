<h2>Fine-tuned GPT for Electroencephalography(EEG) data</h2>
<p>This is LLM Fine tuning model that classifies four movements (left arm, right arm, tongue, foot) from EEG.</p>
<ul>
  <li>LLM performs its own classification operations based on EEG data.</li>
  <li>Fine-tuning and In-context learning methods are used for better performance.</li>
  <li>In this project, extra ML model is used for measuring performance since the data we used has no label(NaN) on their test set.</li>
</ul>
<img width="952" alt="EEG-GPT pipiline" src="https://github.com/LimDoHyeon/EEG-GPT/assets/94499717/fa86e248-801a-49b1-a46c-9c28d38610c8">

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
<p><strong>Download</strong> : BBCI Competition III (https://www.bbci.de/competition/iii/download/index.html?agree=yes&submit=Submit)</p>

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
<p><strong>Cohen's Kappa Coefficient</strong></p>

<br>
<h2>Reference</h2>
<p>- Jonathan W.Kim et al. EEG-GPT: EXPLORING CAPABILITIES OF LARGE LANGUAGE MODELS FOR EEG CLASSIFICATION AND INTERPRETATION. 2024. arXiv:2401.18006 [q-bio.QM]</p>
<p>- G. Gomez-Herrero et al., "Automatic Removal of Ocular Artifacts in the EEG without an EOG Reference Channel," Proceedings of the 7th Nordic Signal Processing Symposium - NORSIG 2006, Reykjavik, Iceland, 2006, pp. 130-133, doi: 10.1109/NORSIG.2006.275210. keywords: {Electroencephalography;Electrooculography;Electrodes;Epilepsy;Scalp;Sensor arrays;Blind source separation;Source separation;Patient monitoring;Principal component analysis}</p>
