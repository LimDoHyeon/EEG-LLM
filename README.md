<h2>Fine-tuned GPT for Electroencephalography(EEG) data</h2>
<p>This project aims to make GPT model for using EEG data more efficient.<br>I utilize fine-tuning and in-context learning methods for extracting useful insights.</p>
<br>
<h2>Requirements</h2>
<blockquote>!pip install -r requirements.txt</blockquote>
<br>
<h2>Data&Training</h2>
<p><strong>Data</strong> : BBCI Competition III (https://www.bbci.de/competition/iii/download/index.html?agree=yes&submit=Submit)</p>
<i>Short description :
  <ul>
    <li>cued motor imagery (multi-class) with 4 classes (left hand, right hand, foot, tongue) three subjects (ranging from quite good to fair performance)</li>
    <li>EEG, 60 channels, 60 trials per class</li>
    <li>performance measure: kappa-coefficient</li></i>
  </ul><br>

<p><strong>Data description</strong> : https://www.bbci.de/competition/iii/desc_IIIa.pdf</p>

<br>
<h2>Evaluation</h2>
<br>
<h2>Reference</h2>
<p>- Jonathan W.Kim et al. EEG-GPT: EXPLORING CAPABILITIES OF LARGE LANGUAGE MODELS FOR EEG CLASSIFICATION AND INTERPRETATION. 2024. arXiv:2401.18006 [q-bio.QM]</p>
<p>- Shunyu Yao et al. Tree of Thoughts: Deliberate Problem Solving with Large Language Models. 2023. arXiv: 2305.10601 [cs.CL].</p>
