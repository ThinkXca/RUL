<div align="center">

  <h1 align="center">Survival Analysis with Machine Learning for Predicting Li-ion Battery Remaining Useful Life</h1>
  <h2 align="center">2025</h2>

### [Paper]() | [web](https://thinkx.ca/research/rul)


<div align="left">

## 📖 Abstract
<p align="justify">

The accurate prediction of RUL for lithium-ion batteries is crucial for enhancing the reliability and longevity of energy storage systems. Traditional methods for RUL prediction often struggle with issues such as data sparsity, varying battery chemistries, and the inability to capture complex degradation patterns over time. In this study, we propose a survival analysis-based framework combined with deep learning models to predict the RUL of lithium-ion batteries. Specifically, we utilize five advanced models: the Cox-type models (including Cox, CoxPHm, and CoxTime), two machine-learning-based models (including DeepHit and MTLR). These models address the challenges of accurate RUL estimation by transforming raw time-series battery data into survival data, including key degradation indicators such as voltage, current, and internal resistance. Advanced feature extraction techniques are employed to enhance the model’s robustness in diverse real-world scenarios, including varying charging conditions and battery chemistries. Our models are tested using 10-fold cross-validation, ensuring generalizability and minimizing overfitting. Experimental results show that our survival-based framework significantly improves RUL prediction accuracy compared to traditional methods, providing a reliable tool for battery management and maintenance optimization. This study contributes to the advancement of predictive maintenance in battery technology, offering valuable insights for both researchers and industry practitioners aiming to enhance the operational lifespan of lithium-ion batteries.

</p>
</div>

<div align="left">

## 🗓️ News

<p>[2024.10.10] Many thanks to <a href="https://github.com/wei872">Longfei Wei</a>, Improvements made to the code</p>

<p>[2024.10.10] Many thanks to <a href="https://github.com/Rasheed19/battery-survival">battery-survival</a>, Support provided for lithium battery data processing.</p>

<p>Some amazing enhancements will also come out this year.</p>

</div>





<div align="left">

## 🗓️ TODO

- [✔] Model information reference: <a href="https://github.com/georgehc/survival-intro">model</a>
- [✔] Dataset preprocessing-related content arrangement dependencies： <a href="https://www.sciencedirect.com/science/article/pii/S2666546824001319">Data preprocessing</a>
- [✔] Li-ion battery data source: <a href="https://data.matr.io/1/.">Toyota dataset</a>

</div>

<strong>Some amazing enhancements are under development. We are warmly welcome anyone to collaborate in improving this repository. Please send me an email if you are interested!</strong>




<div align="left">

## 🚀 Setup

#### Tested Environment
window 11, GeForce 4070, CUDA 12.1 (tested), C++17

#### Clone the repo.
```
git clone https://github.com/ThinkXca/URL.git --recursive
```

#### Environment setup 
```
# All installed libraries and their version information are listed in requirements.txt. If you only need to run the above model, it is not necessary to install all libraries. Any missing dependencies for the model can be found in the requirements.txt file.
```
</div>

<div align="left">

#### Run the codes
```
python Cox.py
python CoxPH.py
python CoxTime.py
python DeepHit.py
python MTLR.py
```

</div>

<div align="left">

#### Comparison of Model Performance

<table>
  <tr>
    <th rowspan="2">Model</th>
    <th colspan="3">Charge</th>
    <th colspan="3">Discharge</th>
  </tr>
  <tr>
    <th>T-AUC</th>
    <th>C-Index</th>
    <th>IBS</th>
    <th>T-AUC</th>
    <th>C-Index</th>
    <th>IBS</th>
  </tr>
  <tr>
    <td><b>Cox</b></td>
    <td>.909 (.027)</td>
    <td>.820 (.030)</td>
    <td>.031 (.006)</td>
    <td>.932 (.018)</td>
    <td>.859 (.020)</td>
    <td>.048 (.008)</td>
  </tr>
  <tr>
    <td><b>CoxTime</b></td>
    <td>.919 (.024)</td>
    <td>.832 (.033)</td>
    <td>.028 (.006)</td>
    <td>.929 (.018)</td>
    <td>.853 (.021)</td>
    <td>.051 (.009)</td>
  </tr>
  <tr>
    <td><b>CoxPH</b></td>
    <td>.889 (.006)</td>
    <td>.798 (.015)</td>
    <td>.035 (.001)</td>
    <td>.896 (.037)</td>
    <td>.826 (.020)</td>
    <td>.056 (.012)</td>
  </tr>
  <tr>
    <td><b>DeepHit</b></td>
    <td>.730 (.076)</td>
    <td>.823 (.044)</td>
    <td>.085 (.012)</td>
    <td>.866 (.059)</td>
    <td>.816 (.046)</td>
    <td>.076 (.020)</td>
  </tr>
  <tr>
    <td><b>MTLR</b></td>
    <td>.809 (.058)</td>
    <td>.844 (.024)</td>
    <td>.040 (.007)</td>
    <td>.922 (.029)</td>
    <td>.835 (.025)</td>
    <td>.051 (.009)</td>
  </tr>
</table>
</div>
