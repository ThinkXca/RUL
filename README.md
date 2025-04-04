<div align="center">

  <h1 align="center">Survival Analysis with Machine Learning for Predicting Li-ion Battery Remaining Useful Life</h1>
  <h2 align="center">2025</h2>

### [Paper](https://doi.org/10.48550/arXiv.2503.13558) | [web](https://thinkx.ca/research/rul) | [Original data] (https://data.matr.io/1/)
</div>

## BibTeX
```
@article{RUL2025,
  author = {Jingyuan Xue, Longfei Wei, Fang Sheng, Yuxin Gao, Jianfei Zhang},
  title = {Survival Analysis with Machine Learning for Predicting Li-ion Battery Remaining Useful Life},
  journal = {arXiv: 2503.13558},
  year = {2025},
  doi = {10.48550/arXiv.2503.13558}
}
```

## 📖 Data
<p align="justify">

The two preprocessed datasets in Data (charge.csv and discharge.csv) have been extracted from the original Toyota data. These two datasets can be loaded by the five survival models directly.

</p>



## 🗓️ News

<p>[2025.03.16] Many thanks to <a href="https://github.com/jianfeizhang">Jianfei Zhang</a> and <a href="https://github.com/wei872">Longfei Wei</a> for their contributions to the code.</p>

<p>[2025.01.10] Many thanks to <a href="https://github.com/Rasheed19/battery-survival">battery-survival</a>, Support provided for lithium battery data processing.</p>

<p>Some amazing enhancements will also come out this year.</p>







## 🗓️ TODO
- [✔] Model information reference: <a href="https://github.com/georgehc/survival-intro">model</a>
- [✔] Processed data would be requested from our <a href="https://thinkx.ca">website</a>
- [✔] Dataset preprocessing-related content arrangement dependencies： <a href="https://www.sciencedirect.com/science/article/pii/S2666546824001319">Data preprocessing</a>
- [✔] Li-ion battery data source: <a href="https://data.matr.io/1/.">Toyota dataset</a>

</div>

<strong>Some amazing enhancements are under development. We warmly welcome anyone to collaborate to improve this repository. Please send us an email if you are interested!</strong>


## 🚀 Setup

#### Tested Environment
window 11, GeForce 4070, CUDA 12.1 (tested), C++17

#### Clone the repo.
```
git clone https://github.com/ThinkXca/RUL.git --recursive
```

#### Environment setup 
```
# All installed libraries and their version information are listed in requirements.txt. If you only need to run the above model, it is not necessary to install all libraries. Any missing dependencies for the model can be found in the requirements.txt file.
```


#### Run the codes
```
python Cox.py
python CoxPH.py
python CoxTime.py
python DeepHit.py
python MTLR.py
```




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
