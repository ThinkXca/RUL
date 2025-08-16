<div align="center">

  <h1 align="center">Survival Analysis with Machine Learning for Predicting Li-ion Battery Remaining Useful Life</h1>
  <h2 align="center">2025</h2>

### [Paper](https://doi.org/10.48550/arXiv.2503.13558) | [Web](https://thinkx.ca/research/rul) | [Original data](https://data.matr.io/1/)
</div>

## BibTeX
```
@article{RUL2025,
  author = {Jingyuan Xue, Longfei Wei, Dongjing Jiang, Fang Sheng, Russell Greiner, Jianfei Zhang},
  title = {Survival Analysis with Machine Learning for Predicting Li-ion Battery Remaining Useful Life},
  journal = {arXiv: 2503.13558},
  year = {2025},
  doi = {10.48550/arXiv.2503.13558}
}
```

## Data
<p align="justify">

### NASA

<p>NASA dataset: <a href="https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip">Download</a></p> 

<p>Specifically, we only used the data contained in <code>./5.+Battery+Data+Set.zip/BatteryAgingARC-FY08Q4.zip</code> as the experimental dataset.</p>

### Toyota

<p>Download original Toyota data:</p>

Dataset 1: <a href="https://data.matr.io/1/api/v1/file/5c86c0b5fa2ede00015ddf66/download">Download</a>

Dataset 2: <a href="https://data.matr.io/1/api/v1/file/5c86bf13fa2ede00015ddd82/download">Download</a>

Dataset 3: <a href="https://data.matr.io/1/api/v1/file/5c86bd64fa2ede00015ddbb2/download">Download</a>

Dataset 4: <a href="https://data.matr.io/1/api/v1/file/5dcef689110002c7215b2e63/download">Download</a>

Dataset 5: <a href="https://data.matr.io/1/api/v1/file/5dceef1e110002c7215b28d6/download">Download</a>

Dataset 6: <a href="https://data.matr.io/1/api/v1/file/5dcef6fb110002c7215b304a/download">Download</a>

Dataset 7: <a href="https://data.matr.io/1/api/v1/file/5dceefa6110002c7215b2aa9/download">Download</a>

Dataset 8: <a href="https://data.matr.io/1/api/v1/file/5dcef152110002c7215b2c90/download">Download</a>


<p>Use <a href="https://github.com/Rasheed19/battery-survival">battery-survival</a> to generate our preprocessed data from the original data</p>

## More
- [✔] Model information reference: <a href="https://github.com/georgehc/survival-intro">model</a>
- [✔] Processed data would be requested from our <a href="https://thinkx.ca">website</a>
- [✔] Dataset preprocessing-related content arrangement dependencies： <a href="https://www.sciencedirect.com/science/article/pii/S2666546824001319">Data preprocessing</a>
- [✔] Li-ion battery data source: <a href="https://data.matr.io/1/.">Toyota dataset</a> and 
<a href="https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip">NASA Battery Dataset</a>

## Acknowledgements

<p>Many thanks to <a href="https://github.com/jianfeizhang">Jianfei Zhang</a> and <a href="https://github.com/wei872">Longfei Wei</a> for their contributions to the code.</p>

<p>Many thanks to <a href="https://github.com/Rasheed19/battery-survival">battery-survival</a>, Support provided for lithium battery data processing.</p>

<p>Some amazing enhancements will also come out this year.</p>

</div>

<strong>Some amazing enhancements are under development. We warmly welcome anyone to collaborate to improve this repository. Please send us an email if you are interested!</strong>


## Run

#### Clone the repo.
```
git clone https://github.com/ThinkXca/RUL.git --recursive
```

#### Environment setup 
```
conda create -n battery-notebook python=3.10.15

conda activate battery-notebook

# Enter the Nasa folder
# For both the NASA and Toyota folder, all installed libraries and their version information are listed in the requirements.txt file. The requirements.txt file under the NASA folder already supports running both NASA and Toyota. If any dependencies are still missing, please refer to the requirements.txt file in the Toyota folder for the required libraries and their versions.

pip install -r requirements.txt
```



#### NASA Run the codes
```
# Enter the NASA folder
pip install jupyter

jupyter notebook

# Read sequentially
deval-discharge.ipynb
deval-valid-data.ipynb
extractfeature.ipynb
model-Cox.ipynb
model-CoxPH.ipynb
model-CoxTime.ipynb
model-DeepHit.ipynb
model-MTLR.ipynb
```


#### Toyota Run the codes
<p> When using the signature path method for battery representation extraction, the procedure is the same as that used for the NASA dataset. Please refer to the ./NASA/extractfeature.ipynb file. The ./Toyota/Signature/ExtractedData directory contains the data obtained through signature-based deep feature extraction.</p>


<p>For the Toyota dataset, we followed the processing example from <a href="https://github.com/Rasheed19/battery-survival">battery-survival</a>. Similar to their approach, we used the 50th cycle in the Toyota dataset as the experimental data, consisting of a total of 362 batteries. Among them, 174 batteries experienced uncensored events and 188 experienced censored events. The processed battery data can be found in the ./Toyota/TLSTM/OriginalData directory.

For the TLSTM model architecture, we drew inspiration from the encoder–decoder based TLSTM project. When extracting TLSTM representations of battery voltage and time, you only need to run:</p>

```
# Enter the Toyota/TLSTM folder
python tlstm_ae_battery_feature_extractor.py
```

<p>The data extracted by TLSTM was placed in the ./Toyota/TLSTM/ExtractedData folder for remaining useful life (RUL) analysis of the batteries. It is important to note that, in order to align with the experimental data reported in the paper, we modified the batch column in the TLSTM-extracted representations to a group column. Based on the group information, the data can be partitioned for survival analysis and predictive evaluation. Please note that if you directly run the downloaded code, it corresponds to the ungrouped case. The code can be executed as follows:</p>

```
# Enter the Toyota/TLSTM folder
python Cox.py
python CoxPH.py
python CoxTime.py
python DeepHit.py
python MTLR.py
```



## Results
#### Performance Comparison of Different Models on the Toyota Dataset

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

#### Performance Comparison of Different Models on the NASA Dataset
<table>
  <tr>
    <th rowspan="2">Model</th>
    <th colspan="3">Discharge</th>
  </tr>
  <tr>
    <th>T-AUC</th>
    <th>C-Index</th>
    <th>IBS</th>
  </tr>
  <tr>
    <td><b>Cox</b></td>
    <td>.999 (.001)</td>
    <td>.965 (.004)</td>
    <td>.010 (.002)</td>
  </tr>
  <tr>
    <td><b>CoxTime</b></td>
    <td>.998 (.001)</td>
    <td>.959 (.002)</td>
    <td>.007 (.001)</td>
  </tr>
  <tr>
    <td><b>CoxPH</b></td>
    <td>.999 (.000)</td>
    <td>.966 (.004)</td>
    <td>.014 (.001)</td>
  </tr>
  <tr>
    <td><b>DeepHit</b></td>
    <td>.975 (.005)</td>
    <td>.930 (.002)</td>
    <td>.045 (.005)</td>
  </tr>
  <tr>
    <td><b>MTLR</b></td>
    <td>.982 (.009)</td>
    <td>.955 (.013)</td>
    <td>.048 (.000)</td>
  </tr>
</table>
