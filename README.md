# Impact of filter feature selection on classification: an empirical study
This repository houses the codes associated with the experiments for our paper titled "Impact of filter feature selection on classification: an empirical study".

## Prerequisites
Implementations in [scikit-feature](https://jundongl.github.io/scikit-feature/index.html) were used in this work. In other to run the codes, prerequisite libraries in **requirements.txt** file must be installed. Before doing so I recommend setting up a [Virtual Environment (VE)](https://docs.python.org/3/library/venv.html) for this experiment after which you install the required libraries in the VE with the following command:
```
pip install -r requirements.txt
```
## Example
After navigating to the path where the experiment scripts are stored (do make sure your VE is activated), a script is executed with the following command:
```
python script-name.py >> outputfile-name.csv
```
This returns the output in a csv file ready to be used for analysis.


## Datasets

Below we present the details of the datasets chosen and used in this work

|           Dataset Name           | OpenML Identifier | #Classes | #Features | #Instances | Class Balance | Factor |
|:--------------------------------:|:-----------------:|:--------:|:---------:|:----------:|:-------------:|:------:|
|            confidence            |        [1015](https://www.openml.org/d/1015)       |     2    |     4     |     72     |     0.055     |  0000  |
| blood-transfusion-service-center |        [1464](https://www.openml.org/d/1464)       |     2    |     5     |     748    |     0.0067    |  0000  |
|           fri_c3_250_10          |        [793](https://www.openml.org/d/793)        |     2    |     11    |     250    |     0.9954    |  0001  |
|           disclosure_z           |        [931](https://www.openml.org/d/931)        |     2    |     4     |     662    |     0.9981    |  0001  |
|            page-blocks           |        [1021](https://www.openml.org/d/1021)       |     2    |     11    |    5473    |     0.4763    |  0010  |
| wilt                             |       [40983](https://www.openml.org/d/40983)       |     2    |     6     |    4839    |     0.3029    |  0010  |
|            delta_elevators       |        [819](https://www.openml.org/d/819)        |     2    |     7     |    9517    |       1       |  0101  |
|               stock              |        [841](https://www.openml.org/d/841)        |     2    |     10    |     950    |     0.9995    |  0101  |
|           synthetic_control      |        [1004](https://www.openml.org/d/1004)       |     2    |     61    |     600    |     0.6500    |  0100  |
|                  ar4             |        [1061](https://www.openml.org/d/1061)       |     2    |     30    |     107    |     0.6950    |  0100  |
|                isolet            |       [41966](https://www.openml.org/d/41966)       |     2    |    618    |     600    |       1       |  0101  |
|            fri_c4_250_100        |        [834](https://www.openml.org/d/834)        |     2    |    101    |     250    |     0.9896    |  0101  |
|           mfeat-zernike          |        [995](https://www.openml.org/d/995)        |     2    |     48    |    2000    |     0.4690    |  0110  |
|                clean2            |       [40666](https://www.openml.org/d/40666)       |     2    |    169    |    6598    |     0.6201    |  0110  |
|                 gina             |       [41158](https://www.openml.org/d/41158)       |     2    |    971    |    3153    |     0.9998    |  0111  |
|            philippine            |       [41145](https://www.openml.org/d/41145)       |     2    |    309    |    5832    |       1       |  0111  |
|              Engine1             |        [4340](https://www.openml.org/d/4340)       |     3    |     6     |     383    |     0.5905    |  1000  |
|              heart-h             |        [1565](https://www.openml.org/d/1565)       |     5    |     14    |     294    |     0.7065    |  1000  |
|     LED-display-domain-7digit    |       [40496](https://www.openml.org/d/40496)       |    10    |     8     |     500    |     0.9971    |  1001  |
|         heart-long-beach         |        [1512](https://www.openml.org/d/1512)       |     5    |     14    |     200    |     0.9365    |  1001  |
|        wine-quality-white        |       [40498](https://www.openml.org/d/40498)       |     7    |     12    |    4898    |     0.6632    |  1010  |
|           volcanoes-d3           |        [1540](https://www.openml.org/d/1540)       |     5    |     4     |    9285    |     0.1761    |  1010  |
|          JapaneseVowels          |        [375](https://www.openml.org/d/375)        |     9    |     15    |    9961    |     0.9881    |  1011  |
|       wall-robot-navigation      |        [1526](https://www.openml.org/d/1526)       |     4    |     5     |    5456    |     0.8573    |  1011  |
|           meta_all.arff          |        [275](https://www.openml.org/d/275)        |     6    |     63    |     71     |     0.6913    |  1100  |
|        meta_ensembles.arff       |        [277](https://www.openml.org/d/277)        |     4    |     63    |     74     |     0.7719    |  1100  |
|         synthetic_control        |        [377](https://www.openml.org/d/377)        |     6    |     61    |     600    |       1       |  1101  |
|        robot-failures-lp3        |        [1518](https://www.openml.org/d/1518)       |     4    |     91    |     47     |     0.9102    |  1101  |
|           Indian_pines           |       [41972](https://www.openml.org/d/41972)       |     8    |    221    |    9144    |     0.6941    |  1110  |
|         cardiotocography         |        [1560](https://www.openml.org/d/1560)       |     3    |     36    |    2126    |     0.614     |  1110  |
|              cnae-9              |        [1468](https://www.openml.org/d/1468)       |     9    |    857    |    1080    |       1       |  1111  |
|              texture             |       [40499](https://www.openml.org/d/40499)       |    11    |     41    |    5500    |       1       |  1111  |

## Contributor
- Uchechukwu Fortune, Njoku (unjoku@essi.upc.edu)

## Citation
