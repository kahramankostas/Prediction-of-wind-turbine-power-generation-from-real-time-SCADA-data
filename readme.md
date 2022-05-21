
## Prediction of wind turbine power generation from real-time SCADA data


The models to be created for this problem are the models that will predict the power values ​​expected to be produced by a wind turbine. Thus, by comparing the actual production of a wind turbine with these estimation results, it will be presented to the investor to what extent the turbine produces less than it should be. From this point of view, the investor will be able to realize that there is a performance problem related to the turbine and will be able to initiate root cause analysis.

The data set presented in the problem consists of real-time SCADA data. Each data value belongs only to the relevant time period and the input variables transmitted in the data set for the time period to be predicted are prepared to be used to predict the power generation result in the same time period.


In the shared data set, the real-time power generation amount (Power(kW)) of a wind turbine belonging to Enerjisa Üretim between 01.01.2019 and 14.08.2021 is given on a 10-minute basis.




## Information presented in the dataset and its units



|Column                                           |Unit   |
|-------------------------------------------------|-------|
|Timestamp                                        |()     |
|Gearbox_T1_High_Speed_Shaft_Temperature          |(°C)   |
|Gearbox_T3_High_Speed_Shaft_Temperature          |(°C)   |
|Gearbox_T1_Intermediate_Speed_Shaft_Temperature  |(°C)   |
|Temperature Gearbox Bearing Hollow Shaft         |(°C)   |
|Tower Acceleration Normal                        |(mm/s²)|
|Gearbox_Oil-2_Temperature                        |(°C)   |
|Tower Acceleration Lateral                       |(mm/s²)|
|Temperature Bearing_A                            |(°C)   |
|Temperature Trafo-3                              |(°C)   |
|Gearbox_T3_Intermediate_Speed_Shaft_Temperature  |(°C)   |
|Gearbox_Oil-1_Temperature                        |(°C)   |
|Gearbox_Oil_Temperature                          |(°C)   |
|Torque                                           |(%)    |
|Converter Control Unit Reactive Power            |(kVAr) |
|Temperature Trafo-2                              |(°C)   |
|Reactive Power                                   |(kVAr) |
|Temperature Shaft Bearing-1                      |(°C)   |
|Gearbox_Distributor_Temperature                  |(°C)   |
|Moment D Filtered                                |(kNm)  |
|Moment D Direction                               |(kNm)  |
|N-set 1                                          |(rpm)  |
|Operating State                                  |( )    |
|Power Factor                                     |( )    |
|Temperature Shaft Bearing-2                      |(°C)   |
|Temperature_Nacelle                              |(°C)   |
|Voltage A-N                                      |(V)    |
|Temperature Axis Box-3                           |(°C)   |
|Voltage C-N                                      |(V)    |
|Temperature Axis Box-2                           |(°C)   |
|Temperature Axis Box-1                           |(°C)   |
|Voltage B-N                                      |(V)    |
|Nacelle Position_Degree                          |(°)    |
|Converter Control Unit Voltage                   |(V)    |
|Temperature Battery Box-3                        |(°C)   |
|Temperature Battery Box-2                        |(°C)   |
|Temperature Battery Box-1                        |(°C)   |
|Hydraulic Prepressure                            |(bar)  |
|Angle Rotor Position                             |(°)    |
|Temperature Tower Base                           |(°C)   |
|Pitch Offset-2 Asymmetric Load Controller        |(°)    |
|Pitch Offset Tower Feedback                      |(°)    |
|Line Frequency                                   |(Hz)   |
|Internal Power Limit                             |(kW)   |
|Circuit Breaker cut-ins                          |( )    |
|Particle Counter                                 |( )    |
|Tower Accelaration Normal Raw                    |(mm/s²)|
|Torque Offset Tower Feedback                     |(Nm)   |
|External Power Limit                             |(kW)   |
|Blade-2 Actual Value_Angle-B                     |(°)    |
|Blade-1 Actual Value_Angle-B                     |(°)    |
|Blade-3 Actual Value_Angle-B                     |(°)    |
|Temperature Heat Exchanger Converter Control Unit|(°C)   |
|Tower Accelaration Lateral Raw                   |(mm/s²)|
|Temperature Ambient                              |(°C)   |
|Nacelle Revolution                               |( )    |
|Pitch Offset-1 Asymmetric Load Controller        |(°)    |
|Tower Deflection                                 |(ms)   |
|Pitch Offset-3 Asymmetric Load Controller        |(°)    |
|Wind Deviation 1 seconds                         |(°)    |
|Wind Deviation 10 seconds                        |(°)    |
|Proxy Sensor_Degree-135                          |(mm)   |
|State and Fault                                  |( )    |
|Proxy Sensor_Degree-225                          |(mm)   |
|Blade-3 Actual Value_Angle-A                     |(°)    |
|Scope CH 4                                       |( )    |
|Blade-2 Actual Value_Angle-A                     |(°)    |
|Blade-1 Actual Value_Angle-A                     |(°)    |
|Blade-2 Set Value_Degree                         |(°)    |
|Pitch Demand Baseline_Degree                     |(°)    |
|Blade-1 Set Value_Degree                         |(°)    |
|Blade-3 Set Value_Degree                         |(°)    |
|Moment Q Direction                               |(kNm)  |
|Moment Q Filltered                               |(kNm)  |
|Proxy Sensor_Degree-45                           |(mm)   |
|Turbine State                                    |( )    |
|Proxy Sensor_Degree-315                          |(mm)   |


# Preprocessing


```python
#!/usr/bin/env python
# coding: utf-8
import numpy as np
import csv

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import pickle
import time
import os

import pandas as pd


```

 ### unzip dataset


```python
import zipfile
path_to_zip_file="enerjisa-uretim-hackathon.zip"
with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
    zip_ref.extractall(path_to_zip_file[:-4])
```

### take dataset files as a lit in "csvs"


```python
def find_the_way(path,file_format):
    files_add = []

    for r, d, f in os.walk(path):
        for file in f:
            if file_format in file:
                files_add.append(os.path.join(r, file))  
    return files_add

path=path_to_zip_file[:-4]
csvs=find_the_way(path,'.csv')
csvs

```




    ['enerjisa-uretim-hackathon\\features.csv',
     'enerjisa-uretim-hackathon\\feature_units.csv',
     'enerjisa-uretim-hackathon\\power.csv',
     'enerjisa-uretim-hackathon\\sample_submission.csv']



### replace nan and inf value with 0


```python
features=pd.read_csv(csvs[0])
labels=pd.read_csv(csvs[2])

features.replace([np.inf, -np.inf], np.nan, inplace=True)
features=features.fillna(0)
```

### create and add a new feature related with timeseries


```python
ay_ve_gun=[]
for i in features["Timestamp"]:
    month=int(i[5:7])*100
    day=(int(i[8:10])//10+1)
    if day==4:
        day=3
    ay_ve_gun.append(month+day)
```


```python
features["ay_ve_gun"]=ay_ve_gun
```

### split labelled and unlabelled data


```python
train_size=len(labels)
main=features[0:train_size]
submission=features[train_size:]
```

### add labels to dataframe


```python
main["Power(kW)"]=labels["Power(kW)"]
```

###  show unlabeled data which we will not use


```python
submission
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Timestamp</th>
      <th>Gearbox_T1_High_Speed_Shaft_Temperature</th>
      <th>Gearbox_T3_High_Speed_Shaft_Temperature</th>
      <th>Gearbox_T1_Intermediate_Speed_Shaft_Temperature</th>
      <th>Temperature Gearbox Bearing Hollow Shaft</th>
      <th>Tower Acceleration Normal</th>
      <th>Gearbox_Oil-2_Temperature</th>
      <th>Tower Acceleration Lateral</th>
      <th>Temperature Bearing_A</th>
      <th>Temperature Trafo-3</th>
      <th>...</th>
      <th>Blade-2 Set Value_Degree</th>
      <th>Pitch Demand Baseline_Degree</th>
      <th>Blade-1 Set Value_Degree</th>
      <th>Blade-3 Set Value_Degree</th>
      <th>Moment Q Direction</th>
      <th>Moment Q Filltered</th>
      <th>Proxy Sensor_Degree-45</th>
      <th>Turbine State</th>
      <th>Proxy Sensor_Degree-315</th>
      <th>ay_ve_gun</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>136730</th>
      <td>2021-08-15 00:00:00</td>
      <td>60.068333</td>
      <td>62.0</td>
      <td>56.000000</td>
      <td>58.000000</td>
      <td>125.218666</td>
      <td>60.000000</td>
      <td>64.707336</td>
      <td>54.348331</td>
      <td>121.000000</td>
      <td>...</td>
      <td>9.493241</td>
      <td>8.925109</td>
      <td>9.014512</td>
      <td>8.266594</td>
      <td>-41.861877</td>
      <td>-37.917656</td>
      <td>5.739297</td>
      <td>1.0</td>
      <td>5.734730</td>
      <td>802</td>
    </tr>
    <tr>
      <th>136731</th>
      <td>2021-08-15 00:10:00</td>
      <td>60.000000</td>
      <td>62.0</td>
      <td>56.000000</td>
      <td>57.036667</td>
      <td>145.160309</td>
      <td>59.279999</td>
      <td>64.127480</td>
      <td>58.098331</td>
      <td>120.971664</td>
      <td>...</td>
      <td>7.507399</td>
      <td>6.937748</td>
      <td>7.022389</td>
      <td>6.287027</td>
      <td>-19.210815</td>
      <td>-19.602339</td>
      <td>5.720869</td>
      <td>1.0</td>
      <td>5.726634</td>
      <td>802</td>
    </tr>
    <tr>
      <th>136732</th>
      <td>2021-08-15 00:20:00</td>
      <td>60.000000</td>
      <td>62.0</td>
      <td>55.853333</td>
      <td>57.000000</td>
      <td>129.239914</td>
      <td>59.000000</td>
      <td>54.563091</td>
      <td>60.360001</td>
      <td>120.028336</td>
      <td>...</td>
      <td>8.065812</td>
      <td>7.497398</td>
      <td>7.581376</td>
      <td>6.844808</td>
      <td>-28.144068</td>
      <td>-34.329105</td>
      <td>5.727475</td>
      <td>1.0</td>
      <td>5.728649</td>
      <td>802</td>
    </tr>
    <tr>
      <th>136733</th>
      <td>2021-08-15 00:30:00</td>
      <td>60.000000</td>
      <td>62.0</td>
      <td>55.000000</td>
      <td>57.000000</td>
      <td>140.151611</td>
      <td>59.000000</td>
      <td>61.899250</td>
      <td>61.715000</td>
      <td>120.000000</td>
      <td>...</td>
      <td>8.132490</td>
      <td>7.565773</td>
      <td>7.654368</td>
      <td>6.909220</td>
      <td>-7.592476</td>
      <td>-11.718444</td>
      <td>5.728980</td>
      <td>1.0</td>
      <td>5.739824</td>
      <td>802</td>
    </tr>
    <tr>
      <th>136734</th>
      <td>2021-08-15 00:40:00</td>
      <td>60.000000</td>
      <td>62.0</td>
      <td>55.000000</td>
      <td>57.000000</td>
      <td>126.124702</td>
      <td>59.000000</td>
      <td>56.804501</td>
      <td>62.698334</td>
      <td>120.000000</td>
      <td>...</td>
      <td>9.546413</td>
      <td>8.974770</td>
      <td>9.064083</td>
      <td>8.313858</td>
      <td>-7.760864</td>
      <td>-9.863355</td>
      <td>5.736651</td>
      <td>1.0</td>
      <td>5.747692</td>
      <td>802</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>154257</th>
      <td>2021-12-14 23:10:00</td>
      <td>65.811668</td>
      <td>0.0</td>
      <td>59.945000</td>
      <td>62.808334</td>
      <td>225.038239</td>
      <td>65.300003</td>
      <td>109.889709</td>
      <td>61.000000</td>
      <td>97.000000</td>
      <td>...</td>
      <td>15.820095</td>
      <td>15.199166</td>
      <td>15.235223</td>
      <td>14.540556</td>
      <td>-29.340843</td>
      <td>-27.513502</td>
      <td>5.746916</td>
      <td>1.0</td>
      <td>5.756082</td>
      <td>1202</td>
    </tr>
    <tr>
      <th>154258</th>
      <td>2021-12-14 23:20:00</td>
      <td>68.586670</td>
      <td>0.0</td>
      <td>62.084999</td>
      <td>65.413330</td>
      <td>229.905838</td>
      <td>67.871666</td>
      <td>106.016670</td>
      <td>61.116665</td>
      <td>97.000000</td>
      <td>...</td>
      <td>16.504293</td>
      <td>15.876278</td>
      <td>15.917643</td>
      <td>15.207320</td>
      <td>-31.925669</td>
      <td>-30.197918</td>
      <td>5.749150</td>
      <td>1.0</td>
      <td>5.755406</td>
      <td>1202</td>
    </tr>
    <tr>
      <th>154259</th>
      <td>2021-12-14 23:30:00</td>
      <td>63.746666</td>
      <td>0.0</td>
      <td>59.965000</td>
      <td>64.051666</td>
      <td>223.352631</td>
      <td>64.461670</td>
      <td>111.690208</td>
      <td>61.293335</td>
      <td>97.000000</td>
      <td>...</td>
      <td>15.331903</td>
      <td>14.720088</td>
      <td>14.768394</td>
      <td>14.064686</td>
      <td>-53.071564</td>
      <td>-48.306511</td>
      <td>5.751807</td>
      <td>1.0</td>
      <td>5.747936</td>
      <td>1202</td>
    </tr>
    <tr>
      <th>154260</th>
      <td>2021-12-14 23:40:00</td>
      <td>66.643333</td>
      <td>0.0</td>
      <td>60.678333</td>
      <td>63.421665</td>
      <td>227.704514</td>
      <td>66.081665</td>
      <td>119.716499</td>
      <td>60.786667</td>
      <td>97.000000</td>
      <td>...</td>
      <td>16.481724</td>
      <td>15.887610</td>
      <td>15.945046</td>
      <td>15.230121</td>
      <td>-28.747763</td>
      <td>-23.844364</td>
      <td>5.747686</td>
      <td>1.0</td>
      <td>5.757787</td>
      <td>1202</td>
    </tr>
    <tr>
      <th>154261</th>
      <td>2021-12-14 23:50:00</td>
      <td>65.593330</td>
      <td>0.0</td>
      <td>60.738335</td>
      <td>64.731667</td>
      <td>223.235413</td>
      <td>65.891670</td>
      <td>103.372475</td>
      <td>60.395000</td>
      <td>97.000000</td>
      <td>...</td>
      <td>16.198933</td>
      <td>15.591414</td>
      <td>15.635881</td>
      <td>14.941538</td>
      <td>-28.904552</td>
      <td>-30.457935</td>
      <td>5.753047</td>
      <td>1.0</td>
      <td>5.761520</td>
      <td>1202</td>
    </tr>
  </tbody>
</table>
<p>17532 rows × 78 columns</p>
</div>



### split two part labelled data as training (67%) and testing (33%)


```python
train_size = int(len(main) * 0.67)
test_size = len(main) - train_size
train, test = main[0:train_size], main[train_size:]
```

### save training and testing datasets as csvs


```python
submission.to_csv("submission.csv",index=False)
train.to_csv("TT.csv",index=False)
test.to_csv("t.csv",index=False)

```

# Machine Learning Step


```python
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import sklearn

```


```python
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import  LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
kernel = DotProduct() + WhiteKernel()
from xgboost import XGBRegressor





```

### List of Machine learning algorithms


```python

estimators = [('ridge', RidgeCV()),
               ('lasso', LassoCV(random_state=42)),
               ('knr', KNeighborsRegressor(n_neighbors=20,
                                          metric='euclidean'))]
final_estimator = GradientBoostingRegressor(
    n_estimators=25, subsample=0.5, min_samples_leaf=25, max_features=1,
    random_state=42)
reg = StackingRegressor(
    estimators=estimators,
    final_estimator=final_estimator)
from sklearn.linear_model import TweedieRegressor
reg1 = GradientBoostingRegressor(random_state=1)
reg2 = RandomForestRegressor(random_state=1)
reg3 = LinearRegression()
ml_list={'LR':LinearRegression(),'DT':DecisionTreeRegressor(),
'BR':BayesianRidge(),
'EL':ElasticNet(),
'twd':TweedieRegressor(),
'LAS':Lasso(),
'rcv':RidgeCV(), 
'lcv':LassoCV(),'BAG':BaggingRegressor(),
'GBR':GradientBoostingRegressor(),
'RF':RandomForestRegressor(),
'KNN':KNeighborsRegressor(),
#'LRVR':LinearSVR(),'SVR':SVR(),
#'iso':IsotonicRegression(),
'vot':VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('lr', reg3)]),
'stc' : StackingRegressor(
    estimators=estimators,
    final_estimator=final_estimator),'XGB':XGBRegressor()}

```

### split dataframe as data (X) and label (y)


```python
def data_and_label(name):
    
    df = pd.read_csv(name)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df=df.fillna(0)
    del df["Timestamp"]
    X =df[df.columns[:-1]]
    X=np.array(X)
    y=np.array(df[df.columns[-1]])
    return X,y
path='./csv/'
```

### Evaluation - Calculate error score


```python
def score_erros(altime,train_time,test_time,expected,predicted,class_based_results,i,cv,dname,ii):     
      
    mse = mean_squared_error(expected, predicted)
    mae=mean_absolute_error(expected, predicted)
    rmse=mean_squared_error(expected, predicted, squared=False)
    r2=r2_score(expected, predicted)    
    precision,recall,f_score=0,0,0 
    print ('%-10s %-3s %-3s %-10s  %-8s %-8s %-8s %-11s %-8s %-8s %-8s %-6s %-6s %-16s' % (dname,i,cv,ii[0:6],str(round((precision),2)),str(round((recall),2)),str(round((f_score),2)),str(round((mse),2)), str(round((mae),2)),
        str(round((rmse),2)), str(round((r2),2)),str(round((train_time),2)),str(round((test_time),2)),altime))
    lines=str(dname)+","+str(i)+","+str(cv)+","+str(ii)+","+str(round((precision),15))+","+str(round((recall),15))+","+str(round((f_score),15))+","+str(round((mse),15))+","+str(round((mae),15))+","+str(round((rmse),15))+","+ str(round((r2),15))+","+str(round((train_time),15))+","+str(round((test_time),15))+"\n"
    
    return lines,class_based_results,mae

```

### ML Function 


```python
def ML(output,file,test_file,i):
    ths = open(output, "a")
    X_test,y_test=data_and_label(test_file)
    ths.write ("Dataset,T,CV,ML_alg,precision,recall,f_scor,mse,mae,rmse, r2  ,tra-T,test-T,total\n")


    fold=5
    repetition=1
    class_based_results= pd.DataFrame()
    target_names=[0,1]


    for ii in ml_list:
        mae_min=1000

        cv=0
        dataset=file[-20:-4]
        clf = ml_list[ii]
        second=time.time()
        X_train,y_train=data_and_label(file)
        clf.fit(X_train, y_train)  
        train_time=(float((time.time()-second)) )
        second=time.time()
        predicted=clf.predict(X_test)
        test_time=(float((time.time()-second)) )
        expected = y_test
        
        error=[]
        for j in range(len(y_test)):
            error.append(abs(float(y_test[j])-float(predicted[j])))
        error.sort()
        cep68 = round((error[round(68 * len(error) / 100)])**(1/2),2)
        cep95 = round((error[round(95 * len(error) / 100)])**(1/2),2)
        cep=str(cep68)+'   '+str(cep95)
        
        line,cb,mae=score_erros(cep,train_time,test_time,expected, predicted,class_based_results,i,cv,dataset,ii)

        filename=f".sav"
        filename=filename.replace('\\','_')
        pickle.dump(clf, open(filename, 'wb'))

        ths.write (line)
    ths.close()  
```

### tarining file


```python
csvs=find_the_way("./",'TT')
csvs
```




    ['./TT.csv']



# Results


```python
print ('%-10s %-3s %-3s %-10s  %-8s %-8s %-8s %-11s %-8s %-8s %-8s %-6s %-6s %-16s' %
                   ("Dataset","T","CV","ML_alg",'prec','rec','f1',"mse","mae","rmse", "r2"  ,"T","t","CDF68    CDF95"))

for num,csv in enumerate(csvs):
    output="./results.csv" #OUTPUT
    test_file=csv.replace('TT','t') # TEST DATA# TEST DATA
    ML(output,csv,test_file,num)
```

    Dataset    T   CV  ML_alg      prec     rec      f1       mse         mae      rmse     r2       T      t      CDF68    CDF95  
    ./TT       0   0   LR          0        0        0        1167041.09  989.28   1080.3   -0.01    2.23   0.01   34.42   40.37   
    ./TT       0   0   DT          0        0        0        24983.24    21.8     158.06   0.98     13.39  0.03   2.47   4.86     
    ./TT       0   0   BR          0        0        0        1168839.83  991.49   1081.13  -0.01    2.95   0.01   34.34   39.97   
    ./TT       0   0   EL          0        0        0        1167041.13  989.28   1080.3   -0.01    2.49   0.01   34.42   40.37   
    ./TT       0   0   twd         0        0        0        1167041.1   989.28   1080.3   -0.01    2.53   0.01   34.42   40.37   
    ./TT       0   0   LAS         0        0        0        1167041.15  989.28   1080.3   -0.01    2.12   0.01   34.42   40.37   
    ./TT       0   0   rcv         0        0        0        1167054.07  989.28   1080.3   -0.01    2.87   0.01   34.42   40.37   
    ./TT       0   0   lcv         0        0        0        1166893.56  991.61   1080.23  -0.01    4.54   0.01   34.37   39.94   
    ./TT       0   0   BAG         0        0        0        8307.35     14.41    91.14    0.99     57.68  0.34   2.2   4.96      
    ./TT       0   0   GBR         0        0        0        9996.79     52.03    99.98    0.99     149.85 0.12   6.76   13.03    
    ./TT       0   0   RF          0        0        0        5502.6      12.68    74.18    1.0      501.61 1.39   2.15   5.24     
    ./TT       0   0   KNN         0        0        0        557629.39   463.57   746.75   0.52     5.2    267.91 22.23   42.1    
    ./TT       0   0   vot         0        0        0        139551.09   341.29   373.57   0.88     761.01 2.17   20.05   23.42   
    ./TT       0   0   stc         0        0        0        519621.29   610.33   720.85   0.55     649.77 333.07 27.57   36.77   
    ./TT       0   0   XGB         0        0        0        9760.45     51.78    98.79    0.99     41.54  0.19   6.81   12.99    
    
