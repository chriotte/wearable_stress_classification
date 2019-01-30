# Stress classifier with AutoML

This repository presents an automated machine learning approach in Python to create a stress monitoring system with data from devices such as fitness trackers. With the rising popularity of trackers that monitors biological signals 24/7, there is just a matter of time before the technology matures and everyone will be wearing their own ‘doctor AI’ on the wrist, this project is one step in that direction.

Note: This code is a simplified version of my master's dissertation done during the summer of 2017. For more information about data handling, and other machine learning approaches, please see the full masters dissertation available [here](https://1drv.ms/b/s!ApqYcVCNnKvChu1nfGu5bMnh8jckkw).

[Code tutorial and data description can be found at my blog dataespresso.com](http://dataespresso.com/).


![dataset](images/dataoverview.png)




Overview

Dataset info
Number of variables 	23
Number of observations 	4129
Total Missing (%) 	5.4%
Total size in memory 	742.0 KiB
Average record size in memory 	184.0 B

Variables types
Numeric 	18
Categorical 	0
Boolean 	0
Date 	0
Text (Unique) 	0
Rejected 	5
Unsupported 	0

Warnings

    AVNN is highly correlated with interval in seconds (ρ = 0.99999) Rejected
    EMG has 106 / 2.6% missing values Missing
    HF has 127 / 3.1% missing values Missing
    HF has 3944 / 95.5% zeros Zeros
    LF has 127 / 3.1% missing values Missing
    LF has 3944 / 95.5% zeros Zeros
    LF_HF has 4071 / 98.6% missing values Missing
    RMSSD is highly correlated with SDNN (ρ = 0.98199) Rejected
    SDNN has 122 / 3.0% missing values Missing
    TP is highly correlated with RMSSD (ρ = 0.94593) Rejected
    ULF is highly correlated with TP (ρ = 0.99843) Rejected
    VLF has 127 / 3.1% missing values Missing
    VLF has 3701 / 89.6% zeros Zeros
    handGSR has 73 / 1.8% missing values Missing
    marker has 407 / 9.9% missing values Missing
    pNN50 has 312 / 7.6% zeros Zeros
    stress has 1121 / 27.1% zeros Zeros
    time is highly correlated with Seconds (ρ = 1) Rejected

Variables

AVNN
Highly correlated

This variable is highly correlated with interval in seconds and should be ignored for analysis
Correlation 	0.99999

ECG
Numeric
Distinct count 	4013
Unique (%) 	97.2%
Missing (%) 	0.0%
Missing (n) 	0
Infinite (%) 	0.0%
Infinite (n) 	0
Mean 	0.17093
Minimum 	-0.69959
Maximum 	0.68171
Zeros (%) 	0.0%
Toggle details

EMG
Numeric
Distinct count 	3363
Unique (%) 	81.4%
Missing (%) 	2.6%
Missing (n) 	106
Infinite (%) 	0.0%
Infinite (n) 	0
Mean 	0.60448
Minimum 	-0.6978
Maximum 	9.4917
Zeros (%) 	0.1%
Toggle details

HF
Numeric
Distinct count 	59
Unique (%) 	1.4%
Missing (%) 	3.1%
Missing (n) 	127
Infinite (%) 	0.0%
Infinite (n) 	0
Mean 	1.0975e-06
Minimum 	0
Maximum 	0.00026121
Zeros (%) 	95.5%
Toggle details

HR
Numeric
Distinct count 	3292
Unique (%) 	79.7%
Missing (%) 	0.0%
Missing (n) 	0
Infinite (%) 	0.0%
Infinite (n) 	0
Mean 	83.136
Minimum 	11.8
Maximum 	372
Zeros (%) 	0.0%
Toggle details

LF
Numeric
Distinct count 	60
Unique (%) 	1.5%
Missing (%) 	3.1%
Missing (n) 	127
Infinite (%) 	0.0%
Infinite (n) 	0
Mean 	3.3789e-06
Minimum 	0
Maximum 	0.00061584
Zeros (%) 	95.5%
Toggle details

LF_HF
Numeric
Distinct count 	59
Unique (%) 	1.4%
Missing (%) 	98.6%
Missing (n) 	4071
Infinite (%) 	0.0%
Infinite (n) 	0
Mean 	3.5557
Minimum 	0.41853
Maximum 	9.1172
Zeros (%) 	0.0%
Toggle details

NNRR
Numeric
Distinct count 	21
Unique (%) 	0.5%
Missing (%) 	0.0%
Missing (n) 	0
Infinite (%) 	0.0%
Infinite (n) 	0
Mean 	0.97538
Minimum 	0.97059
Maximum 	0.98148
Zeros (%) 	0.0%
Toggle details

RESP
Numeric
Distinct count 	4121
Unique (%) 	99.8%
Missing (%) 	0.0%
Missing (n) 	0
Infinite (%) 	0.0%
Infinite (n) 	0
Mean 	29.847
Minimum 	-12.606
Maximum 	52.09
Zeros (%) 	0.0%
Toggle details

RMSSD
Highly correlated

This variable is highly correlated with SDNN and should be ignored for analysis
Correlation 	0.98199

SDNN
Numeric
Distinct count 	850
Unique (%) 	20.6%
Missing (%) 	3.0%
Missing (n) 	122
Infinite (%) 	0.0%
Infinite (n) 	0
Mean 	0.057812
Minimum 	7.84e-09
Maximum 	0.16458
Zeros (%) 	0.0%
Toggle details

Seconds
Numeric
Distinct count 	4129
Unique (%) 	100.0%
Missing (%) 	0.0%
Missing (n) 	0
Infinite (%) 	0.0%
Infinite (n) 	0
Mean 	2278.8
Minimum 	12.53
Maximum 	5005.4
Zeros (%) 	0.0%
Toggle details

TP
Highly correlated

This variable is highly correlated with RMSSD and should be ignored for analysis
Correlation 	0.94593

ULF
Highly correlated

This variable is highly correlated with TP and should be ignored for analysis
Correlation 	0.99843

VLF
Numeric
Distinct count 	302
Unique (%) 	7.3%
Missing (%) 	3.1%
Missing (n) 	127
Infinite (%) 	0.0%
Infinite (n) 	0
Mean 	0.00072777
Minimum 	0
Maximum 	0.035841
Zeros (%) 	89.6%
Toggle details

footGSR
Numeric
Distinct count 	4125
Unique (%) 	99.9%
Missing (%) 	0.0%
Missing (n) 	0
Infinite (%) 	0.0%
Infinite (n) 	0
Mean 	6.4877
Minimum 	0.97111
Maximum 	22.582
Zeros (%) 	0.0%
Toggle details

handGSR
Numeric
Distinct count 	4047
Unique (%) 	98.0%
Missing (%) 	1.8%
Missing (n) 	73
Infinite (%) 	0.0%
Infinite (n) 	0
Mean 	11.523
Minimum 	-28.382
Maximum 	31.22
Zeros (%) 	0.0%
Toggle details

interval in seconds
Numeric
Distinct count 	782
Unique (%) 	18.9%
Missing (%) 	0.0%
Missing (n) 	0
Infinite (%) 	0.0%
Infinite (n) 	0
Mean 	0.78844
Minimum 	0.52794
Maximum 	1.0401
Zeros (%) 	0.0%
Toggle details

marker
Numeric
Distinct count 	3585
Unique (%) 	86.8%
Missing (%) 	9.9%
Missing (n) 	407
Infinite (%) 	0.0%
Infinite (n) 	0
Mean 	12.057
Minimum 	0
Maximum 	35.624
Zeros (%) 	0.1%
Toggle details

newtime
Numeric
Distinct count 	4129
Unique (%) 	100.0%
Missing (%) 	0.0%
Missing (n) 	0
Infinite (%) 	0.0%
Infinite (n) 	0
Mean 	35172
Minimum 	12.53
Maximum 	70174
Zeros (%) 	0.0%
Toggle details

pNN50
Numeric
Distinct count 	107
Unique (%) 	2.6%
Missing (%) 	0.0%
Missing (n) 	0
Infinite (%) 	0.0%
Infinite (n) 	0
Mean 	0.034246
Minimum 	0
Maximum 	0.25714
Zeros (%) 	7.6%
Toggle details

stress
Numeric
Distinct count 	349
Unique (%) 	8.5%
Missing (%) 	0.0%
Missing (n) 	0
Infinite (%) 	0.0%
Infinite (n) 	0
Mean 	0.51608
Minimum 	0
Maximum 	1
Zeros (%) 	27.1%
Toggle details

time
Highly correlated

This variable is highly correlated with Seconds and should be ignored for analysis
Correlation 	1
Correlations
Sample
	ECG 	EMG 	HR 	RESP 	Seconds 	footGSR 	handGSR 	interval in seconds 	marker 	newtime 	stress 	time 	NNRR 	AVNN 	SDNN 	RMSSD 	pNN50 	TP 	ULF 	VLF 	LF 	HF 	LF_HF
0 	-0.001974 	-0.004737 	77.815789 	10.801842 	12.529684 	2.417132 	10.889447 	0.614632 	NaN 	12.529684 	0.0 	12.529684 	0.973684 	0.617297 	3.558630e-02 	0.015203 	0.055556 	0.001238 	0.0 	0.000696 	0.000407 	0.000135 	3.00200
1 	0.002935 	-0.004457 	101.978261 	10.750609 	30.503500 	2.417109 	11.251065 	0.647826 	NaN 	30.503500 	0.0 	30.503500 	0.978261 	0.647889 	1.354660e-02 	0.013858 	0.045455 	0.000144 	0.0 	0.000009 	0.000060 	0.000075 	0.79371
2 	0.006745 	-0.003426 	104.957447 	10.557234 	52.523021 	2.226872 	11.379638 	0.646383 	NaN 	52.523021 	0.0 	52.523021 	0.978723 	0.645000 	2.240000e-08 	0.000000 	0.000000 	NaN 	0.0 	NaN 	NaN 	NaN 	NaN
3 	-0.004043 	-0.002532 	87.702128 	10.640128 	74.402170 	2.173021 	11.470830 	0.645000 	NaN 	74.402170 	0.0 	74.402170 	0.978723 	0.645000 	2.240000e-08 	0.000000 	0.000000 	NaN 	0.0 	NaN 	NaN 	NaN 	NaN
4 	0.012745 	-0.004426 	88.829787 	10.699319 	96.219617 	2.017106 	11.135255 	0.645000 	NaN 	96.219617 	0.0 	96.219617 	0.978723 	0.645000 	2.240000e-08 	0.000000 	0.000000 	NaN 	0.0 	NaN 	NaN 	NaN 	NaN




