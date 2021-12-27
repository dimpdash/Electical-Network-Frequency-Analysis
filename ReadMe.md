## _Electrical Network Frequnecy Analysis_
# Background
Electrical network frequency analysis analyzes the frequency of hums from AC power captured during multimedia recordings. Once a frequency of a time period is identified it can be convolved with an existing repository with times to timestamp the data.
# Introduction
This is a simple python app that currenlty produces a the ENF signal from a audio clip. The current process using short time fourier transform to identify the frequencies. Additionally the higher order modes are also analyzed and averaged to reduce the effect of noise 

![alt text](https://raw.githubusercontent.com/dimpdash/Electical-Network-Frequency-Analysis/master/images/Recorded_Data_1.png)
*The electrical networks frequency variation over time as captured from audio*

The audio clip "Record_Data_1.wav" was taken from https://github.com/SaminYeasar/ENF-Extraction-of-Bangladesh-Grid for analysis purposes. 

## License