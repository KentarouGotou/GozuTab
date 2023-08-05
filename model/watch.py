"""
2020/09/27
HHiromasa 
Helper script to plot the memory usage of GPUs
* This script was designed for a 4 GPU server. Some alterations may be needed in order for it to run properly on other GPU servers.
* WARNING! this script uses "while True:". Only run script when it is terminatable.
"""

import os
import sys
import time
import subprocess
import matplotlib.pyplot as plt

#nvidia-smi --display=MEMORY -q
def get_values():
    bashCmd = ["nvidia-smi", "--display=MEMORY","-q"]
    process = subprocess.Popen(bashCmd, stdout=subprocess.PIPE)

    out, error = process.communicate()
    FB_total = out.split()[28]
    FB_used_0 = out.split()[32]
    FB_used_1 = out.split()[64]
    FB_used_2 = out.split()[96]
    FB_used_3 = out.split()[128]

    #convert from byte > string > int. This is simpler to implement than doing byte > int.
    FB_total = int(FB_total.decode("utf-8")) 
    FB_used_0 = int(FB_used_0.decode("utf-8")) 
    FB_used_1 = int(FB_used_1.decode("utf-8"))
    FB_used_2 = int(FB_used_2.decode("utf-8"))
    FB_used_3 = int(FB_used_3.decode("utf-8"))


    return FB_total,FB_used_0,FB_used_1,FB_used_2,FB_used_3


FB_used_0s = []
FB_used_1s = []
FB_used_2s = []
FB_used_3s = []

is_first_iter = True
while True:
    #get the FB values
    FB_total,FB_used_0,FB_used_1,FB_used_2,FB_used_3 = get_values()
    FB_used_0s.append(FB_used_0)
    FB_used_1s.append(FB_used_1)
    FB_used_2s.append(FB_used_2)
    FB_used_3s.append(FB_used_3)
    seconds = list(range(len(FB_used_0s)))

    #plot the data and overwrite it to gpu_plot.png
    plt.plot(seconds,FB_used_0s,'r--',label='GPU 0')
    plt.plot(seconds,FB_used_1s,'b--',label='GPU 1')
    plt.plot(seconds,FB_used_2s,'g--',label='GPU 2')
    plt.plot(seconds,FB_used_3s,'c--',label='GPU 3')
    if is_first_iter:
        plt.legend()
        is_first_iter = False
    plt.title(f"GPU Memory (max {FB_total}MiB)")
    plt.ylabel('FB MEMORY USAGE (MiB)')
    plt.xlabel('TIME (seconds)')
    #plt.ylim(top = FB_total) #uncomment this to set the limit to the max FB memory limit
    plt.savefig('gpu_plot.png',dpi=300)
    
    
    print(f"plotted. length is {seconds[-1]}")
    time.sleep(1) #sleep for a second to avoid running like crazy.