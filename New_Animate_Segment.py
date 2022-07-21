from itertools import count
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Phidget22.Devices.VoltageRatioInput import *
import keyboard

#Play around with style
plt.style.use('fivethirtyeight')

x_vals = []
y_vals = []
index = count()

plt.plot([], [], label = 'Load Cell')

#r = [0.000023778132175]
r= [0]
def animate(i,output_,r_):
    ch = VoltageRatioInput()
    ch.openWaitForAttachment(1000)
    voltageRatio = ch.getVoltageRatio()
    next_index = next(index)
    x_vals.append(next_index)
    # y_vals.append(random.randint(0,5))
    if keyboard.is_pressed('t'):
        print("Tared")
        r_[0] = ch.getVoltageRatio()

    new_ratio = (voltageRatio - r_[0]) * 33291104.7583953
    #tare_val = ch.getMinVoltageRatio()
    #new_ratio = (voltageRatio - tare_val) * 33291104.7583953

    new_ratio_F = new_ratio * .009806650028638
    y_vals.append(new_ratio_F)

    ax = plt.gca()
    line1, = ax.lines #may change to get_lines
    line1.set_data(x_vals,y_vals)
    
    xlim_low, xlim_high = ax.get_xlim()
    ylim_low, ylim_high = ax.get_ylim()
    
    ax.set_xlim(xlim_low, (max(x_vals) + 5)) #(x_vals.max() + 5))
    y_max = max(y_vals) #y_vals.max()
    y_min = min(y_vals) #y_vals.min()
    ax.set_ylim((y_min - 5), (y_max + 5))

    # if key == ord("q"):
        # ch.close
    # data = [str(next_index), str(new_ratio_F)]
    output_.append(new_ratio_F)
         

output = []

ani = FuncAnimation(plt.gcf(), animate, fargs= [output, r], interval=50)
plt.tight_layout()
plt.show()

#print(r)
header = ['Test','whatever']
with open('testing.csv', 'w', encoding='UTF8') as f:
    f.write(",".join(header) + "\n")
    for x in output:
        f.write((str(x)) + "\n")

    #f.write(str(new_ratio_F)+"\n")


