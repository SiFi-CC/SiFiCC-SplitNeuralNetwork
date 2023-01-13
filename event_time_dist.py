import numpy as np
import uproot
import matplotlib.pyplot as plt


def linear(x, m, b):
    return m * x + b


# data collection
rootfile_path = "run00493_single.root"
rootfile = uproot.open(rootfile_path)
events = rootfile[b"events"]
ary_time = events["time"].array()
ary_time -= min(ary_time)

# binning
step = 1e12
upper_lim = int(max(ary_time) / step) + 1
bins = np.arange(0, upper_lim * step, step)
hist_times, _ = np.histogram(ary_time, bins=bins)

for i in range(len(hist_times) - 1):
    if abs(hist_times[i] - hist_times[i + 1]) > hist_times[i] / 2:
        x1 = bins[i+1]
        y1 = hist_times[i]
        idx_low = i
        break


for i in range(len(hist_times) - 1):
    if abs(hist_times[i] - hist_times[i + 1]) > hist_times[i] / 10:
        x2 = bins[i+1]
        y2 = hist_times[i+1]
        idx_high = i + 1

# slope
m = (y2 - y1) / (x2 - x1)
b = y1 - m * x1
times = np.linspace(min(ary_time), max(ary_time), 1000)

# integral sum
integral = hist_times[idx_low+1:idx_high-1]
bin_middles = bins[:-1] + step/2
bin_middles = bin_middles[idx_low+1:idx_high-1]

plt.rcParams.update({'font.size': 18})
plt.figure()
plt.title(rootfile_path)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.xlabel("Time [ps]")
plt.ylabel("Counts per second")
plt.hist(ary_time, bins=bins, histtype=u"step", color="black", label="signal")
plt.plot(times, linear(times, m, b), color="red", label="offset fit")
plt.plot(bin_middles, integral, color="blue", label="integral", alpha=0.7)
plt.tight_layout()
plt.legend()
plt.show()

# final integral
for i in range(len(integral)):
    integral[i] -= int(linear(bin_middles[i], m, b))

list_diff = []
for i in range(len(integral)):
    if integral[i] >= max(integral) * 0.75:
        list_diff.append(integral[i])
print("Integrated event number: ", np.sum(integral))
print("Integral height: {:.2f} +- {:.2f}".format(np.mean(list_diff), np.std(list_diff)))