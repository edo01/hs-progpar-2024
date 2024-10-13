import matplotlib.pyplot as plt
import pandas as pd

# Organizing the data into a dictionary
data = {
    "CPU": ["cortex", "cortex", "cortex", "cortex", "denver", "denver","full_system", "full_system"],
    "Threads": [1, 2, 3, 4, 1, 2, 5, 6],
    "Time": [12.681, 6.950, 5.206, 5.155, 9.207, 5.058, 2.815, 2.760],
	"Speedup": [1, 1.824, 2.435, 2.49, 1, 1.820 , 4.504, 4.594],
	"Efficiency": [1, 0.912, 0.81, 0.622, 1, 0.91, 0.9, 0.750]
    
}

#data = {
#    "CPU": ["cortex", "cortex", "cortex", "cortex", "denver", "denver","full_system", "full_system"],
#    "Threads": [1, 2, 3, 4, 1, 2, 5, 6],
#    "Time": [533.469, 271.465, 208.882, 143.754, 270.261, 152.505, 138.832, 148.999],
#	"Speedup": [1, 1.963, 2.55, 3.71, 1, 1.77, 3.85, 3.56],
#	"Efficiency": [1, 0.981, 0.85, 0.927, 1, 0.885, 0.77, 0.593] 
#}

# Creating a DataFrame
df = pd.DataFrame(data)

# Creating the line graph with separate lines for different CPUs
plt.figure(figsize=(10, 6))

# Plotting Cortex data
cortex_data = df[df["CPU"] == "cortex"]
#plt.plot(cortex_data["Threads"], cortex_data["Time"], marker='o', linestyle='-', color='b', label="Cortex time")
plt.plot(cortex_data["Threads"], cortex_data["Speedup"], marker='.', linestyle='-', color='b', label="Cortex speedup")
plt.plot(cortex_data["Threads"], cortex_data["Efficiency"], marker='+', linestyle='--', color='b', label="Cortex efficiency")


# Plotting Denver data (in a different color)
denver_data = df[df["CPU"] == "denver"]
#plt.plot(denver_data["Threads"], denver_data["Time"], marker='o', linestyle='-', color='g', label="Denver time")
plt.plot(denver_data["Threads"], denver_data["Speedup"], marker='.', linestyle='-', color='g', label="Denver speedup")
plt.plot(denver_data["Threads"], denver_data["Efficiency"], marker='+', linestyle='--', color='g', label="Denver efficiency")


# Plotting full_system data
full_system_data = df[df["CPU"] == "full_system"]
#plt.plot(full_system_data["Threads"], full_system_data["Time"], marker='o', linestyle='-', color='r', label="Full system time")
plt.plot(full_system_data["Threads"], full_system_data["Speedup"], marker='.', linestyle='-', color='r', label="Full system speedup")
plt.plot(full_system_data["Threads"], full_system_data["Efficiency"], marker='+', linestyle='--', color='r', label="Full system efficiency")

plt.axhline(y=1, color='black', linestyle=':', label="Desired efficiency for strong scalability")

# Adding titles, labels, and legend
plt.title("Mandel: Scalability Test with Tile: 1024*64 and Scheduling: Dynamic,3")
plt.xlabel("Threads")
plt.ylabel("Speedup and Efficiency")
plt.legend()

# Display the line graph
plt.show()
