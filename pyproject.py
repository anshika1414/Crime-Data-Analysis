import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style for compelling visuals
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 16, 'axes.labelsize': 12})

# Load the dataset
df = pd.read_csv("Crime_Data.csv")

# --- Data Cleaning ---
df["Mocodes"].fillna("Null", inplace=True)
df["Vict Sex"].fillna("Unknown", inplace=True)
df["Vict Descent"].fillna("Unknown", inplace=True)
df["Premis Desc"].fillna("Unknown", inplace=True)
df["Weapon Desc"].fillna("Unknown", inplace=True)
df["Weapon Used Cd"].fillna("N/A", inplace=True)
df["Crm Cd 1"].fillna("N/A", inplace=True)
df["Crm Cd 2"].fillna("N/A", inplace=True)
df["Crm Cd 3"].fillna("N/A", inplace=True)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# OBJECTIVE 1: Analyze and Visualize Top 10 Crime Categories
crime_counts = df['Crm Cd Desc'].value_counts()
top_crimes = crime_counts.head(10)

short_labels = {
    'VEHICLE - STOLEN': 'Stolen Vehicle',
    'BATTERY - SIMPLE ASSAULT': 'Battery Assault',
    'BURGLARY FROM VEHICLE': 'Burglary (Vehicle)',
    'VANDALISM - FELONY ($400 & OVER, ALL CHURCH VANDALISMS)': 'Felony Vandalism',
    'ASSAULT WITH DEADLY WEAPON, AGGRAVATED ASSAULT': 'Aggr. Assault w/ Weapon',
    'INTIMATE PARTNER - SIMPLE ASSAULT': 'IP Assault',
    'BURGLARY': 'Burglary',
    'THEFT PLAIN - PETTY ($950 & UNDER)': 'Petty Theft',
    'THEFT FROM MOTOR VEHICLE - PETTY ($950 & UNDER)': 'Theft from Vehicle',
    'VANDALISM - MISDEMEANOR ($399 OR UNDER)': 'Misdemeanor Vandalism'
}
top_crimes.index = top_crimes.index.map(short_labels)

# Top 10 Crime Categories Bar Plot (Enhanced)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_crimes.values, y=top_crimes.index, palette="cubehelix", width=0.5)
plt.title("Top 10 Crime Categories", fontsize=20, weight='bold')
plt.xlabel("Number of Crimes", fontsize=16)
plt.ylabel("Crime Type", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14, fontweight='bold')
plt.tight_layout()
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.show()

# OBJECTIVE 2: Time-Series Analysis of Crime Trends by Month and Year
df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], errors='coerce')
df['Year'] = df['DATE OCC'].dt.year
df['Month'] = df['DATE OCC'].dt.month
df['Day'] = df['DATE OCC'].dt.day

df_filtered = df[(df['Year'] >= 2000) & (df['Year'] <= pd.Timestamp.now().year)]
crime_trends_by_month = df.groupby('Month').size()

# Enhanced Time Series Analysis (Average Crime Trends by Month)
plt.figure(figsize=(10, 6))
sns.barplot(x=crime_trends_by_month.index, y=crime_trends_by_month.values, palette="crest", width=0.7)
plt.title("Average Crime Trends by Month", fontsize=20, weight='bold')
plt.xlabel("Month", fontsize=16)
plt.ylabel("Number of Crimes", fontsize=16)
plt.xticks(ticks=np.arange(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=14)
plt.yticks(fontsize=14, fontweight='bold')
plt.tight_layout()

# Adding gridlines for a more refined look
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.show()

# OBJECTIVE 3: Crime Hotspot Detection Using Grid-Based Heatmap
lat_bins = np.linspace(df["LAT"].min(), df["LAT"].max(), 4)
lon_bins = np.linspace(df["LON"].min(), df["LON"].max(), 4)

crime_density, _, _ = np.histogram2d(df["LAT"], df["LON"], bins=[lat_bins, lon_bins])

# Enhanced Heatmap with Light Shades
plt.figure(figsize=(10, 6))
plt.imshow(crime_density.T, cmap='Blues', origin='lower', aspect='auto',
           extent=[df["LON"].min(), df["LON"].max(), df["LAT"].min(), df["LAT"].max()])
plt.colorbar(label='Crime Density')
plt.title("Crime Hotspots (Enhanced Heatmap)", fontsize=20, weight='bold')
plt.xlabel("Longitude", fontsize=16)
plt.ylabel("Latitude", fontsize=16)
plt.tight_layout()

# Adding gridlines and annotations for better clarity
plt.grid(True, linestyle='-', color='black', alpha=0.3)
plt.show()

# OBJECTIVE 4: Victim Age Distribution by Crime Type and Gender
df_valid = df[(df['Vict Age'] > 0) & (df['Vict Age'] <= 100)]
top5_crimes = df_valid['Crm Cd Desc'].value_counts().head(5).index
df_top5 = df_valid[df_valid['Crm Cd Desc'].isin(top5_crimes)]

# Stripplot for Victim Age Distribution by Crime Type and Gender
plt.figure(figsize=(12, 6))
sns.stripplot(data=df_top5, x='Crm Cd Desc', y='Vict Age', hue='Vict Sex', 
              jitter=True, dodge=True, alpha=0.7, palette='coolwarm')
plt.title("Victim Age Distribution by Crime Type and Gender", fontsize=20, weight='bold')
plt.xlabel("Crime Type", fontsize=16)
plt.ylabel("Victim Age", fontsize=16)
plt.xticks(rotation=25, fontsize=14)
plt.legend(title='Victim Sex', fontsize=14)
plt.tight_layout()
plt.show()

# OBJECTIVE 5: Analyze Top 5 Weapons Used in Crimes
weapon_counts = df['Weapon Desc'].value_counts().head(5)

# Enhanced Pie Chart (Exploded View & Color Palette)
plt.figure(figsize=(8, 8))  # Increased size of the pie chart
explode = (0.1, 0.1, 0.1, 0.1, 0.1)  # Exploding each slice slightly

# Create the pie chart without labels, displaying only percentages
colors = sns.color_palette("Spectral", n_colors=len(weapon_counts))  # Ensure correct number of colors
plt.pie(weapon_counts, autopct='%1.1f%%', startangle=120, colors=colors, 
        textprops={'fontsize': 12, 'fontweight': 'bold'}, explode=explode, labels=None, pctdistance=0.85) 

plt.title("Top 5 Weapons Used in Crimes", fontsize=18, weight='bold')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

# Create a color-coded legend on the right side
labels = weapon_counts.index.tolist()  # Convert Index to list of labels
plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) 
                    for color in colors], labels=labels, title="Weapon Descriptions", loc='center left', 
           bbox_to_anchor=(1.05, 0.5), fontsize=12)

plt.tight_layout()
plt.show()

# OBJECTIVE 6: Victim Age Distribution (Histogram)
df_valid = df[(df['Vict Age'] > 0) & (df['Vict Age'] <= 100)]

# Histogram for Victim Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df_valid['Vict Age'], bins=20, kde=True, color="green")
plt.title("Victim Age Distribution", fontsize=20, weight='bold')
plt.xlabel("Age", fontsize=16)
plt.ylabel("Frequency", fontsize=16)
plt.tight_layout()
plt.show()

# OBJECTIVE 7: Analyze Crime Distribution by Status Description 
status_desc_counts = df['Status Desc'].value_counts()

# Bar Plot for Crime Distribution by Status Description
plt.figure(figsize=(10, 6))
status_desc_counts.plot(kind='bar', stacked=True, color=sns.color_palette("Set2"))
plt.title("Distribution of Crimes by Status Description", fontsize=20, weight='bold')
plt.xlabel("Status", fontsize=16)
plt.ylabel("Number of Crimes", fontsize=16)
plt.xticks(rotation=0, fontsize=14)
plt.tight_layout()
plt.show()

# OBJECTIVE 8: Monthly Crime Trends by Area in 2020
df['Month'] = df['DATE OCC'].dt.strftime('%b')
df_2020 = df[df['Year'] == 2020]
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
df_2020['Month'] = pd.Categorical(df_2020['Month'], categories=month_order, ordered=True)

monthly_crimes = df_2020.groupby(['Month', 'AREA NAME']).size().unstack(fill_value=0)

# Lineplot for Monthly Crime Trends by Area (2020)
plt.figure(figsize=(10, 8))
for area in monthly_crimes.columns:
    sns.lineplot(y=monthly_crimes.index, x=monthly_crimes[area], label=area, marker='o')

plt.title("Monthly Crime Trends by Area (2020, Horizontal View)", fontsize=20, weight='bold')
plt.xlabel("Number of Crimes", fontsize=16)
plt.ylabel("Month", fontsize=16)
plt.legend(title="Area", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
plt.tight_layout()
plt.show()

# Save statistics to CSV
df.to_csv("crime_data_cleaned.csv")
print("-----Analysis Done!-------")
