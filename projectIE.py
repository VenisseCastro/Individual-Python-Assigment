import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as pyo


# create data:
df = pd.read_csv('Admission.csv')
pyo.plot([{
    'x': df.index,
    'y': df[col],
    'name': col
} for col in df.columns])

print(df['Admission Chance'].describe())
# Create the distribution plot
sns.distplot(df['Admission Chance'],kde=False)
sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True);

#g = sns.PairGrid(df)
#g.map(sns.scatterplot)
numeric_columns = df.select_dtypes(exclude=['object'])

g = sns.PairGrid(df, x_vars=['Admission Chance'], y_vars=numeric_columns.columns, diag_sharey=False)
g.map(sns.scatterplot)
# Create the scatter plot
sns.set(style="whitegrid")  # Optional: Set the plot style
sns.scatterplot(x=df['Admission Chance'], y=df['LOR '])

# title & labels
plt.title("Scatter Plot")
plt.xlabel("Admission Chance")
plt.ylabel("LOR ")

# Show the plot
plt.show()
# Create the scatter plot
sns.set(style="whitegrid")  # Optional: Set the plot style
sns.scatterplot(x=df['Admission Chance'], y=df['SOP'])

# title & labels
plt.title("Scatter Plot")
plt.xlabel("Admission Chance")
plt.ylabel("SOP")

# Show the plot
plt.show()
# Create the scatter plot
sns.set(style="whitegrid")  # Optional: Set the plot style
sns.scatterplot(x=df['Admission Chance'], y=df['Research'])

# title & labels
plt.title("Scatter Plot")
plt.xlabel("Admission Chance")
plt.ylabel("Research")

# Show the plot
plt.show()

St.header ('Research')
sex_dist=df['Research'].value_counts()
st.dataframe(sex_dist)
with st.continer():
    fig,ax=plt.subplot()
    ax.pie(sex_dist,autopct='%0.2%%',labels=['Male','Female'])
    st.pyplot(fig)

fig, ax = plt.subplot
ax.bar(sex_dist.index, sex_dist)
st.pytplot(fig)

