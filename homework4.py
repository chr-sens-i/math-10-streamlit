import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

#1. Title
st.title("Pandas Column Plotter")

#2. Markdown cell with name
st.markdown("[Christopher Hong](https://github.com/mnhkkro)")

#3. File Uploader
uploadedFile = st.file_uploader(label="Upload your .csv file", type="csv")

#4. reads and turns the file into a pd dataframe
if uploadedFile is not None:    
    df = pd.read_csv(uploadedFile)
    
    #5. if x is an empty string, make it numpy NaN
    #otherwise enter x
    df = df.applymap(lambda x: np.nan if x == " " else x)

    #6. make a list of cols that can be made numeric
    def canBeNumeric(c):
        try:
            pd.to_numeric(df[c])
            return True
        except:
            return False
    goodCols = [c for c in df.columns if canBeNumeric(c)]
    st.write(f"{len(goodCols)} columns out of {len(df.columns)} total columns can be plotted.")
    
    #7. makes cols numeric
    df[goodCols] = df[goodCols].apply(pd.to_numeric, axis = 0)
    
    #8. select x and y axes from dataframe
    xAxis = st.selectbox("Choose an x-axis", goodCols)
    yAxis = st.selectbox("Choose a y-axis", goodCols)
    
    #9. Range slider to select rows they want plotted
    mySlider = st.slider("Select the rows you want plotted", 0, len(df.index), (0,len(df.index)))
    df = df.iloc[mySlider[0]:mySlider[1],:]
    
    #10. f-string to show information about chosen data
    st.write(f"The selected columns to be plotted are {xAxis} on the x-axis and {yAxis} on the y-axis, with the rows {mySlider[0]} to {mySlider[1]} being plotted.")
    
    #11. altair chart
    myChart = alt.Chart(df).mark_circle().encode(
        x = xAxis,
        y = yAxis,
        tooltip = [xAxis, yAxis]
        )
    st.altair_chart(myChart)
    
#12 tooltip in altair and displaying how many of the columns of the csv can be plotted