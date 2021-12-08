import pandas as pd
import altair as alt
from pandas.api.types import is_numeric_dtype
import streamlit as st
import tensorflow as tf
from tensorflow import keras

#Title and where to find dataset
st.title("Analysis of Green Coffee")
st.write("The coffee dataset can be obtained from [Kaggle](https://www.kaggle.com/volpatto/coffee-quality-database-from-cqi?select=merged_data_cleaned.csv).")
st.write("The code for this app can be found on [GitHub](https://github.com/mnhkkro/math-10-streamlit). The file name is math10final.py")
st.header("Description of Dataset")
st.write("This dataset consists of different measures of quality of green coffee and where they originated from. The measures of quality are aroma, flavor, aftertaste, acidity, body, balance, uniformity, clean cup, sweetness. cupper points, and total cup points. This last measure is a sum of the others. Other measures in this dataset are number of defects and moisture.")

#Dataset/PD Dataframe cleaning
cf = pd.read_csv("merged_data_cleaned.csv", na_values = " ")
cf.columns = cf.columns.str.replace('.', ' ') #'.' may cause errors
del cf['unit_of_measurement'] #these columns are unneeded, as there are many missing values and some are in different units
del cf['altitude_low_meters']
del cf['altitude_mean_meters']
del cf['altitude_high_meters']
del cf['Quakers']
cf_list = [x for x in cf.columns if is_numeric_dtype(cf[x])] #creates a list of numeric columns which can be plotted
cf = cf.drop(1310) #outlier that remained, so they are removed manually
cf = cf.drop(1177)
cn = cf[cf_list] #returns a pandas dataframe of only numeric columns
cplot = cf_list
del cplot[0:2] #removes numerical columns that shouldn't be plotted (index and number of bags)
cn = cn[cn.notna().all(axis = 1)] #removes all columns with NaN values, so we are left with numeric data
cf['Country Key']=cf['Country of Origin'] #Here I will create a dictionary and use pandas to convert the countries to keys so I can use them in keras, since strings are not allowed to be the output
clist = cf['Country Key'].tolist() #Makes a list out of the country key column
cset = set(clist) #Turns the list into a set, removing duplicate country values
cnum = list(range(0,36,1)) #There are 36 countries, so 36 keys
ckey = dict(zip(cset,cnum))
cf['Country Key']=cf['Country Key'].map(ckey) #Converts country to key values
cf['Country Key']=cf['Country Key'].map(int) #Turns keys into int values

cu = cn.loc[:,'Aroma':] #Numeric values only

#Altair Plots
st.header("Plotting Measures of Quality")
st.write("First, let's plot the Acidity and Sweetness measures of quality and compare the two species.")

acidS = alt.Chart(cf).mark_square().encode(
    alt.X('Acidity:Q',scale = alt.Scale(zero = False)),
    alt.Y('Sweetness:Q', scale = alt.Scale(zero = False)),
    #size = 'Total Cup Points',
    color = alt.Color('Species', scale=alt.Scale(scheme='set1')),
    tooltip = ['Unnamed: 0','Species','Total Cup Points']
).interactive()

st.altair_chart(acidS, use_container_width=True)

st.write('This chart shows me that Robusta coffee tends to be less sweet than Arabica coffee, but there are a few Arabica entries that are less sweet than the Robusta entries. Comparing acidity, both species appear to be similar.')

st.write("Now let's see if we can find any other correlations between coffee species, meaasures of quality, moisture, and/or defects. Here we will plot each column of the dataset against another column.")

#Displays what columns can be plotted
xplot = st.selectbox("Select a column to be plotted along the x-axis", cplot, index=0)
yplot = st.selectbox("Select a column to be plotted along the y-axis", cplot, index=3)

chart2 = alt.Chart(cf).mark_circle().encode(
    alt.X(xplot,scale = alt.Scale(zero = False)),
    alt.Y(yplot, scale = alt.Scale(zero = False)),
    color = alt.Color('Species', scale=alt.Scale(scheme='dark2')),
    tooltip = ['Unnamed: 0','Species']
).interactive()

st.altair_chart(chart2, use_container_width=True)

st.write("Below are graphs of every column plotted against a different column of the dataset, allowing us to quickly see if anything stands out in the graphs. If anything does, we can use the plotter above to obtain a closer look.")

multchart = alt.Chart(cf).mark_point().encode(
    alt.X(alt.repeat("column"), type='quantitative',scale = alt.Scale(zero = False)),
    alt.Y(alt.repeat("row"), type='quantitative',scale = alt.Scale(zero = False)),
    color = 'Species',
    tooltip = ['Unnamed: 0','Species']
    ).properties(
    width=200,
    height=200
).repeat(
    row=cplot,
    column=cplot
)
with st.expander("Display All Possible Graphs (May be laggy)"):
    st.altair_chart(multchart)

st.write("From the comparing the data using the graphs above, we see that each of the measures of quality has a positive correlation with total cup points, which makes sense, since the total cup points are a sum of the measures of quality. Aside from this, Robusta beans seem to consistently score higher in the clean cup category and the uniformity category, while also scoring lower in sweetness. However, there are relatively few Robusta entries in the data set, so this may just be chance. In the other categories, Robusta coffee beans score around the median of each category, with there being Arabica beans that perform both better and worse in these categories. So it appears that Arabica coffee beans and Robusta coffee beans are rather similar, so there is no clear connection between the measures of quality and the species of coffee.")



#Machine Learning
st.header("Neural Network")
cu = cn.loc[:,'Aroma':]
X_train = cu
y_train = cf["Country Key"]
y_train2 = (cf["Species"]=='Arabica')

#Model for measurements of quality and country of origin
model = keras.Sequential(
        [
            keras.layers.InputLayer(input_shape =(14,)),
            keras.layers.Dense(24,activation ='relu'),
            keras.layers.Dense(24,activation ='relu'),
            keras.layers.Dense(36,activation='softmax')
        ]
    )
    
model.compile(loss="sparse_categorical_crossentropy", 
                  optimizer=keras.optimizers.SGD(
                        learning_rate=0.01
                    ),
                  metrics=["accuracy"])

#Model for measurements of quality and species
model2 = keras.Sequential(
    [
     keras.layers.InputLayer(input_shape = (14,)),
     keras.layers.Dense(16,activation = 'sigmoid'),
     keras.layers.Dense(16,activation = 'sigmoid'),
     keras.layers.Dense(1,activation = 'sigmoid')
    ]
)

model2.compile(loss="binary_crossentropy",
               optimizer=keras.optimizers.SGD(
                   learning_rate=0.01)
               ,
               metrics=["accuracy"])

#makes it so that the model isn't run upon changing the selectbox
if 'fit1' not in st.session_state:
        
    with st.spinner('Training Model'):
        fit1 = model.fit(X_train,y_train,epochs=100, validation_split = 0.2, verbose = False)
        
    st.session_state['fit1'] = fit1
    st.success('Done!')
    
    
    with st.spinner('Training Model'):
        fit2 = model2.fit(X_train,y_train2,epochs=50, validation_split=0.2,verbose=False)
        
    st.session_state['fit2'] = fit2
    st.success('Done!')
else:
    fit1 = st.session_state['fit1']
    fit2 = st.session_state['fit2']
    
#Species NN
st.subheader("Differentiating Coffee Species")
st.write("Let's see if we can use a neural network to identify coffee species by qualities of measurement.")
st.write("I will be using a neural network with 14 input variables, two hidden layers with 16 neurons, and an output layer with one option, using binary crossentropy. They will all use sigmoid activation")
st.write("Using this model, the loss value on our final iteration is "+str(fit2.history['loss'][-1])+", but our validation loss value is "+str(fit2.history['val_loss'][-1])+".")
st.write("This shows that our neural network is overfitted, which is probably because there are too few Robusta entries, and those that we have are too similar to the Arabica coffee, so we cannot conclude anything about how coffee species may correlate with different measures of quality.")
st.subheader("Diffentiating Countries of Origin using Measures of Quality")
st.write("Now we will use a neural network to see if we can predict the country of origin from the measures of quality.")
st.write("We will be using 24 neurons for the hidden layers this time, of which there are 2. These will have relu activation. The output layer will have 36 neurons, as there are 36 different countries of origin, and will use softmax activation.")
st.write("Our loss value on the final iteration is "+str(fit1.history['loss'][-1])+", and our validation loss value is "+str(fit1.history['val_loss'][-1])+".")
st.write("Our accuracy value on the final iteration is "+str(fit1.history['accuracy'][-1])+", and our validation accuracy value is "+str(fit1.history['val_accuracy'][-1])+".")
st.write("This shows that our model is slightly overfitted, as the two loss values are still slightly different, but our accuracy is still low, so even if our data is only slightly overfitted, the accuracy is too low to be able to consistently choose the correct country of origin from the measures of quality.")

st.header("References")
st.write("Multiple Altair Charts: https://altair-viz.github.io/user_guide/compound_charts.html#repeated-charts")
st.write("Converting 2 Lists into a Single Dictionary: https://www.tutorialspoint.com/convert-two-lists-into-a-dictionary-in-python")
