# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 12:25:32 2023

@author: Steve
"""

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency
from scipy.stats import probplot
from plotly.subplots import make_subplots
from PIL import Image
import plotly.express as px
import math
import plotly
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer
from sklearn.metrics import precision_score,recall_score,accuracy_score,f1_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
colors = plotly.colors.qualitative.D3
st.set_option('deprecation.showPyplotGlobalUse', False)
sns.set_style('darkgrid')
# Plans:
# Overview - Why do we care? Variables overview, Is it really obvious?
# Basic checks - Sampling data and possible values, Null values and imputation(Hooray!), Outlier treatment
# Approach and intuition - Process flow, Separation between variables, Quantification 
# Univariate Analysis- Quantitative - Overall picture, Playground, Develop hypotheses, Test the hypotheses
# Univariate Analysis- Categorical - Overall picture, Playground, Develop hypotheses, Test the hypotheses
# Bivariate Analysis- Quantitative - Overall picture, Playground, Develop hypotheses, Test the hypotheses
# Bivariate Analysis- Categorical - Overall picture, Playground, Develop hypotheses, Test the hypotheses
# Summary- Overall results, Next steps, Applications (Porbability threshold slider)

df=pd.read_csv('heart.csv')
quantitative_cols=[i for i in df.select_dtypes(include=[np.int64,np.float64]) if df[i].nunique()>5 and i != 'HeartDisease']
categorical_cols=[i for i in df.select_dtypes(include=[object,np.int64]) if df[i].nunique()<=5 and i != 'HeartDisease']
hyp2=[]
hyp=[]
df_outliers=pd.read_csv('heart.csv') ######################################################without outlier treatment
##############Outlier Treatment###########################
df_en=df.copy()

df_en.Sex.replace({'M':1,'F':0},inplace=True)
df_en.ChestPainType.replace({'ATA':0,'NAP':1,'ASY':2,'TA':3},inplace=True)
df_en.RestingECG.replace({'Normal':0,'ST':1,'LVH':2},inplace=True)
df_en.ExerciseAngina.replace({'Y':1,'N':0},inplace=True)
df_en.ST_Slope.replace({'Up':0,'Flat':1,'Down':2},inplace=True)
df_en.Cholesterol.replace({0:np.NaN},inplace=True)
df_en.RestingBP.replace({0:np.NaN},inplace=True)

scaler = MinMaxScaler()
df2 = pd.DataFrame(scaler.fit_transform(df_en), columns = df_en.columns)

imputer = KNNImputer(n_neighbors=5)
df3 = pd.DataFrame(imputer.fit_transform(df2),columns = df2.columns)

df4=pd.DataFrame(scaler.inverse_transform(df3),columns=df2.columns) ##############outlier treated

df4.Sex.replace({1:'M',0:'F'},inplace=True)
df4.ChestPainType.replace({0:'ATA',1:'NAP',2:'ASY',3:'TA'},inplace=True)
df4.RestingECG.replace({0:'Normal',1:'ST',2:'LVH'},inplace=True)
df4.ExerciseAngina.replace({1:'Y',0:'N'},inplace=True)
df4.ST_Slope.replace({0:'Up',1:'Flat',2:'Down'},inplace=True)
################################################################
df=df4.copy()
df.FastingBS=df.FastingBS.astype(object)
df.HeartDisease=df.HeartDisease.astype(int)
def order_df(df_input, order_by, order):
    df_output=pd.DataFrame()
    for var in order:    
        df_append=df_input[df_input[order_by]==var].copy()
        df_output = pd.concat([df_output, df_append])
    return(df_output)

with st.sidebar:
    selected = option_menu("Content", ["Overview", "Basic Checks","EDA-distributions",'EDA-relationships',"Further exploration","Univariate Analysis-Quantitative",
                                       "Univariate Analysis-Categorical","Summary"],
                          default_index=0,
                          orientation="vertical",
                          icons=['house','ui-checks-grid','file-bar-graph','people','binoculars','123','badge-vo','flag'],
                          menu_icon='cast',
                          styles={"nav-link": {"font-size": "15px", "text-align": "centre", "margin": "0px", 
                                                "--hover-color": "#0068c9"},
                                   "icon": {"font-size": "15px"},
                                   "container" : {"max-width": "3500px"},
                                   "nav-link-selected": {"background-color": "#ff2b2b"}})
    
if selected=='Overview':
    sec0,sec1,sec2,sec4,sec5 = st.tabs(['App overview','Why do we care?','Goal','Variables Overview','Process flow'])
    
    sec1.title("Why do we care?")
    
    sec1.markdown("""
    Heart disease is a significant global health concern that affects millions of people every year. 
    Understanding its impact is crucial for raising awareness and promoting preventive measures. 
    Here are some key numbers and statistics that emphasize the importance of addressing heart disease:
    """)
    
    sec1.subheader("1. Global Prevalence:")
    sec1.write("According to the World Health Organization (WHO), heart disease is the leading cause of death worldwide, accounting for **17.9 million deaths annually**.")
    
    sec1.subheader("2. Economic Burden:")
    sec1.write("Heart disease poses a substantial economic burden. In the United States alone, the American Heart Association estimates the total cost of heart diseases and stroke in 2019 was **$351.2 billion**.")
    
    sec1.subheader("3. Risk Factors:")
    sec1.write("Several risk factors, including high blood pressure, high cholesterol, smoking, obesity, and diabetes, contribute to heart disease. Identifying these factors can significantly reduce the risk of developing heart-related issues.")
    
    sec1.subheader("4. Preventive Measures:")
    sec1.write("There are numerous causes of Heart Disease. Finding these variables can help prevent heart disease by manufacturing medicines to address these factors or make lifestyle changes accordingly.")
    
    sec1.subheader("Relevant Articles:")
    sec1.markdown("[Understanding Heart Disease: Risk Factors and Prevention](https://www.heart.org/en/health-topics/heart-attack/understand-your-risks-to-prevent-a-heart-attack)")
    sec1.markdown("[Global Heart Disease Statistics and Trends](https://www.who.int/health-topics/cardiovascular-diseases/#tab=tab_1)")
    
    sec2.markdown("""
    The goal of this web app is to serve as a powerful tool for researchers, healthcare professionals, and policymakers in the field of cardiology. The app's primary objective is to **identify variables causing heart disease** through EDA. However, its impact extends far beyond mere variable identification. Here's how the app achieves its goal and its potential for future research and collaboration:
    
    ## **1. Identification of Variables Causing Heart Disease:**
    The app utilizes EDA and statistical methods to help pinpoint variables that significantly influence heart disease occurrence. Researchers can use this app for future heart disease datasets, visualizing the relationship between variables, developing hypotheses, validating hypotheses and identifying key factors contributing to heart-related issues.
    
    ## **2. Future Potential and Long-term Impact:**
    - **Hypothesis Development and Testing:** Users can experiment with different variables, formulate hypotheses, and validate their assumptions in real-time, accelerating the pace of research and discovery.
    - **Educational Tool:** The app serves as an educational resource, enhancing analytical skills and domain knowledge for students and aspiring researchers.
    - **Clinical Decision Support:** Healthcare professionals can utilize the app for personalized patient care, leading to more effective treatment plans and prevention strategies.
    - **Public Health Policy Impact:** Policymakers can make informed decisions based on the app's insights, developing targeted interventions and optimizing resource allocation.
    
    The app acts as a dynamic platform, empowering users to explore, learn, and collaborate. It signifies a significant step towards a deeper understanding of heart disease and the development of impactful interventions, ultimately leading to a healthier global population.
    """)
    
    
    sec4.dataframe(df.sample(5))
    sec4.markdown("""
    The UCI Heart Disease dataset contains various clinical and demographic variables associated with heart disease. Below is a detailed overview of the dataset variables:
    
    ### 1. **Age:**
    - **Description:** Age of the patient.
    - **Type:** Continuous.
    
    ### 2. **Sex:**
    - **Description:** Gender of the patient.
    - **Type:** Categorical (F = female, M = male).
    
    ### 3. **ChestPainType :**
    - **Description:** Type of chest pain experienced by the patient.
    - **Type:** Categorical (TA, ATA, NAP, ASY).
    - **Categories:**
      - TA: Typical angina.
      - ATA: Atypical angina.
      - NAP: Non-anginal pain.
      - ASY: Asymptomatic.
    
    ### 4. **RestingBP:**
    - **Description:** Resting blood pressure of the patient (in mm Hg).
    - **Type:** Continuous.
    
    ### 5. **Cholesterol:**
    - **Description:** Serum cholesterol level of the patient (in mg/dl).
    - **Type:** Continuous.
    
    ### 6. **FastingBS:**
    - **Description:** Fasting blood sugar level of the patient.
    - **Type:** Categorical (0 = blood sugar < 120 mg/dl, 1 = blood sugar > 120 mg/dl).
    
    ### 7. **RestingECG:**
    - **Description:** Resting electrocardiographic measurement results.
    - **Type:** Categorical (Normal, ST, LVH).
    - **Categories:**
      - Normal: Normal.
      - ST: Abnormality related to ST-T wave.
      - LVH: Hypertrophy of the left ventricle.
    
    ### 8. **MaximumHR:**
    - **Description:** Maximum heart rate achieved during exercise.
    - **Type:** Continuous.
    
    ### 9. **ExerciseAngina:**
    - **Description:** Exercise-induced angina (chest pain) observed.
    - **Type:** Categorical (N = no, Y = yes).
    
    ### 10. **Oldpeak:**
    - **Description:** ST depression induced by exercise relative to rest.
    - **Type:** Continuous.
    
    ### 11. **ST_Slope:**
    - **Description:** Slope of the peak exercise ST segment.
    - **Type:** Categorical (Up, Flat, Down).
    - **Categories:**
      - Up: Upsloping.
      - Flat: Flat.
      - Down: Downsloping.
    
    ### 14. **HeartDisease:**
    - **Description:** Presence or absence of heart disease.
    - **Type:** Categorical (0 = no heart disease, 1 = heart disease).
    
    Researchers and data scientists use this dataset to analyze factors contributing to heart disease and build predictive models for early detection and prevention.
    """)
    
    sec0.markdown("""
    ## Interactive Heart Disease Analysis Web App
    
    ### Overview
    
    You are here! This section gives a broad overview of the app, the problem we are trying to solve, the goal, information about the 
    dataset, and the overall process flow.
    
    ### Basic checks
    
    Basic checks are done to test the integrity of the data before proceeding with the analysis
    
    ### EDA
    
    There are two types of EDA sections. One focuses on the distribution whereas the other focuses on the relationship between variables
    
    ### Further exploration
    
    Here, we cover advance topics like using the variables and insights we got through EDA to find variables that are good
    predictors.
    
    ### Summary
    
    We conclude our analysis with the key takeaways and further steps
    
    """)
    
    image=Image.open('process_flow.jpg')
    sec5.image(image,width=800)
    
if selected=="Basic Checks": ########################################################################################################
    sec1, sec2, sec3 = st.tabs(['Missing Values', 'Duplicate Values', 'Outlier treatment'])
    
    sec1.write("## Are there null values? Let's check")
    sec1.dataframe(pd.DataFrame(df.isna().sum(),columns=['Null Value Count']))
    sec1.write('#### :green[There are no missing values and no problems with missingness!]')
    
    sec2.write("## Let's check for duplicate values as they may skew our analysis and cause problems")
    sec2.dataframe(df[df.duplicated()])
    sec2.write(f'#### :green[The number of duplicated values are: {df.duplicated().sum()}]')
    
    sec3.write('## Checking for outliers')
    outlier_cols = sec3.multiselect('## Which features are you interested in?', quantitative_cols,['Age','Cholesterol'],key='outlier_sec3')
    if len(outlier_cols)>0:
        fig = make_subplots(rows=1, cols=len(outlier_cols))
        for i, var in enumerate(outlier_cols):
            if var in ['Cholesterol','RestingBP']:
                color='red'
            else:
                color='gray'
            fig.add_trace(
                go.Box(y=df_outliers[var],
                name=var, marker_color=color),
                row=1, col=i+1
            )
        fig.update_traces(boxpoints='all', jitter=.3)
        fig.update_layout(title_text=f"Boxplots of {', '.join(outlier_cols)} to find outliers", yaxis_title="Variable values")
        sec3.plotly_chart(fig)
    sec3.write('##### :red[We can see that there are some concerning outliers for Cholesterol and RestingBP. Cholesterol and RestingBP cannot be 0 for a living human being!]')
    sec3.write('The outlier rows cannot be dropped since there are a lot of outliers for the Cholesterol column. We can replace the outliers with the KNN imputer to make sure the integrity of the data is maintained and the correlation structure is not affected much.')
    sec3.write('**We do the outlier treatment using KNN by following the below steps:**')
    sec3.markdown("""
    1. Encode the categorical variables
    2. Scale the data
    3. Use KNN imputer by assuming the outliers as nulls
    4. Reverse scale the data
    5. Decode the categorical variables to their original labels for understanding purposes
    
    **Note:** Even though, in the backend, the outliers have been treated, it will not be displayed on this plot to ensure we can see what the outliers are. But for the analysis, we use the dataframe with outliers treated.
                  """)
    
if selected=='Further exploration': #################################################################################################
    sec2, sec3,sec4 = st.tabs(['Quantitative variables - intuition','Categorical variables - intuition','Quantification'])
    
    sec2.write('## Linear Separation:')
    sec2.write('Linear separation is crucial because it forms the foundation for many classification algorithms. If we are able to draw a line that distinguishes the two classes effectively, our predictive models can learn and make decisions. For example: ')
    lin_sep=Image.open('Linear_sep.png')
    sec2.image(lin_sep,width=800)
    sec2.write('Here, it is easy to draw a line separating the two distributions. We can draw it close to 6 as follows:')
    lin_sep2=Image.open('Linear_sep2.png')
    sec2.image(lin_sep2,width=800)
    sec2.write('It is clear that if the sepal length is greater than 5.8, it is most probably Virginica and if it is less than 5.8, it is most probably Setosa')
    sec2.write('**Thus, it is easy to distinguish between the two distributions based on this variable. Therefore, this variable is a good predictor. This concept is what we will leverage to find the best predictor of heart disease**')
    sec2.write('If the distribution looks like the one below, it is not linearly separable and is not a good predictor:')
    nonlin=Image.open('non.png')
    sec2.image(nonlin,width=800)
    sec2.write('**This is not a good predictor because you cannot draw a line to separate out the 3 classes effectively and so it is not a good predictor**')
    
    sec3.write('## Rates or Proportions')
    sec3.write('We can use proportions to determine if a specific class of a categorical variable is at risk of heart disease. For example: ')
    cat_im=Image.open('cat1.jpg')
    sec3.image(cat_im,width=800)
    sec3.write('Here, the heart disease percent for Males is more than 60% and for females, it is less than 30%. So, we can say that the probability of males getting heart disease is 60% and for females, it is 30%. **This helps us predict whether a person has heart disease or not. If the person is a male, we can say that the probability of getting heart disease is higher and this can help predict heart disease.** This is a good predictor of heart disease')
    sec3.write("Let's take a look at a different example:")
    cat_im2=Image.open('catim2.png')
    sec3.image(cat_im2,width=800)
    sec3.write('Let us say we want to identify whether the date variable can help us predict the Cultivar variable. Here, for all the classes of date, the cultivar proportion for c39 and c52 is almost the same. **Therefore, whatever class of date you take, the probability of cultivar being c39 or c52 are almost the same. We are not able to distinguish between c39 and c52 based on the classes of the date variable.** This is a bad predictor of cultivar.')
    
    sec4.markdown("""
    ## Quantification:
    
    Although we may have intuition and guesses as to which variable might be a good predictor based on EDA, we need to make sure through quantification methods. A sound mathematical basis is necessary to quantify our intuitions and hypotheses.  
    
    ### Quantitative Variables:
        
    For quantitative variables, we can use **Logistic Regression** to quantify our hypotheses.
    
    #### Logistic Regression:
        
    Since we are looking for good predictors of heart disease, why not let a predictive model tell us if our hypotheses are right or not? We can use logistic regression to select the "best decision boundary" based on its criteria and we can compare it with our own. That is the basis for the selection of the best decision boundary.
    
    For quantifying our hypotheses, we use accuracy, precision, and recall to see how good of a predictor this variable is. If we use this predictor alone to classify the data, the accuracy of the best decision boundary is displayed to decide if the variable is a good predictor or not. Arbitrary thresholds can be set, but for this app, a variable having more than 60% accuracy is considered as a good predictor.
    
    **Relevant Articles:**
        
    -[Logistic regression](https://en.wikipedia.org/wiki/Logistic_regression)
    
    -[Conceptual understanding](https://www.analyticsvidhya.com/blog/2021/08/conceptual-understanding-of-logistic-regression-for-data-science-beginners/)
    
    -[classification metrics - precision, recall, accuracy, and F1 score](https://www.analyticsvidhya.com/blog/2021/07/metrics-to-evaluate-your-classification-model-to-take-the-right-decisions/)
    
    ### Categorical Variables:
    
    For categorical variables, we can use **Chi-Squared test** to quantify our hypotheses.
    
    #### Chi-Squared test:
    
    For comparing categorical variables with Heart disease, we use the chi-squared test to determine if there exists a relationship between the two variables. If there is a relationship between the two variables, we can leverage that relationship to predict heart disease.
    
    Two tables are output. One is the actual frequency table and the other is the expected frequency table. The expected frequency table is what is expected when the two variables are totally independent of each other. If there is significant deviation from the expected frequency, there might be a relationship between the two variables.
    
    This existence of the relationship is quantified by the p-value which indicates the probability of there not being a relationship between the two variables. If the p-value is very low, it means that there exists some kind of relationship between the two variables. Arbitrary values of the p-value can be set, the most common being 0.05.
    
    **Relevant Articles:**
        
    -[Chi Squared test](https://en.wikipedia.org/wiki/Chi-squared_test)
    
    -[Testing relationship between two categorical variables](https://www.pluralsight.com/guides/testing-for-relationships-between-categorical-variables-using-the-chi-square-test)
                  """)
    
if selected=='Univariate Analysis-Quantitative': ###################################################################################
    sec1,sec2,sec3,sec4 = st.tabs(['Overall Picture', 'Playground', 'Develop Hypotheses','Test the Hypotheses'])
    
    sec1.markdown("""
    #### Note:
    Remember that you are trying to find plots with the best decision boundary. Refer to the **"Further exploration -> "Quantitative variables - intuition"** page in the app for more details. Try to shortlist variables that seem like they are a good predictor.
                  """)
    quant_chart = sec1.selectbox('Which plot would you like?',['Filled KDE','Histogram','Stepped histogram','KDE plot','KDE plot+Rug plot','Boxplot'])
    
    fig,ax=plt.subplots(math.ceil(len(quantitative_cols)/2),2)
    fig.set_figheight(math.ceil(len(quantitative_cols)/2*5))
    fig.set_figwidth(10)
    for i in range(len(quantitative_cols)):
        
        if quant_chart=='Histogram':
            sns.histplot(data=df, x=quantitative_cols[i],hue='HeartDisease',ax=ax[i//2,i%2])
        elif quant_chart=='Stepped histogram':
            sns.histplot(data=df, x=quantitative_cols[i],hue='HeartDisease',ax=ax[i//2,i%2],element='step')
        elif quant_chart=='Filled KDE':
            sns.kdeplot(data=df,x=quantitative_cols[i],hue='HeartDisease',ax=ax[i//2,i%2],fill=True)
        elif quant_chart=='KDE plot+Rug plot':
            sns.kdeplot(data=df,x=quantitative_cols[i],hue='HeartDisease',ax=ax[i//2,i%2])
            sns.rugplot(data=df,x=quantitative_cols[i],hue='HeartDisease',ax=ax[i//2,i%2])
        elif quant_chart=='KDE plot':
            sns.kdeplot(data=df,x=quantitative_cols[i],hue='HeartDisease',ax=ax[i//2,i%2])
        else:
            sns.boxplot(data=df,x='HeartDisease',y=quantitative_cols[i],ax=ax[i//2,i%2])
    fig.suptitle(f'{quant_chart} of quantitative variables across Heart Disease')
            
    if len(quantitative_cols)%2==1:
        fig.delaxes(ax[math.ceil(len(quantitative_cols)/2)-1][1])
    fig.tight_layout()
    sec1.pyplot()

    
    sec2.markdown("""
    #### Note:
    Try to find the decision boundaries for different variables to further confirm your intuition. Customize the decision boundary to see which decision boundaries can help serve your need for accuracy, precision, recall or F1 score (For some, precision might be more important than accuracy!). Also, try to play around with which section of the graph you think heart disease is more likely. (Before or after the line)
    
    **Try to find decision boundaries which have optimal values of all 4 metrics**
    
    For definition of metrics refer to **"Further exploration --> Quantification**". Metrics greater than 0.6 are :green[green]
                  """)
    option=sec2.selectbox('Select a variable to visualize it',quantitative_cols)
    line_val=sec2.slider('Choose the decision boundary',min_value=df[option].min(),max_value=df[option].max())
    relation=sec2.selectbox('Define the relationship: Is heart disease more likely before or after the dotted line?',['After','Before'])
    fig,ax=plt.subplots(1)
    fig.set_figheight(3)
    fig.set_figwidth(3)
    p=sns.kdeplot(data=df,x=option,hue='HeartDisease',ax=ax)
    plt.xticks(fontsize=7)
    plt.xlabel(option,fontsize=7)
    plt.yticks(fontsize=7)
    plt.ylabel('Density',fontsize=7)
    plt.setp(p.get_legend().get_texts(), fontsize='7')
    plt.setp(p.get_legend().get_title(), fontsize='7') 
    plt.title(f'KDE of {option} across Heart Disease',fontsize=7)
    ylimits=[max(p.lines[0].get_xydata()[:,1]),max(p.lines[1].get_xydata()[:,1])]
    maxylimit=max(ylimits)
    plt.vlines(x=line_val,ymin=0,ymax=maxylimit,color='black',linestyles='dotted',linewidth=2)
    sec2.pyplot(fig)
    if relation == 'Before':
        ypreds= df[option]<line_val
    else:
        ypreds= df[option]>line_val
    
    acc=round(accuracy_score(df.HeartDisease, ypreds),2)
    pre=round(precision_score(df.HeartDisease, ypreds),2)
    rec=round(recall_score(df.HeartDisease, ypreds),2)
    f1=round(f1_score(df.HeartDisease, ypreds),2)
    
    if acc>=0.6:
        sec2.write(f'### :green[The accuracy is {acc}]')
    else:
        sec2.write(f'### :red[The accuracy is {acc}]')
        
    if pre>=0.6:
        sec2.write(f'### :green[The precision is {pre}]')
    else:
        sec2.write(f'### :red[The precision is {pre}]')    
        
    if rec>=0.6:
        sec2.write(f'### :green[The recall is {rec}]')
    else:
        sec2.write(f'### :red[The recall is {rec}]')
    
    if f1>=0.6:
        sec2.write(f'### :green[The f1 score is {f1}]')
    else:
        sec2.write(f'### :red[The f1 score is {f1}]')
    
    
    sec3.write('### Develop Hypotheses!')
    sec3.write('##### Some hypotheses that we can develop are: ')
    sec3.write('1. Age variable can be used to predict Heart disease')
    sec3.write('2. MaxHR variable can also be used to predict Heart disease')
    sec3.write('3. OldPeak variable can be used to predict Heart disease as it has a clear decision boundary')
    
    sec3.write("#### Add your additional hypothesis too!")
    quant_hyp=sec3.text_input('Add your hypothesis')
    hyp.append(quant_hyp)
    if len(hyp)>0:
        sec3.write('**Additional Hypothesis:**')
    for i in hyp:
        sec3.write(f'-{i}')
    sec3.write('**Verify your hypothesis on the next tab!**')
    
    
    sec4.write('### Verify your Hypotheses!')
    
    sec4.markdown("""
    #### Note:
    We have used a logistic regression classifier to help us determine the optimum decision boundary and the different metrics based on this decision boundary. 
    
    See if your hypothesis was right by comparing it with the Logistic regression classifier! For the variables, all the metrics greater than 0.6 will be :green[green].
                  """)
    
    logit_option=sec4.selectbox('## Select a variable to get relevant reports!',quantitative_cols)
    model=LogisticRegression()
    model.fit(np.reshape(np.array(df[logit_option]),(-1,1)),df.HeartDisease)
    coef=model.coef_
    intercept=model.intercept_
    db= -1*intercept/coef
    score=model.score(np.reshape(np.array(df[logit_option]),(-1,1)),df.HeartDisease)
    yp_model=model.predict(np.reshape(np.array(df[logit_option]),(-1,1)))
    
    fig,ax=plt.subplots(1)
    fig.set_figheight(3)
    fig.set_figwidth(3)
    p=sns.kdeplot(data=df,x=logit_option,hue='HeartDisease',ax=ax)
    plt.xticks(fontsize=7)
    plt.xlabel(logit_option,fontsize=7)
    plt.yticks(fontsize=7)
    plt.ylabel('Density',fontsize=7)
    plt.setp(p.get_legend().get_texts(), fontsize='7')
    plt.setp(p.get_legend().get_title(), fontsize='7') 
    ylimits=[max(p.lines[0].get_xydata()[:,1]),max(p.lines[1].get_xydata()[:,1])]
    maxylimit=max(ylimits)
    
    if score>0.6: ###Set a slider for them to determine desired accuracy
        plt.vlines(x=db,ymin=0,ymax=maxylimit,color='green',linestyles='dotted',linewidth=2)
        plt.title(f'Best decision boundary for {logit_option}',fontsize=7)
        sec4.pyplot(fig)
        sec4.markdown("## :green[This is a good predictor of Heart disease!]")
    else:
        plt.vlines(x=db,ymin=0,ymax=maxylimit,color='red',linestyles='dotted',linewidth=2)
        plt.title(f'Best decision boundary for {logit_option}',fontsize=7)
        sec4.pyplot(fig)
        sec4.markdown("## :red[This is a bad predictor of Heart disease!]")
    
    acc_m=round(accuracy_score(df.HeartDisease, yp_model),2)
    pre_m=round(precision_score(df.HeartDisease, yp_model),2)
    rec_m=round(recall_score(df.HeartDisease, yp_model),2)
    f1_m=round(f1_score(df.HeartDisease, yp_model),2)
    
    if acc_m>=0.6:
        sec4.write(f'### :green[The accuracy is {acc_m}]')
    else:
        sec4.write(f'### :red[The accuracy is {acc_m}]')
        
    if pre_m>=0.6:
        sec4.write(f'### :green[The precision is {pre_m}]')
    else:
        sec4.write(f'### :red[The precision is {pre_m}]')    
        
    if rec_m>=0.6:
        sec4.write(f'### :green[The recall is {rec_m}]')
    else:
        sec4.write(f'### :red[The recall is {rec_m}]')
    
    if f1_m>=0.6:
        sec4.write(f'### :green[The f1 score is {f1_m}]')
    else:
        sec4.write(f'### :red[The f1 score is {f1_m}]')
        
if selected=='Univariate Analysis-Categorical':###################################################################################
    
    sec1,sec2, sec3, sec4 = st.tabs(['Overall Picture', 'Playground' ,'Develop Hypotheses','Test the Hypotheses'])
    
    sec1.markdown("""
                  #### Note:
                  Try to find variables whose classes show drastically differing proportions of heart disease. For more details on the intuition see the **"Further exploration" -> "Categorical variables - intuition"**
                  These variables can be good predictors as we can use the individual classes of the variable to predict the heart disease
                  """)
    type_of_cat= sec1.selectbox('Which plot would you like?', ['100% bar plot','Frequency plot','Side-by-side bar chart'])
    
    
    if type_of_cat=='Frequency plot':
        sec1.write('#### Warning!: Be careful while interpreting frequency plots for proportions as they might be misleading due to the varying height of the bars! Use frequency plots to view the frequencies alone')
        for i in range(len(categorical_cols)):
            fig=px.histogram(df, x=categorical_cols[i], color="HeartDisease",color_discrete_sequence=['green','red'])
            fig.update_layout(bargap=0.30,title_text=f'{type_of_cat} of {categorical_cols[i]}',title_x=0.4)
            sec1.plotly_chart(fig)
    elif type_of_cat=='Side-by-side bar chart':
        plt.figure(figsize=(5,5*len(categorical_cols)))
        for i in range(len(categorical_cols)):
            plt.subplot(len(categorical_cols),1,i+1)
            sns.countplot(data=df,x=categorical_cols[i],hue='HeartDisease')
            plt.title(f'Side-by-side bar chart of {categorical_cols[i]}')
            sec1.pyplot()
    else:
        for i in range(len(categorical_cols)):
            fig=px.histogram(df, x=categorical_cols[i], color="HeartDisease",barnorm='percent',color_discrete_sequence=['green','red'])
            fig.update_layout(bargap=0.30,title_text=f'{type_of_cat} of {categorical_cols[i]}',title_x=0.35)
            sec1.plotly_chart(fig)
    
    sec2.markdown("""
                  #### Note:
                  Let's try to identify variables whose classes have drastically different proportions of Heart disease. Choose a risk percent.(The default value is good enough for most cases)
                  and try to see if the different classes of a variable are in different colors. If they are, it means that the classes in the variable can help predict whether a person
                  has heart disease or not. If all the classes are of the same color, it means that this variable is not a good predictor. 
                  
                  **Warning:** For certain extreme risk values, all classes will be in the same colour even if the variable is a good predictor. Therefore, we will choose values around the middle(50%)
                  """)
    
    
    
    cat_option=sec2.selectbox('Choose a particular variable',categorical_cols)
    risk_perc=sec2.slider('Choose the risk percent',min_value=30,max_value=70,value=50)
    
    temp=pd.crosstab(index=df.HeartDisease,columns=df[cat_option],values=df.HeartDisease,aggfunc=len,normalize='columns')
    temp2=pd.DataFrame(temp.loc[1])
    ax=temp2[temp2<=risk_perc/100].plot.bar(figsize=(6,4),color='gray')
    temp2[temp2>risk_perc/100].plot.bar(figsize=(6,4),color='red',ax=ax)
    plt.xticks(rotation=0, ha='right')
    plt.title('Heart disease risk')
    plt.ylabel('Heart disease percent')
    plt.xlabel(f'{cat_option}')
    ax.get_legend().remove()
    sec2.pyplot()
    
    sec3.write('### Develop Hypotheses!')
    sec3.write('##### Some hypotheses that we can develop are:')
    sec3.write('1. Males are more likely to get heart disease')
    sec3.write('2. When chest pain is ASY, the risk of heart disease is more')
    sec3.write('3. When FastingBS is 1, risk of heart disease is more')
    sec3.write('4. When ExerciseAngina is Y, risk of heart disease is more')
    sec3.write('5. When ST_Slope is Flat or Down, risk of heart disease is more')
    sec3.markdown("""
                  #### Note: 
                  The restingECG might be a moderately good predictor since class ST has differing proportions than the rest of the classes but it is not that different and so we omit it from the hypotheses list""")
    
    sec3.write("#### Add your additional hypothesis too!")
    cat_hyp=sec3.text_input('Add your hypothesis')
    hyp2.append(cat_hyp)
    if len(hyp2)>0:
        sec3.write('**Additional Hypothesis:**')
    for i in hyp2:
        sec3.write(f'-{i}')
    sec3.write('**Verify your hypothesis on the next tab!**')
    
    
    sec4.markdown("""
                  ## Verify your Hypotheses!
                  #### Note:
                  Here we have two tables with the columns denoting the presence or absence of Heart disease. One denotes the expected frequency if the two variables are independent. If the actual frequency table
                  deviates significantly from it, we say that there is some relationship between the two variables that we can leverage. Otherwise, we cannot. A small p-value indicates that there exists a relationship
                  between the two variables. If the p-value is high, there is no relationship.
                  """)
    chi_option=sec4.selectbox('Select a variable to see if it has a relationship with Heart disease',categorical_cols)
    p_val=sec4.selectbox('Choose the p-value threshold',[0.001,0.005,0.01,0.05,0.1])
    data=pd.crosstab(index=df.HeartDisease,columns=df[chi_option],values=df.HeartDisease,aggfunc=len)
    stat, p, dof, expected = chi2_contingency(data)
    out=pd.DataFrame(expected.T,columns=data.index,index=data.columns)
    out=out.round(2)
    col1,col2=sec4.columns(2)
    col1.write('#### Actual frequency:')
    col1.dataframe(data.T)
    col2.write('#### Expected frequency: ')
    col2.dataframe(out)
    
    
    if p<p_val:
        sec4.write('**:green[The p value for {} is given by {:.2e}. This is a good predictor as there is significant deviation from the expected value!]**'.format(chi_option,p))
    else:
        sec4.write('**:red[The p value for {} is given by {:.2e}. This is not a good predictor as there is no significant deviation from the expected value based on our threshold!]**'.format(chi_option,p))
    
if selected=='EDA-relationships':
    sec1, sec2, sec3, sec4, sec5  = st.tabs(['The big picture','Finding relationships','Finding clusters','Visualizing frequency','Cat and Quant variables'])
    sec1.markdown("""
    ## The big picture
    It's a good idea to get the big picture to visualize the variables. Choose one or more of the columns(quantitative) to get their distribution and scatter plots in one-go
                  """)
    pairs=sec1.multiselect('Choose the variables you want to visualize:',quantitative_cols,['Age','Oldpeak'])
    if pairs:
        sns.pairplot(df[pairs+['HeartDisease']],palette=['green','purple'],hue='HeartDisease',corner=True)
        plt.suptitle(f'Pairplot of {", ".join(pairs)}',y=1.08)
        sec1.pyplot()
    
    sec1.markdown("""
    This chart helps us visualize both the scatter plots and the distribution plots in one go which is very convenient if we want to 
    see how the variable is distributed and how it relates to other variables in a single chart. **This chart can be used as a stepping stone
    to identify variables to deep-dive into, using the other tools present in this app.**
    
                  """)
    
    sec2.markdown("""
    ## Let's look for correlation between the variables
    It's important to know the relationships between different variables when analyzing a dataset.
    There might be some relationships that we can exploit to plot interesting graphs
    
    **Note:** The red line tries to give the best fit for this data using the lowess line. 
                """)
    
    xax=sec2.selectbox('Select the x-axis',quantitative_cols)
    yax=sec2.selectbox('Select the y-axis',['Oldpeak']+quantitative_cols[:-1])
    fig=px.scatter(df,xax,yax,trendline='lowess',trendline_color_override="red")
    # fig.add_vline(x=np.mean(df[xax]),line_dash='dash', line_color='red')
    # fig.add_hline(y=np.mean(df[yax]),line_dash='dash',line_color='red')
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(title_text=f'Scatter plot of {yax} vs {xax}',title_x=0.35)
    sec2.plotly_chart(fig)
    
    corr=round(df.corr(numeric_only=True)[xax][yax],2)
    if corr>=0.5 or corr<=-0.5:
        sec2.write(f'#### :red[The two variables are highly correlated! The correlation is {corr}]')
    else:
        sec2.write(f'#### :green[The two variables are not highly correlated. The correlation is {corr}]')
        
    
    sec2.markdown("""
    ## Takeaways:
    There is no significant correlation between the variables. So, there is no problem of multicollinearity between the variables if we want
    to perform machine learning in the future
                  """)
    
    sec3.markdown("""
    ## Let's look for clusters
    Let's see if there are any clusters that are formed across the different variables. These clusters can help us identify if there are any
    natural groupings between the variables that we can further investigate to get interesting insights.
                  """)
    xax2=sec3.selectbox('Select the x-axis',quantitative_cols,key='cluster_x')
    yax2=sec3.selectbox('Select the y-axis',['Oldpeak']+quantitative_cols[:-1],key='cluster_y')
    cat2=sec3.selectbox('Select the categorical variable you want to color:',categorical_cols+['HeartDisease'])
    fig2=px.scatter(df,xax2,yax2,color=cat2,color_discrete_sequence=['blue','red','green','yellow','brown','magenta'])
    fig2.update_xaxes(showgrid=False)
    fig2.update_yaxes(showgrid=False)
    fig2.update_layout(title_text=f'Scatter plot of {yax2} vs {xax2} across {cat2}',title_x=0.3)
    sec3.plotly_chart(fig2)
    sec3.markdown("""
    ## Takeaways:
    
    **There are no clear clusters formed, but we can notice some insights:**

    1. Males have higher spread of MaxHR across age
    2. The cholesterol for RestingECG=normal is more spread out than other values of RestingECG
    3. When ST_Slope is flat, it has more spread out values of Cholesterol
                  """)
    
    sec4.write("""
    ## Let's visualize the frequency of pairs of categorical variables
    
    Choose two categorical variables to see the frequency with which different values fall into the different cells. 
    This can help us understand the relationship across different categorical variables and also if some pairs are more likely to 
    occur than others. If a specific trend is observed, we can deep-dive into why that is happening.
               """)
    cat1=sec4.selectbox('Select one categorical variable',categorical_cols+['HeartDisease'])
    cat2=sec4.selectbox('Select the other categorical variable',['ST_Slope']+categorical_cols[:-1]+['HeartDisease'])
    agg=pd.crosstab(df[cat1],df[cat2],values=df[cat2],aggfunc=len)
    agg.fillna(0,inplace=True)
    
    fig=px.imshow(agg,text_auto=True,aspect="auto",color_continuous_scale='RdBu_r')
    fig.update_layout(title_text=f'Frequency heatmap of {cat1} vs {cat2}',title_x=0.25)
    sec4.plotly_chart(fig)
    
    sec4.markdown("""
    ### We can get the following insights:
    1. Most males have ST_Slope to be flat whereas most females have ST_Slope to be up
    2. The proportion of males having ExerciseAngina to be Y is greater than that of females
    3. When RestingECG is LVH or normal, patients are more likely to have No exercise related Angina. But when RestingECG is ST, the risk increases to 50%
                  """)
    
    sec5.markdown("""
    ## Let's see the relationship between a categorical and quantitative variable
    
    Interactions between categorical variable and quantitative variables are very useful to see if there are different distributions
    across certain values of the categorical variable. This can also help in identifying linear separation which we'll cover later in this 
    app
    
    Choose a quantitative and categorical variable to see their relationship
                  """)
    cat=sec5.selectbox('Choose a categorical variable',categorical_cols+['HeartDisease'])
    quant=sec5.selectbox('Choose a quantitative variable',quantitative_cols)
    plot_type=sec5.radio('Choose the type of plot',['Violin plot','Box plot'])
    
    if plot_type=='Violin plot':
        fig = px.violin(df, y=quant, x=cat, color=cat, box=True, points=False, hover_data=df.columns,color_discrete_sequence=px.colors.qualitative.Dark24)
        fig.layout.update(showlegend=False)
        fig.update_layout(title_text=f'{plot_type} of {quant} across {cat}',title_x=0.35)
        sec5.plotly_chart(fig)
    else:
        fig = px.box(df, y=quant, x=cat, color=cat, points=False, hover_data=df.columns,color_discrete_sequence=px.colors.qualitative.Dark24)
        fig.update_layout(title_text=f'{plot_type} of {quant} across {cat}',title_x=0.35)
        fig.layout.update(showlegend=False)
        sec5.plotly_chart(fig)
    
    sec5.markdown("""
    ### We get the below insights from these plots:
    1. Females generally tend to have max heart rates around the middle whereas the max heart rates of males are more spread out
    2. FastingBS is more likely to be 1 for higher values of age
    3. ST_Slope being up has higher values of MaxHR than other values of ST_Slope
    4. Heart disease is more likely when MaxHR is less
    5. ST_Slope being flat has more spread out values of Cholesterol
                  """)

if selected=='EDA-distributions':
    
    sec1,sec3,sec2,sec4 = st.tabs(["Distribution-Quant","Distribution-Cat","Risk",'Normality'])
    
    sec1.markdown("""
    ## Let's see how the variables are spread out!
    
    Let's see how the variables are spread out on its own or across different categories. This can help us understand how the data is
    distributed and if there are any interesting patterns we can observe
    
    **Note:** The black line indicates the mean and the red line indicates the median. This can give us a clue about the skewness of the section of the data that you choose.
                  """)
    
    option1=sec1.selectbox('Select a quantitative variable:',quantitative_cols)
    option2=sec1.selectbox('Select a categorical variable to color by:',['No color']+categorical_cols+['HeartDisease'])
    range_vals=sec1.slider('Select a range you want to focus on:',min_value=df[option1].min(),max_value=df[option1].max(),value=(df[option1].min(),df[option1].max()))
    if option2=='No color':
        fig=px.histogram(df[(df[option1]<=range_vals[1]) & (df[option1]>=range_vals[0])],option1)
    else:
        fig=px.histogram(df[(df[option1]<=range_vals[1]) & (df[option1]>=range_vals[0])],option1,color=option2,color_discrete_sequence=px.colors.qualitative.Set2)
    fig.add_vline(x=np.mean(df[(df[option1]<=range_vals[1]) & (df[option1]>=range_vals[0])][option1]),line_dash='dash',line_color='black')
    fig.add_vline(x=np.median(df[(df[option1]<=range_vals[1]) & (df[option1]>=range_vals[0])][option1]),line_dash='dash',line_color='red')
    fig.update_layout(title_text=f'Histogram of {option1}',title_x=0.4)
    sec1.plotly_chart(fig)
    sec1.write('### See the portion of the data you selected by toggling the below switch!')
    on=sec1.toggle('See the data!',True)
    if on:
        sec1.dataframe(df[(df[option1]<=range_vals[1]) & (df[option1]>=range_vals[0])])
    sec1.markdown("""
    ### Takeaways:
        
    1. We can see that Females are more likely to have oldpeak closer to 0 than other values of oldpeak
    2. The cholesterol for RestingECG=normal is more spread out than other values of RestingECG
    3. MaxHR is higher for ExerciseAngina being No than for ExerciseAngina being Yes
    4. Oldpeak is higher for ExerciseAngina being yes than for ExerciseAngina being no
    5. ST_Slope being down has higher values of oldpeak than for ST_Slope being up or flat.
    6. Prevalence of Heart disease is more as age increases
                  """)
    
    sec2.markdown("""
    ## Let's explore the proportion of heart disease for different classes
    
    Seelct a categorical variable and a specific class to see the proportion of Heart Disease for that class. This can help us understand
    how much risk is present for all the classes in terms of getting heart disease. 
    
    **Note:** The heart disease risk is highlighted in red and is shown as a separate portion
                  """)
    pie_cat=sec2.selectbox('Select a categorical variable:',categorical_cols)
    pie_class=sec2.selectbox('Select the class of the variable',df[pie_cat].unique())
    modified_df=df[['HeartDisease']+[pie_cat]][df[pie_cat]==pie_class]
    modified_df.HeartDisease.replace({1:'Yes',0:'No'},inplace=True)
    grouped=modified_df.groupby('HeartDisease')[pie_cat].count()
    # modified_df=order_df(df_input = modified_df, order_by='HeartDisease', order=['Yes','No'])
    # modified_df.reset_index(drop=True,inplace=True)
    plt.pie(grouped,labels=grouped.index,explode=[0,0.1],autopct='%.0f%%',colors=['gray','red'])
    plt.title(f'Pie chart of {pie_cat}-{pie_class}')
    sec2.pyplot()
    
    sec2.markdown("""
    ### We can get the following insights:
    1. Males are more likely to get heart disease
    2. When chest pain is ASY, the risk of heart disease is more
    3. When FastingBS is 1, risk of heart disease is more
    4. When ExerciseAngina is Y, risk of heart disease is more
    5. When ST_Slope is Flat or Down, risk of heart disease is more
                  """)
    
    sec3.markdown("""
    ## Let's visualize the distribution of categorical variables
    
    This can help us understand if there is any imbalance among the classes and if there are certain proportions that stand out and can aid
    our analysis
                  """)
    cat_dist=sec3.selectbox('Choose a categorical variable:',categorical_cols)
    dist_hue=sec3.selectbox('Choose the hue:',['No color']+categorical_cols)
    dist_type=sec3.radio('Choose the type of plot!',['Frequency','100% bar'])
    
    if dist_type=='100% bar':
        if dist_hue=='No color':
            fig=px.histogram(df, x=cat_dist,barnorm='percent')
            fig.update_layout(bargap=0.30)
        else:
            fig=px.histogram(df, x=cat_dist, color=dist_hue,barnorm='percent',color_discrete_sequence=px.colors.qualitative.Bold)
            fig.update_layout(bargap=0.30)
    else:
        if dist_hue=='No color':
            fig=px.histogram(df, x=cat_dist)
            fig.update_layout(bargap=0.30)
        else:
            fig=px.histogram(df, x=cat_dist, color=dist_hue,color_discrete_sequence=px.colors.qualitative.Bold)
            fig.update_layout(bargap=0.30)
    fig.update_layout(title_text=f'{dist_type} chart of {cat_dist}',title_x=0.4)
    sec3.plotly_chart(fig)
            
    sec3.markdown("""
    ### We get the following insights:
    1. Males are the dominant class in Sex variable
    2. There are more instances of ASY than any other type of chest pain
    3. FastingBS = 0  is the dominant class in FastingBS
    4. RestingECG being normal has the highest frequency relatively
    5. Most patients have no ExerciseAngina 
    6. Most patients have ST_Slope to be either up or flat
                  """)
    
    sec4.markdown("""
    ## Let's test the data for normality!
    
    **Note:** When the data is normal, there are some convenient methods we can use to know more information about the data. 
    Many of the tests like ANOVA depend on the assumption of normality. Choose a variable and see the below plot. 
    **If the data is normal, the data points would be very close to the red line. If not, they would deviate away from the line**
                  """)
    norm_var=sec4.selectbox('Choose a quantitative variable:',quantitative_cols)
    
    qq = probplot(df[norm_var], sparams=(1))
    x = np.array([qq[0][0][0], qq[0][0][-1]])
    
    fig = go.Figure()
    fig.add_scatter(x=qq[0][0], y=qq[0][1], mode='markers')
    fig.add_scatter(x=x, y=qq[1][1] + qq[1][0]*x, mode='lines',line_color='red')
    fig.layout.update(showlegend=False,title_text=f'Q-Q plot of {norm_var}',title_x=0.4,xaxis_title='quantiles',yaxis_title=norm_var)
    sec4.plotly_chart(fig)
    
    sec4.markdown("""
    ### We get the following information from the plot above:
    
    All variables except oldpeak are approximately normal since they are very close to the red line
                  """)
    
if selected=='Summary':
    sec1,sec2,sec3,sec4=st.tabs(['Applications','Key takeaways','Next steps','References'])
    
    sec1.markdown("""
    Heart disease is a widespread health concern affecting millions worldwide. Developing web apps to predict heart disease and identify its contributing factors can significantly impact public health. Here’s how this app can help resolve associated problems:
    
    ## 1. **Early Detection and Prevention:**
    - **Timely Intervention:** We can explore the relationship between variables to identify high-risk individuals, allowing for early medical intervention and preventive measures.
    
    ## 2. **Resource Optimization:**
    - **Efficient Healthcare:** Resources can be allocated efficiently, focusing on high-risk populations and ensuring cost-effective interventions.
    - **Data-Driven Policy:** Data collected from apps can inform healthcare policies, leading to targeted public health initiatives.
    
    ## 3. **Raising Awareness and Education:**
    - **Interactive Learning:** User-friendly apps can educate the public about risk factors, promoting understanding and awareness.
    
    ## 4. **Research and Development:**
    - **Data Collection:** Aggregated data can aid medical research, offering insights into patterns and intervention effectiveness.
    
    ## 5. **Global Impact:**
    - **Accessibility:** Web apps are globally accessible, bridging healthcare gaps in regions with limited resources.
    - **Global Collaboration:** Aggregated data enables global research collaborations, enhancing our understanding and treatment of heart disease.
    
    By harnessing technology, data-driven insights, and community engagement, predictive heart disease web apps can revolutionize healthcare, promote healthier lifestyles, and save lives globally.
    """)
    
    sec2.markdown("""
    ### From this analysis, the key takeaways are:
        
    **We are able to notice that the following quantitative variables have a strong relationship with Heart Disease:**
        
    1. Age
    2. MaxHR
    3. Oldpeak
    
    **We have verified that these variables are good predictors using classification metrics and logistic regression decision boundaries**
    
    **We are able to notice that the following categorical variables have a strong relationship with Heart Disease:**
    
    1. Sex
    2. ChestPainType
    3. FastingBS
    4. ExerciseAngina
    5. ST_Slope
    
    **We have verified that these variables are good predictors using the Chi-squared test**
    
    ### Based on this, we can give the following recommendations:
    
    1. As people age, more care should be taken in terms of lifestyle, food habits, and overall health as they are at a higher risk of heart disease.
    2. Maximum heart rate should be monitored regularly to and if it is within certain risky regions, precaution must be taken.
    3. When the oldpeak variable is NOT close to 0, regular monitoring is necessary as they are at a higher risk of heart disease.
    4. When males experience all these risky symptoms, they must be on high alert as they are more prone to heart diseases.
    5. When patients experienca an ASY ChestPainType, regular check-ups can be done to preempt heart disease
    6. Dietary changes can be done to reduce fasting blood sugar as it is has a stong relationship with Heart Disease
    7. Exercises shouldn't be too strenous to induce Exercise Angina as it is highly related to Heart Disease
    8. When ST_Slope is Flat or Down, caution must be taken as they are at a higher risk of heart disease
                  """)
    
    
    sec3.markdown("""
    Embarking on the next phase of our project, our Heart Disease Prediction Web App is poised for the application of machine learning methodologies. 
    Armed with the insights gleaned from Exploratory Data Analysis (EDA), we can focus on machine learning and deep learning methodologies. 
    Harnessing the power of data-driven analysis, we are set to explore traditional machine learning algorithms, such as Random Forest and Support Vector Machines, 
    enhancing our predictive accuracy. Concurrently, we will delve into the intricate world of deep learning, leveraging neural networks to decipher complex patterns within our dataset. 
    The variables identified through our web app's interactive platform will serve as the bedrock for these advanced models, 
    ensuring our predictions are anchored in the most relevant factors related to heart disease. 
    Continuous refinement will be our mantra, with rigorous testing and hyperparamter tuning.
    
    We can use powerful classification algorithms to further see if we can predict heart disease. We can also try various hyperparameter tuning
    methods to optimize the algorithms.
    
    **Some of the algorithms we can use are:**
    1. Random Forest Classifier
    2. Logistic regression
    3. Neural Networks
    4. Support Vector Machines
    5. Naive Bayes Algorithm
    6. Decision trees
    7. KNN classifier
    
    We will be deploying these algorithms to see if we can predict heart disease with high accuracy which will be invaluable in the field of medicine.
                  """)
    
    sec4.markdown("""
    The references to the useful links for this project are:
    
    1. [Data](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
    2. [Understanding Heart Disease: Risk Factors and Prevention](https://www.heart.org/en/health-topics/heart-attack/understand-your-risks-to-prevent-a-heart-attack)
    3. [Global Heart Disease Statistics and Trends](https://www.who.int/health-topics/cardiovascular-diseases/#tab=tab_1)
    4. [Logistic regression](https://en.wikipedia.org/wiki/Logistic_regression)
    5. [Conceptual understanding - Logistic Regression](https://www.analyticsvidhya.com/blog/2021/08/conceptual-understanding-of-logistic-regression-for-data-science-beginners/)
    6. [Evaluation metrics](https://www.analyticsvidhya.com/blog/2021/07/metrics-to-evaluate-your-classification-model-to-take-the-right-decisions/)
    7. [Chi Squared test](https://en.wikipedia.org/wiki/Chi-squared_test)
    8. [Chi Squared test implementation](https://www.pluralsight.com/guides/testing-for-relationships-between-categorical-variables-using-the-chi-square-test)
                  """)
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


