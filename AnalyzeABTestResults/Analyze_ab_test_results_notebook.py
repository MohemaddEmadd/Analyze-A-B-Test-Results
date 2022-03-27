#!/usr/bin/env python
# coding: utf-8

# # Analyze A/B Test Results 
# 
# This project will assure you have mastered the subjects covered in the statistics lessons. We have organized the current notebook into the following sections: 
# 
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# - [final results ](#final)
# 
# Specific programming tasks are marked with a **ToDo** tag. 
# 
# <a id='intro'></a>
# ## Introduction
# 
# A/B tests are very commonly performed by data analysts and data scientists. For this project, you will be working to understand the results of an A/B test run by an e-commerce website.  Your goal is to work through this notebook to help the company understand if they should:
# - Implement the new webpage, 
# - Keep the old webpage, or 
# - Perhaps run the experiment longer to make their decision.
# 
# Each **ToDo** task below has an associated quiz present in the classroom.  Though the classroom quizzes are **not necessary** to complete the project, they help ensure you are on the right track as you work through the project, and you can feel more confident in your final submission meeting the [rubric](https://review.udacity.com/#!/rubrics/1214/view) specification. 
# 
# >**Tip**: Though it's not a mandate, students can attempt the classroom quizzes to ensure statistical numeric values are calculated correctly in many cases.
# 
# <a id='probability'></a>
# ## Part I - Probability
# 
# To get started, let's import our libraries.

# In[66]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# ### ToDo 1.1
# Now, read in the `ab_data.csv` data. Store it in `df`. Below is the description of the data, there are a total of 5 columns:
# 
# <center>
# 
# |Data columns|Purpose|Valid values|
# | ------------- |:-------------| -----:|
# |user_id|Unique ID|Int64 values|
# |timestamp|Time stamp when the user visited the webpage|-|
# |group|In the current A/B experiment, the users are categorized into two broad groups. <br>The `control` group users are expected to be served with `old_page`; and `treatment` group users are matched with the `new_page`. <br>However, **some inaccurate rows** are present in the initial data, such as a `control` group user is matched with a `new_page`. |`['control', 'treatment']`|
# |landing_page|It denotes whether the user visited the old or new webpage.|`['old_page', 'new_page']`|
# |converted|It denotes whether the user decided to pay for the company's product. Here, `1` means yes, the user bought the product.|`[0, 1]`|
# </center>
# Use your dataframe to answer the questions in Quiz 1 of the classroom.
# 
# 
# >**Tip**: Please save your work regularly.
# 
# **a.** Read in the dataset from the `ab_data.csv` file and take a look at the top few rows here:

# In[67]:


df=pd.read_csv('Data./ab_data.csv')
df.head(5)


# **b.** Use the cell below to find the number of rows in the dataset.

# In[68]:


df.shape


# **c.** The number of unique users in the dataset.

# In[69]:


len(df['user_id'].unique())


# **d.** The proportion of users converted.

# In[70]:


df.query('converted == 1').count().converted / df['converted'].count()


# **e.** The number of times when the "group" is `treatment` but "landing_page" is not a `new_page`.

# In[71]:


df.query('group == "treatment" and landing_page == "old_page"').shape[0]


# In[72]:


df_A_not_B = df.query('group == "treatment" & landing_page != "new_page"')

df_B_not_A = df.query('group != "treatment" & landing_page == "new_page"')
len(df_A_not_B) + len(df_B_not_A)


# **f.** Do any of the rows have missing values?

# In[73]:


df.isnull().sum()


# ### ToDo 1.2  
# In a particular row, the **group** and **landing_page** columns should have either of the following acceptable values:
# 
# |user_id| timestamp|group|landing_page|converted|
# |---|---|---|---|---|
# |XXXX|XXXX|`control`| `old_page`|X |
# |XXXX|XXXX|`treatment`|`new_page`|X |
# 
# 
# It means, the `control` group users should match with `old_page`; and `treatment` group users should matched with the `new_page`. 
# 
# However, for the rows where `treatment` does not match with `new_page` or `control` does not match with `old_page`, we cannot be sure if such rows truly received the new or old wepage.  
# 
# 
# Use **Quiz 2** in the classroom to figure out how should we handle the rows where the group and landing_page columns don't match?
# 
# **a.** Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz.  Store your new dataframe in **df2**.

# In[74]:


# Remove the inaccurate rows, and store the result in a new dataframe df2
df_1=df.drop(df[  (df.group == 'treatment') & (df.landing_page != 'new_page' )  ].index )
df2=df_1.drop (df[ (df.group == 'control') & (df.landing_page != 'old_page')   ].index )


# In[75]:


# Double Check all of the incorrect rows were removed from df2 - 
# Output of the statement below should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# ### ToDo 1.3  
# Use **df2** and the cells below to answer questions for **Quiz 3** in the classroom.

# **a.** How many unique **user_id**s are in **df2**?

# In[76]:


df2.head()


# In[77]:


df2['user_id'].nunique()


# **b.** There is one **user_id** repeated in **df2**.  What is it?

# In[78]:


df2[df2.duplicated(['user_id'])]


# **c.** Display the rows for the duplicate **user_id**? 

# In[79]:


df2[df2['user_id'] == 773192]


# **d.** Remove **one** of the rows with a duplicate **user_id**, from the **df2** dataframe.

# In[80]:


# Remove one of the rows with a duplicate user_id..
# Hint: The dataframe.drop_duplicates() may not work in this case because the rows with duplicate user_id are not entirely identical. 
df2.drop(labels=2893 , inplace=True)
# Check again if the row with a duplicate user_id is deleted or not
df2[df2['user_id'] == 773192]


# In[81]:


df2.head()
df2.shape


# ### ToDo 1.4  
# Use **df2** in the cells below to answer the quiz questions related to **Quiz 4** in the classroom.
# 
# **a.** What is the probability of an individual converting regardless of the page they receive?<br><br>
# 
# >**Tip**: The probability  you'll compute represents the overall "converted" success rate in the population and you may call it $p_{population}$.
# 
# 

# In[82]:


probability=df2[df2['converted'] == 1].count() / df2['converted'].count()
probability.user_id


# **b.** Given that an individual was in the `control` group, what is the probability they converted?

# In[83]:


prob_control=df2.query('group == "control" and converted == 1').count() / df2.query('group == "control"').count()
prob_control.user_id


# **c.** Given that an individual was in the `treatment` group, what is the probability they converted?

# In[84]:


prob_treatment=df2.query('group == "treatment" and converted == 1').count() / df2.query('group == "treatment"').count()
prob_treatment.user_id


# >**Tip**: The probabilities you've computed in the points (b). and (c). above can also be treated as conversion rate. 
# Calculate the actual difference  (`obs_diff`) between the conversion rates for the two groups. You will need that later.  

# In[85]:


# Calculate the actual difference (obs_diff) between the conversion rates for the two groups.
obs_diff = prob_treatment - prob_control
obs_diff= obs_diff.user_id
obs_diff


# In[86]:


df2.head()


# **d.** What is the probability that an individual received the new page?

# In[87]:


prop_individual_new=df2.query('landing_page == "new_page"').count()/df2.landing_page.count()
prop_individual_new.user_id


# **e.** Consider your results from parts (a) through (d) above, and explain below whether the new `treatment` group users lead to more conversions.

# ### no there is no statistical evidence  to say that the new treatment page leads to more conversions.
# 
# ### from my analyse i've found half of the population received the old page and half of the population received the  new page , total capacity equal (290584) users   
# 
# ###  about 12.04 % received the old page (control group) and were converted.
# 
# ### about 11.88 % received  the new page (treatment group) and were converted.
# 
# ### hence we can't say the new page gave us more conversions  than the old one 
# 

# <a id='ab_test'></a>
# ## Part II - A/B Test
# 
# Since a timestamp is associated with each event, you could run a hypothesis test continuously as long as you observe the events. 
# 
# However, then the hard questions would be: 
# - Do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  
# - How long do you run to render a decision that neither page is better than another?  
# 
# These questions are the difficult parts associated with A/B tests in general.  
# 
# 
# ### ToDo 2.1
# For now, consider you need to make the decision just based on all the data provided.  
# 
# > Recall that you just calculated that the "converted" probability (or rate) for the old page is *slightly* higher than that of the new page (ToDo 1.4.c). 
# 
# If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should be your null and alternative hypotheses (**$H_0$** and **$H_1$**)?  
# 
# You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the "converted" probability (or rate) for the old and new pages respectively.

# $$H_0:  P_{new}<= P_{old} $$
# 
# $$H_1:  P_{new}> P_{old}$$
# 
# $$OR$$            
# 
# $$H_0:  P_{new}- p_{old}<= 0$$
# 
# $$H_1:  P_{new}- P_{old}> 0  $$

# ### ToDo 2.2 - Null Hypothesis $H_0$ Testing
# Under the null hypothesis $H_0$, assume that $p_{new}$ and $p_{old}$ are equal. Furthermore, assume that $p_{new}$ and $p_{old}$ both are equal to the **converted** success rate in the `df2` data regardless of the page. So, our assumption is: <br><br>
# <center>
# $p_{new}$ = $p_{old}$ = $p_{population}$
# </center>
# 
# In this section, you will: 
# 
# - Simulate (bootstrap) sample data set for both groups, and compute the  "converted" probability $p$ for those samples. 
# 
# 
# - Use a sample size for each group equal to the ones in the `df2` data.
# 
# 
# - Compute the difference in the "converted" probability for the two samples above. 
# 
# 
# - Perform the sampling distribution for the "difference in the converted probability" between the two simulated-samples over 10,000 iterations; and calculate an estimate. 
# 
# 
# 
# Use the cells below to provide the necessary parts of this simulation.  You can use **Quiz 5** in the classroom to make sure you are on the right track.

# **a.** What is the **conversion rate** for $p_{new}$ under the null hypothesis? 

# In[88]:


df2.head()


# In[93]:


p_new=df2.query('converted == 1 ').count() / df2.user_id.count()
p_new=p_new.user_id
p_new


# **b.** What is the **conversion rate** for $p_{old}$ under the null hypothesis? 

# In[94]:


p_old=df2.query('converted == 1 ').count() / df2.user_id.count()
p_old=p_old.user_id
p_old


# In[95]:


p_diff= p_new - p_old
p_diff


# ### Under null hypothesis p_old is equal to p_new 
# 

# **c.** What is $n_{new}$, the number of individuals in the treatment group? <br><br>
# *Hint*: The treatment group users are shown the new page.

# In[96]:


# the number of individuals in the treatment group
n_new=len(df2.query('landing_page == "new_page"'))
n_new


# **d.** What is $n_{old}$, the number of individuals in the control group?

# In[97]:


#the number of individuals in the control group
n_old=len(df2.query('landing_page != "new_page"'))
n_old


# **e. Simulate Sample for the `treatment` Group**<br> 
# Simulate $n_{new}$ transactions with a conversion rate of $p_{new}$ under the null hypothesis.  <br><br>
# *Hint*: Use `numpy.random.choice()` method to randomly generate $n_{new}$ number of values. <br>
# Store these $n_{new}$ 1's and 0's in the `new_page_converted` numpy array.
# 

# In[98]:


# Simulate a Sample for the treatment Group under the null
# treatment group 
new_page_converted =np.random.choice([0, 1], size=n_new, p = [p_new, 1-p_new])
new_page_converted= new_page_converted.mean()
new_page_converted


# **f. Simulate Sample for the `control` Group** <br>
# Simulate $n_{old}$ transactions with a conversion rate of $p_{old}$ under the null hypothesis. <br> Store these $n_{old}$ 1's and 0's in the `old_page_converted` numpy array.

# In[99]:


# Simulate a Sample for the control Group under the null
#control group
old_page_converted= np.random.choice([0,1] , size=n_old, p=[p_old , 1-p_old])
old_page_converted= old_page_converted.mean()
# the difference  between theconverted  p_new and the  converted p_old
new_page_converted -old_page_converted


# **g.** Find the difference in the "converted" probability $(p{'}_{new}$ - $p{'}_{old})$ for your simulated samples from the parts (e) and (f) above. 

# In[100]:


# the difference between theconverted  p_new and the  converted p_old
new_page_converted -  old_page_converted


# 
# **h. Sampling distribution** <br>
# Re-create `new_page_converted` and `old_page_converted` and find the $(p{'}_{new}$ - $p{'}_{old})$ value 10,000 times using the same simulation process you used in parts (a) through (g) above. 
# 
# <br>
# Store all  $(p{'}_{new}$ - $p{'}_{old})$  values in a NumPy array called `p_diffs`.

# In[101]:


# create Sampling distribution for difference in p_new and p_old page 
# our sample will be 10000
# treatment for new page
# control for old page 
p_diffs = []
for _ in range(10000):
    treatment= np.random.choice([0, 1], size=n_new, replace = True , p = [p_new, 1-p_new]) 
    control =  np.random.choice([0,1] , size=n_old, replace = True , p=[p_old , 1-p_old])
    obs_diffs = treatment.mean() - control.mean()
    p_diffs.append(obs_diffs)


# **i. Histogram**<br> 
# Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.<br><br>
# 
# Also, use `plt.axvline()` method to mark the actual difference observed  in the `df2` data (recall `obs_diff`), in the chart.  
# 
# >**Tip**: Display title, x-label, and y-label in the chart.

# In[102]:


# convert to numpy array
p_diffs=np.array(p_diffs)


# In[103]:


# plot our result usign histogram
plt.hist(p_diffs);


# In[104]:


# create distribution under the null hypothesis
null_vals = np.random.normal(0,p_diffs.std(),p_diffs.size)


# In[105]:


# plot null distribution
plt.hist(null_vals)

# plot line for observed statistic
plt.axvline(obs_diff,c='r');


# **j.** What proportion of the **p_diffs** are greater than the actual difference observed in the `df2` data?

# In[106]:


# compute p value
(null_vals > obs_diffs).mean()


# **k.** Please explain in words what you have just computed in part **j** above.  
#  - What is this value called in scientific studies?  
#  - What does this value signify in terms of whether or not there is a difference between the new and old pages? *Hint*: Compare the value above with the "Type I error rate (0.05)". 

# - we just computed the p-value it means  the probability of getting a result that is either the same of more extreme than the     actual observations. 
# - 43.38% is the proportion of the null_vals that are greater than the actual difference observed in ab_data.csv. 
# - as we know the type error if 0.05 than
# - ### the result is not statistically significant and hence  don't reject the null hypothsis.
# - ##  we do not have sufficient evidence that the new_page has a higher conversion rate than the old_page.

# 
# 
# **l. Using Built-in Methods for Hypothesis Testing**<br>
# We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. 
# 
# Fill in the statements below to calculate the:
# - `convert_old`: number of conversions with the old_page
# - `convert_new`: number of conversions with the new_page
# - `n_old`: number of individuals who were shown the old_page
# - `n_new`: number of individuals who were shown the new_page
# 

# In[107]:


df2.head()


# In[132]:


import statsmodels.api as sm

# number of conversions with the old_page
convert_old = len(df2.query('landing_page == "old_page" and converted == 1'))

# number of conversions with the new_page
convert_new = len(df2.query('landing_page == "new_page" and converted == 1'))

# number of individuals who were shown the old_page
n_old = len(df2.query(' landing_page == "old_page" '))

# number of individuals who received new_page
n_new = len(df2.query('landing_page == "new_page"'))


# **m.** Now use `sm.stats.proportions_ztest()` to compute your test statistic and p-value.  [Here](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportions_ztest.html) is a helpful link on using the built in.
# 
# The syntax is: 
# ```bash
# proportions_ztest(count_array, nobs_array, alternative='larger')
# ```
# where, 
# - `count_array` = represents the number of "converted" for each group
# - `nobs_array` = represents the total number of observations (rows) in each group
# - `alternative` = choose one of the values from `[‘two-sided’, ‘smaller’, ‘larger’]` depending upon two-tailed, left-tailed, or right-tailed respectively. 
# >**Hint**: <br>
# It's a two-tailed if you defined $H_1$ as $(p_{new} = p_{old})$. <br>
# It's a left-tailed if you defined $H_1$ as $(p_{new} < p_{old})$. <br>
# It's a right-tailed if you defined $H_1$ as $(p_{new} > p_{old})$. 
# 
# The built-in function above will return the z_score, p_value. 
# 
# ---
# ### About the two-sample z-test
# Recall that you have plotted a distribution `p_diffs` representing the
# difference in the "converted" probability  $(p{'}_{new}-p{'}_{old})$  for your two simulated samples 10,000 times. 
# 
# Another way for comparing the mean of two independent and normal distribution is a **two-sample z-test**. You can perform the Z-test to calculate the Z_score, as shown in the equation below:
# 
# $$
# Z_{score} = \frac{ (p{'}_{new}-p{'}_{old}) - (p_{new}  -  p_{old})}{ \sqrt{ \frac{\sigma^{2}_{new} }{n_{new}} + \frac{\sigma^{2}_{old} }{n_{old}}  } }
# $$
# 
# where,
# - $p{'}$ is the "converted" success rate in the sample
# - $p_{new}$ and $p_{old}$ are the "converted" success rate for the two groups in the population. 
# - $\sigma_{new}$ and $\sigma_{new}$ are the standard deviation for the two groups in the population. 
# - $n_{new}$ and $n_{old}$ represent the size of the two groups or samples (it's same in our case)
# 
# 
# >Z-test is performed when the sample size is large, and the population variance is known. The z-score represents the distance between the two "converted" success rates in terms of the standard error. 
# 
# Next step is to make a decision to reject or fail to reject the null hypothesis based on comparing these two values: 
# - $Z_{score}$
# - $Z_{\alpha}$ or $Z_{0.05}$, also known as critical value at 95% confidence interval.  $Z_{0.05}$ is 1.645 for one-tailed tests,  and 1.960 for two-tailed test. You can determine the $Z_{\alpha}$ from the z-table manually. 
# 
# Decide if your hypothesis is either a two-tailed, left-tailed, or right-tailed test. Accordingly, reject OR fail to reject the  null based on the comparison between $Z_{score}$ and $Z_{\alpha}$. We determine whether or not the $Z_{score}$ lies in the "rejection region" in the distribution. In other words, a "rejection region" is an interval where the null hypothesis is rejected iff the $Z_{score}$ lies in that region.
# 
# >Hint:<br>
# For a right-tailed test, reject null if $Z_{score}$ > $Z_{\alpha}$. <br>
# For a left-tailed test, reject null if $Z_{score}$ < $Z_{\alpha}$. 
# 
# 
# 
# 
# Reference: 
# - Example 9.1.2 on this [page](https://stats.libretexts.org/Bookshelves/Introductory_Statistics/Book%3A_Introductory_Statistics_(Shafer_and_Zhang)/09%3A_Two-Sample_Problems/9.01%3A_Comparison_of_Two_Population_Means-_Large_Independent_Samples), courtesy www.stats.libretexts.org
# 
# ---
# 
# >**Tip**: You don't have to dive deeper into z-test for this exercise. **Try having an overview of what does z-score signify in general.** 

# In[114]:


#insatall statsmodels
pip install statsmodels


# In[115]:


import statsmodels.api as sm

# number of conversions with the old_page
convert_old = len(df2.query('landing_page == "old_page" and converted == 1'))

# number of conversions with the new_page
convert_new = len(df2.query('landing_page == "new_page" and converted == 1'))

# number of individuals who were shown the old_page
n_old = len(df2.query(' landing_page == "old_page" '))

# number of individuals who received new_page
n_new = len(df2.query('landing_page == "new_page"'))


# In[116]:


import statsmodels.api as sm

#Computing z_score and p_value using  sm.stats.proportions_ztest()
z_score, p_value = sm.stats.proportions_ztest([convert_old , convert_new] , [n_old, n_new], value=None, alternative='smaller' )

print(z_score, p_value)


# **n.** What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?<br><br>
# 
# >**Tip**: Notice whether the p-value is similar to the one computed earlier. Accordingly, can you reject/fail to reject the null hypothesis? It is important to correctly interpret the test statistic and p-value.

# - we just computed the p-value using another approach called z-score and got the same results as we got earlier because the p-value is greater then alpha or type error 1 that means the result is not statistically significant and hance don't reject the null hypothesis.
# - ##  we do not have sufficient evidence that the new_page has a higher conversion rate than the old_page.

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# ### ToDo 3.1 
# In this final part, you will see that the result you achieved in the A/B test in Part II above can also be achieved by performing regression.<br><br> 
# 
# **a.** Since each row in the `df2` data is either a conversion or no conversion, what type of regression should you be performing in this case?

# - we will choose the logistic regression because our raw in the df2  is  either a (converted or no converted) variable

# **b.** The goal is to use **statsmodels** library to fit the regression model you specified in part **a.** above to see if there is a significant difference in conversion based on the page-type a customer receives. However, you first need to create the following two columns in the `df2` dataframe:
#  1. `intercept` - It should be `1` in the entire column. 
#  2. `ab_page` - It's a dummy variable column, having a value `1` when an individual receives the **treatment**, otherwise `0`.  

# In[117]:


# create our intercept
df2['intercept']= 1
#create our ab_page it is a dummy variable having 1 value when an individual receives the treatment, onterwise =0
df2['ab_page'] = pd.get_dummies(df2['group'])['treatment']
#show our data_frame
df2.head()


# **c.** Use **statsmodels** to instantiate your regression model on the two columns you created in part (b). above, then fit the model to predict whether or not an individual converts. 
# 

# In[118]:


import statsmodels.api as sms 
#import scipy from stats  for getting chisqprob
from scipy import stats
stats.chisqprob = lambda chisq, df2: stats.chi2.sf(chisq, df2)

lm=sms.Logit(df2['converted'],df2[['intercept','ab_page']])
results=lm.fit()


# **d.** Provide the summary of your model below, and use it as necessary to answer the following questions.

# In[119]:


results.summary()


# **e.** What is the p-value associated with **ab_page**? Why does it differ from the value you found in **Part II**?<br><br>  
# 
# **Hints**: 
# - What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in **Part II**? 
# - You may comment on if these hypothesis (Part II vs. Part III) are one-sided or two-sided. 
# - You may also compare the current p-value with the Type I error rate (0.05).
# 

# - the Alternative hypothesis from part two :- is the conversion rate  for the new page is bigger than the old page 
#   and in any one our result is not statistically significant and hance don't reject the null hypothesis.
# $$H_0:  P_{new}<= P_{old} $$
# 
# $$H_1:  P_{new}> P_{old}$$
# 
# $$OR$$            
# 
# $$H_0:  P_{new}- p_{old}<= 0$$
# 
# $$H_1:  P_{new}- P_{old}> 0  $$
# 
# - two-tailed testing.
# 
# - from my analysis the p-value is associated with ab_page is 0.190 and in Comparison with alpha or type one error will be higher than 0.05 thus, the coefficient is not significant.

# **f.** Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# - I don't know our additional factors will influence the result in which direction.
#   and it  the disadvantage that our model gonna be harder  and more complexity

# **g. Adding countries**<br> 
# Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives in. 
# 
# 1. You will need to read in the **countries.csv** dataset and merge together your `df2` datasets on the appropriate rows. You call the resulting dataframe `df_merged`. [Here](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.join.html) are the docs for joining tables. 
# 
# 2. Does it appear that country had an impact on conversion?  To answer this question, consider the three unique values, `['UK', 'US', 'CA']`, in the `country` column. Create dummy variables for these country columns. 
# >**Hint:** Use `pandas.get_dummies()` to create dummy variables. **You will utilize two columns for the three dummy variables.** 
# 
#  Provide the statistical output as well as a written response to answer this question.

# In[120]:


# Read the countries.csv
countries= pd.read_csv('Data./countries.csv')
countries.head()


# In[121]:


# inner Join with the df2 dataframe
df_new_country=countries.set_index('user_id').join(df2.set_index('user_id'), how='inner' )


# In[122]:


# Create the necessary dummy variables
df_new_country.head()


# In[123]:


df_new_country.country.unique()


# In[124]:


# mean of conversion rate
df_new_country.ab_page.mean()


# In[125]:


# mean conversion rate by country
df_new_country.groupby(['country'], as_index=False).mean()


# In[126]:


#create our ab_page it is a dummy variable having  three unique values
df_new_country[['CA','UK', 'US']]= pd.get_dummies(df_new_country['country'])
df_new_country.head()


# In[127]:


# fit our liner model and obtain the summery
df_new_country['intercept'] = 1
lm = sms.Logit(df_new_country['converted'], df_new_country[['intercept','ab_page','UK','US']])
results = lm.fit()
results.summary()


# - for me the analysis, countries doesn't appear to have influence on the conversion rate ,
#   and the p- value for the two dummies country  variables are above than type error one or alpha 0.05.
#   and If we look closely  to UK variable get closes to 0.05.

# In[ ]:





# **h. Fit your model and obtain the results**<br> 
# Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if are there significant effects on conversion.  **Create the necessary additional columns, and fit the new model.** 
# 
# 
# Provide the summary results (statistical output), and your conclusions (written response) based on the results. 
# 
# >**Tip**: Conclusions should include both statistical reasoning, and practical reasoning for the situation. 
# 
# >**Hints**: 
# - Look at all of p-values in the summary, and compare against the Type I error rate (0.05). 
# - Can you reject/fail to reject the null hypotheses (regression model)?
# - Comment on the effect of page and country to predict the conversion.
# 

# - we will looked at the interactions between page and countries to see if there significant effects on conversion

# In[128]:


# mean conversion rate by country and landing_page - 
# checking for possible interactions whether the influence of landing_page is different for the countries
df_new_country.groupby(['country','ab_page'], as_index=False).mean()


# - after we looked at the intersection 
#   check if the the influence of the landing_page maybe work in the US 
# 

# In[129]:


df_new_country.head()
df_new_country['interaction_us_ab_page'] = df_new_country['US'] *df_new_country['ab_page']
df_new_country['interaction_ca_ab_page'] = df_new_country['CA'] *df_new_country['ab_page']

df_new_country.head()
                                


# In[130]:


df_new_country['intercept'] = 1

lm = sms.Logit(df_new_country['converted'],df_new_country[['intercept','ab_page','US','interaction_us_ab_page','CA','interaction_ca_ab_page']])
results = lm.fit()
results.summary()


# <a id='final'></a>
# 
# # final results
# 
# ### Based on the statistical tests I used, the Z-test, logistic regression model, and actual difference observed, the results have shown that the new and old page have an approximately equal chance of converting users. We fail to reject the null hypothesis. I recommend to the e-commerce company to keep the old page. This will save time and money on creating a new page.

# In[133]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Analyze_ab_test_results_notebook.ipynb'])


# In[ ]:




