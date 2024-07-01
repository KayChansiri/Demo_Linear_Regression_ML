# Linear Regression from a Machine Learning Perspective

Let's talk about one of the simplest machine learning (ML) algorithms today: regression. Just like other ML algorithms, regression can predict both continuous outcomes (i.e., linear regression) and categorical outcomes (i.e., binomial or multinomial regression). Today, we'll focus on linear regression.

## Differences Between Traditional Statistics and Machine Learning

Before we get started, let's briefly discuss the differences between linear regression from a traditional statistics standpoint versus a machine learning standpoint.

### Traditional Statistics
* **Dataset Usage**: Traditional statistics uses only one dataset without dividing it into testing, training, and validation sets for modeling.
* **Data Preparation**: Involves standard data preparation processes, such as encoding variables, checking for and imputing missing data, and testing linear assumptions (e.g., linearity, normality, independence of errors, heteroscedasticity). Linear regression from a traditional statistics perspective also involves checking for multicollinearity using methods such as VIF (Variance Inflation Factor) or correlation tests. For VIF, if the value of a specific variable is more, the variable is likely to be highly correlated with other variables in the dataset. You may found that some sources use VIF more than 10 as the cutoff point if they want to be a bit moe liberal. Regarding pearson correlations, if any pairs of variables are correlated more than 0.9, the varibales are highly correlated. If multicollinearity issues exist, feature selection processes, such as dimensionality reduction (e.g., exploratory factor analysis, confirmatory factor analysis), centering (especially when interaction terms are included in the model), or excluding predictors with conceptually less importance may be performed.

As linear regression relies on the ordinary least squares approach and gradient descent methods to identify the best regression coefficients  that reduce error variances of the model, standardizing predictors to the same scale is also vital to ensure the optimal solution. If your features are on different scales, interpreting the model fit and its predictability would be challenging. For example, imagine you are predicting customer satisfaction towards an airline service on a scale of 1 to 5 (this is your y or target outcome) by customer age (rangesfrom 5 to 100) (X1) and customers’ frequency in flying per week (ranges from 0 to 7) (X2). An increase in one unit of the coefficient of your X1 would be different from an increase in one unit of the coefficient of your X2, and X1 can increase up to 100, whereas X2 can increase up to  7. To know which feature yields more impact on the outcome, you have to standardize them to be on the same scale. While standardization is  important for linear regression, keep in mind that it might not be as crucial for some other ML algorithms, such as random forests or decision   trees. For those algorithms, different scales of features do not matter as much since the algorithm relies on decision boundaries in fitting the model.

* **Model Emphasis**: Linear regression from a traditional approch emphasizes inference and determining if a model explains past data well, rather than predicting future data.
  
### Machine Learning
* **Dataset Usage**: Unlike traidtional statistics, linear regression from an ML perspective performs data separation into testing, training, and validation sets to ensure the model works well with unseen data. For projects with less computational power or budget, separating the data into testing and training sets without a validation set may be sufficient.
* **Data Preparation**: Similar to linear regression from a traditional standpoint, the ML approach should ensure that linear assumptions are met, missing data are imputed or excluded, all continious variables are standardized, and no multicollinearity issues exist. Taking an additional step from traditional statistical approach, linear regression from an ML perspective uses regularization techniques (e.g., ridge, lasso, elastic net) and fine-tunes hyperparameters (e.g., lambda and alpha) to deal with multicollinearity or variance issues from meaningless variables. The process also involves K-fold cross-validation to adjust the model's hyperparametersd during the traning process before model testing deployment. You will understand these concepts better later in the post.
* **Model Emphasis**: Unlike traditional statistics that focus on explaining existing data, ML focuses on building robust models that can generalize well to new data.

<img width="642" alt="Screen Shot 2024-06-25 at 3 11 42 PM" src="https://github.com/KayChansiri/LinearRegressionML/assets/157029107/95f956ba-fe96-4475-a546-8f09163d187e">

## Loss Function
The basis of linear regression is the loss function. To understand how linear regression works, you first have to understand how the loss function works. The loss function is a mathematical function that tries to minimize the distance between observed and predicted values so that our model is as accurate as possible. The function can be represented by the following equation:

<img width="357" alt="Screen Shot 2024-06-25 at 8 21 59 PM" src="https://github.com/KayChansiri/LinearRegressionML/assets/157029107/3aeb517a-efab-452b-a340-d6e91bbd8982">

Here SSR refers to the sum of squared residuals. X and Y values are the observed data (i.e., the X and Y columns of the data you collected, and {β0,β1} are the unknown parameters. The goal of the loss function, as an optimization process, is to find the set of {β0,β1} that provides the minimum value of this function (AKA the mimium distance between observed and predicted values). If you are still confused about the idea, let's start by looking at a typical linear regression equation below:

<img width="158" alt="Screen Shot 2024-06-25 at 7 21 36 PM" src="https://github.com/KayChansiri/LinearRegressionML/assets/157029107/c189b145-1833-4e8c-8fbd-3b2fcf444f8b">

Here, *i* refers to sample *i* . Say you have a dataset with 100 samples or observations, and you just randomly come up with an idea that your first pair of β<sub>0</sub> and  β<sub>1</sub> should be 2.5 and 0.8, respectively. If your sample *i*  has  X = 20, the model would predict their *y* as 2.5 + 0.8(20) = 18.5. However, if their true y score is 20, it means that our error term is 20 - 18.5 = 1.5. Remember, we have 100 observations in the dataset, so we will need to aggregate them in some way to get the total amount of errors. This can be calculated using different strategies, such as SSR, which I mentioned previously, or the sum of absolute residuals (SAR). SSR is more preferable as SAR has absolute values that are mathematically more challenging to deal with.

Now, imagine we get the SSR for all samples, which represents the total errors in the dataset based on our β<sub>0</sub> and  β<sub>1</sub>, which are 2.5 and 0.8. **One challenge is that we do not know if β<sub>0</sub> and  β<sub>1</sub> are the best coefficient estimates**. Thus, we have to try out many pairs of β<sub>0</sub> and β<sub>1</sub> and see what is the best that we could get, meaning what pair of β<sub>0</sub> and  β<sub>1</sub> could best reduce the sum of squared errors.

If you try different values of β<sub>0</sub> and  β<sub>1</sub> ranging from -10 to 10 and plot the beta values against the SSR, you will get something like the figure below. 


<img width="592" alt="Screen Shot 2024-06-25 at 7 41 10 PM" src="https://github.com/KayChansiri/LinearRegressionML/assets/157029107/ffc21698-3249-4ea2-9a2d-1f13e9d25234">

The lowest point is called *the global minimum*, which is the point where the values of β<sub>0</sub> and  β<sub>1</sub> yield the least error in the model. In other words, this is the point where the regression coefficient values provide the optimal solution. You may hear people use some alternative phrases to refer to the global minimum, such as the point where the model converges or where we find a convex surface.

Behind the scene when you run a linear regression model, the machine (e.g., your laptop) uses linear algebra to the find the global minimum, which can be simply written in a matrix operation form as below. Now keep in mind that in this matrix operation example, we have only one predictor  β<sub>1</sub>  which I set in to range from 1 to 10, and the intercept (β<sub>0</sub> ), which I set it to be 1 for simplicity.  In the real world, you will 100% have more than one predictor and will get a more complex matrix operation equation. 


<img width="299" alt="Screen Shot 2024-06-25 at 8 09 27 PM" src="https://github.com/KayChansiri/LinearRegressionML/assets/157029107/424e6eba-7d7b-4e49-a279-9efeafcecf3e">


Now that we have the matrix form, we can find the values of the betas by performing some matrix transposition and inversion, as shown in the equation below:

<img width="206" alt="Screen Shot 2024-06-25 at 8 15 31 PM" src="https://github.com/KayChansiri/LinearRegressionML/assets/157029107/da19f29d-eba3-4e96-b61b-c87ed6c72ef6">

Note that in linear regression, the equation above provides a closed-form solution as we have a global minimum due to the convex nature of the loss function. For some other types of machine learning models that are non-linear (e.g., neural networks), you may end up having many *local minima*, as shown in the picture below. This can make the interpretation of the model more complicated. For those types of algorithms, we use an iterative process like gradient descent to find the best spot that is as close to the lowest point as possible, thereby obtaining the optimal beta values. These types of equations do not have a closed-form solution because their loss functions are non-convex and can have multiple local minima.


<img width="524" alt="Screen Shot 2024-06-25 at 8 32 39 PM" src="https://github.com/KayChansiri/LinearRegressionML/assets/157029107/f3e7f8e1-208c-4d99-941f-9e5ff9eaab5d">

In reality, you don't have to calculate the loss function manually, as there are packages for linear regression in Python such as scikit-learn, statsmodels, and numpy or caret if you are an R user.

## Multicollinearity and Feature Selection

Now that you understand the idea of the loss function, let’s discuss ways to deal with multicollinearity from an ML perspective. Spoiler alert, the idea is pretty much similar to traditional statistics with some additional functions that we can play with!

### Tolerance

* Tolerance refers to the amount of variance that is unique to one predictor and cannot be explained by the rest of the predictors in the model.
* When a variable’s tolerance is 0 or close to zero, you will get a singularity issue warning when you try to fit the model. The warning indicates that two or more variables essentially explain the same thing in the outcome.
* When we encounter the singularity warning, ordinary least squares cannot find the global minimum. Variables that explain the same thing could lead to multiple global minima, resulting in your model not converging.

## VIF 

* VIF is the inverse of tolerance (VIF = 1/tolerance) and explains how much the variance of a regression coefficient is inflated due to collinearity with other features.
* For instance, a VIF value of 10.45 for feature X indicates that the variance of the regression coefficient for variable X is 10.45 times higher than it would be if X were not correlated with other features in the model.
* The standard error of a regression coefficient that you often see in an output when you run a regression model is the square root of the variance. Thus, a VIF of 10.45 means the standard error is √10.45 ≈ 3.23 times larger than when feature X is not collinear with other features in the model. It is important not to include features with high VIFs into the model as larger standard errors mean that regression coefficients are less accurate. This could result in a final model with high variance or less generalization ability to unseen data.
* You may then wonder what to do if a feature in your dataset suffers from a high VIF. Just like traditional linear regression, there are several techniques you can use for machine learning linear regression such as factor analysis (e.g., principal axis factoring, exploratory factor analysis, confirmatory factor analysis), variable selection (e.g., forward selection, backward elimination, or stepwise regression - similar to traditional statistical methods), or use previous literature in your specific area to conceptually inform your feature selection. Another technique that appears in both traditional statistics and ML, but is more discussed in the ML world for dealing with feature selection, is regularization.

## Regularization

Regularization is the process of adding penalty terms (i.e., lamnda, alpha)  into model fitting to prevent overfitting from multicolinearity issues, especially when we have too many predictors in the model, but not every predictor is meaningful. There are three major types of regularization: ridge, lasso, and elastic net (the combination of both ridge and lasso).

### 1. Ridge Regression

For ridge regression, we add a penalty term (lambda, λ) to every coefficient in the model. Imagine if we have *p* coefficients (i.e., p predictors), then the penalty term integration can be written as the following equation (keep in mind that we do not add this penalty term to the intercept, only predictors!):


<img width="118" alt="Screen Shot 2024-07-01 at 2 46 08 PM" src="https://github.com/KayChansiri/LinearRegressionML/assets/157029107/be9d8fea-18cc-4073-98bc-500698a4bb1e">

When we fit the penalty term into a loss  function, the function becomes something like this:

<img width="401" alt="Screen Shot 2024-07-01 at 2 46 58 PM" src="https://github.com/KayChansiri/LinearRegressionML/assets/157029107/aa653892-8a84-4990-a1d5-d561899adedb">

Technically, λ can take any positive value between 0 and ∞. If we set λ = 0, the loss function of ridge regression would end up in the same form as the loss function of a typical regression.

If you still don't quite understand how adding penalty terms alters the shape of your data and regression solution, imagine trying to fit a model to your dataset where the best β<sub>1</sub> that could reduce SSE is quite large. The shape of the data might look like the plot below. 

-- plot -- 

However, the goal of machine learning is to produce a model that can best predict future datasets. Simply put, we want a model whose shape is generalizable. Imagine you have to explain to a minor ethnicity islander who has never experienced the outside world what a cup is. You would want to show them picture A below, not picture B, as picture A is more like a cup according to a global standard.

--Insert pictures of a normal-looking cup and a weird cup.

Now apply the cup metaphor to the beta plot previously. if we penalize β<sub>1</sub> by multiplying a λ term to the coefficient, the shape of your dataset in a 3D dimensional space would look more general and less specific to your current data only. It would look something more the plot below on the right. Notice that the scale of β<sub>1</sub> is smaller after being penalized, ranging from , compared to before being penalized, of which the coefficients range from . This is because the original coefficient of β<sub>1</sub> was quite large, contributing to a more significant decrease in its scale during the adjustment process of the loss function to minimize loss. 


Keep in mind that in reality, your dataset will have more than the intercept and one predictor. Thus, the 3D visual I inserted above is for simplicity in demonstration. In the real world, your data could have hundreds of dimensions that would be too challenging for human eyes to comprehend!

At the beginning, I mentioned standardizing your variables. This is very important for ridge regression. If you don't standardize, ridge regression's penalty will be amplified for the coefficients of those variables with a larger range of values (e.g., age will be penalized more than the number of flights per week). This could result in inaccurate model fitting as age gets penalized not because it causes variance but because it is naturally on a larger scale before standardization.

Finally, just like other hyperparameters of ML algorithms, you can fine-tune λ. You can use cross-validation to test different values of λ with an increment as small as 0.01 if you have enough computational power and time. Note that ridge regression will shrink coefficients to close to zero but will never be zero.


### 2. Lasso Regression

The concept of lasso regression is quite similar to ridge regression. The only difference is that lasso applies a different penalty term to the loss function by extracting the absolute values of each coefficient instead of its raw values. Thus, some variables may have zero regression coefficients after we apply the penalty term. See the equation below:


<img width="571" alt="Screen Shot 2024-07-01 at 5 02 49 PM" src="https://github.com/KayChansiri/LinearRegressionML/assets/157029107/bf378ba3-71c3-42ce-88c1-48b0151942f7">

Now you may have a question about when to use ridge or lasso regression as these two functions work pretty closely. I would suggest that if you don’t want to completely exclude any features from the model (say if you do not have a large number of predictors and all predictors are conceptually meaningful), you may want to use ridge regression. However, if you have a large dataset, for instance, if you work with World Bank data with tens of thousands of predictors, using lasso regression to force certain features to be zero should be fine to ensure the model has generalizability and fewer variance issues.

### 3. Elastic Net

Elastic net simply combines ridge and lasso regression into one by mixing them together with a weighted average (alpha) as you can see in the equation below:

Unlike ridge and lasso regression, we have two parameters to fine-tune in Elastic Net: α and λ. We can consider all possible combinations of these two hyperparameters and try to find the optimal combination using cross-validation techniques.

Elastic net is useful when you want to deal with variance issues by suppressing the coefficients of a certain predictor with large coefficients and still want to perform feature selection if you have a very large dataset with too many predictors by forcing some of them to be zero.

<img width="618" alt="Screen Shot 2024-07-01 at 5 03 53 PM" src="https://github.com/KayChansiri/LinearRegressionML/assets/157029107/03c64fc5-1087-40d8-ad83-d2860c167ca6">

For the final note before I show you a real-world example, regularization is not a process particular to regression. In some other ML algorithms (e.g., neural networks and support vector machines), we use regularization as well.

## Real-World Example

Now that you understand the basics of linear regression from an ML perspective, let’s take a look at a real-world example.

The dataset I use today consists of features related to airline customers and their flight histories for a specific airline. Let's call that ML Airline!. Each record represents a unique customer, identified by the `customer_id`. The dataset includes demographic information, detailed flight information, previous flight cancellations, and specific reasons for cancellations as you can see below. The outcome variable, `customer_flight_frequency`, measures the frequency of a customer's flights per year, ranging from 0 to 100. Our goal is to indicate factors contributing to customers' flying habits and loyalty to the airline.

```ruby
data.columns
```

<img width="641" alt="Screen Shot 2024-07-01 at 5 22 30 PM" src="https://github.com/KayChansiri/LinearRegressionML/assets/157029107/37bf0053-91a9-4ccd-8a37-4b3d9561eff6">

The first step before we jump right into the ML process is to standardize all continuous features first.

```ruby
#standardize continuous variables 
from sklearn.preprocessing import StandardScaler

# Select the columns to standardize
columns_to_standardize = ['days_since_last_flight', 'customer_age', 'number_of_bookings', 'number_of_flights']

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the selected columns
data[columns_to_standardize] = scaler.fit_transform(data[columns_to_standardize])
```

Let's check for the VIF:

```ruby
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# Function to calculate VIF
def calculate_vif(data):
    vif = pd.DataFrame()
    vif["Variable"] = data.columns
    vif["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    return vif


# Remove non-predictor columns
X = data.drop(columns=['customer_id', 'flight_start_date', 'first_flight_date', 'customer_flight_frequency'])

# Check VIF
vif = calculate_vif(X)
print(vif)

# Add a constant to the model (intercept)
X = sm.add_constant(X)

# Define the target outcome
y = data['customer_flight_frequency']

```


<img width="434" alt="Screen Shot 2024-07-01 at 6 05 14 PM" src="https://github.com/KayChansiri/LinearRegressionML/assets/157029107/bef5c90b-86fa-47f9-bf19-f348b1b9f44c">


According to the VIF output,  flight_type_first_class, number_of_flights, and cancellation_reason_technical have high VIFs, although the values might not be that high if we use a libearl cutoff point (10). The values indicate that the coefficients of those three variables are likely inflated about 10 times compared to if those are not correlated with other predictor in the models. As this is just a demo, I will just leave the features there and do not perform any dimensional reduciton techniques as those could take time to explain and be even another long post that deseves their own airtime!


Now the next step is to check the assumptions such as linearity, normality, independence of errors, and heterodasceity. To do that, we have to fit the model first. At this stage, you can just fit a traditional stat model. No need to seperate the data into traning/testing set yet.

```ruby
# Fit the linear regression model
model = sm.OLS(y, X)
result = model.fit()

# Print the summary of the model
print(result.summary())
```


Then check for linear modeling assumption: 

```ruby
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro, kstest, levene


# Assumption Checks

# 1. Linearity
plt.figure(figsize=(12, 6))
sns.regplot(x=result.fittedvalues, y=result.resid, lowess=True, line_kws={'color': 'red'})
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Linearity Check: Fitted Values vs Residuals')
plt.show()

# 2. Independence of errors
# Durbin-Watson test
from statsmodels.stats.stattools import durbin_watson
dw = durbin_watson(result.resid)
print(f'Durbin-Watson statistic: {dw}')

# 3. Normality of errors
plt.figure(figsize=(12, 6))
qqplot(result.resid, line='s')
plt.title('Normal Q-Q Plot')
plt.show()

# Shapiro-Wilk test
shapiro_test = shapiro(result.resid)
print(f'Shapiro-Wilk test: {shapiro_test}')

# Kolmogorov-Smirnov test
ks_test = kstest(result.resid, 'norm')
print(f'Kolmogorov-Smirnov test: {ks_test}')

# 4. Homoscedasticity 
plt.figure(figsize=(12, 6))
sns.regplot(x=result.fittedvalues, y=np.sqrt(np.abs(result.resid)), lowess=True, line_kws={'color': 'red'})
plt.xlabel('Fitted values')
plt.ylabel('Square root of Abs(Residuals)')
plt.title('Homoscedasticity Check: Fitted Values vs Sqrt(Abs(Residuals))')
plt.show()

# Levene's test for homogeneity of variance
_, pvalue = levene(result.fittedvalues, result.resid)
print(f'Levene’s test p-value: {pvalue}')


```

According to the plot below testing the linearity assumption, the residuals (errors) are not randomly distributed around zero as they should be, indicating that the data is likely non-linear. The Durbin-Watson statistic < 2  alsoindicates positive autocorrelation among residuals, which breaks the assumption of linear modeling that errors should be independent of each other.


<img width="1095" alt="Screen Shot 2024-07-01 at 6 30 41 PM" src="https://github.com/KayChansiri/LinearRegressionML/assets/157029107/10ad91b4-8d11-4de2-b606-6ce0e9f98c60">


For normality of errors (not the observed values themselves, as many people tend to misunderstand this point), the deviation of the dots from the line suggests non-normality. The significant Kolmogorov-Smirnov test also suggests non-normality:


<img width="1107" alt="Screen Shot 2024-07-01 at 6 31 45 PM" src="https://github.com/KayChansiri/LinearRegressionML/assets/157029107/84884591-9b79-4694-a972-836a4d867283">


For homoscedasticity, the unequal spread of the residuals across all levels of the fitted values and the significant Levene’s test p-value suggest heteroscedasticity or the inconstant variance of errors: 

<img width="1082" alt="Screen Shot 2024-07-01 at 6 32 41 PM" src="https://github.com/KayChansiri/LinearRegressionML/assets/157029107/86c64f17-f2ed-47d3-b912-96fdf169658d">


So many assumption fails indicate that the current dataset is likely not explained by a linear algorithm. However, I will keep going just for the purpose of demonstration.

Now let's get to the most exciting part, which is to build a model. We will start with a lasso regression first:

### 1. Lasso Regression 

We will start with splitting the data into a traning and testing set and then conduct a cross-validation technique to get the best lamda value. If you are not familiar with the technique, feel free refer back to this post I wrote. 


```ruby

from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with standard scaling and LassoCV
pipeline = make_pipeline(
    StandardScaler(),
    LassoCV(cv=5, random_state=42)  # 5-fold cross-validation
)

# Fit the model
pipeline.fit(X_train, y_train)

# Get the best lamda value
lasso = pipeline.named_steps['lassocv']
print(f"Best lambda value: {lasso.lamda_}")

```

In this code, I did not define the range of lamda that I want too test. The default range of alpha values tested by LassoCV in the code is automatically chosen by the algorithm. The default behavior typically tests a wide range of alpha values on a logarithmic scale. However, you can specify a range of alpha values to be tested by using the code below:

```ruby

pipeline = make_pipeline(
    StandardScaler(),
    LassoCV(alphas=np.logspace(-6, 6, 100), cv=5, random_state=42)  # 5-fold cross-validation
)

```

In the code above, np.logspace(-6, 6, 100) creates 100 alpha values ranging from 10−610−6 to 106106 on a logarithmic scale. You can adjust the range and the number of lamda values according to your objective and computational power that you have. Using the default behavior, the best lambda value us 11.28471657029729. Notice that  the suggested lamnda is quite big, indicating a strong regularization. I am not surprised is this data is non-linear according to the assumption tests. Trying to fit the linear model to non-linear data would result in a high variance model. Thus, the model chooses a large lambda value to suppress coefficients, which in turn reduce variance. Using this big lamdda, most coefficients are pushed to exactlty zero.

Let's test the model on the testing set:

```ruby
y_pred = pipeline.predict(X_test)
r2_score = pipeline.score(X_test, y_test)
print(f"R^2 score on the test set: {r2_score}")

# Optional: Print out the feature names with their corresponding coefficients
feature_names = X.columns
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
print(coef_df.sort_values(by='Coefficient', ascending=False))
```
The output: 

<img width="420" alt="Screen Shot 2024-07-01 at 6 58 48 PM" src="https://github.com/KayChansiri/LinearRegressionML/assets/157029107/f714d081-c06b-4c84-8984-8847ba1b602a">

As predicted, so many coefficients are put to exactly zero.
