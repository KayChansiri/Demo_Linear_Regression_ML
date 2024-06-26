# Linear Regression from a Machine Learning Perspective

Let's talk about one of the simplest machine learning (ML) algorithms today: regression. Just like other ML algorithms, regression can predict both continuous outcomes (i.e., linear regression) and categorical outcomes (i.e., logistic regression). Today, we'll focus on linear regression.

## Differences Between Traditional Statistics and Machine Learning

Before we get started, let's briefly discuss the differences between linear regression from a traditional statistics standpoint versus a machine learning standpoint.

### Traditional Statistics
* **Dataset Usage**: Uses only one dataset without dividing it into testing, training, and validation sets for modeling.
* **Data Preparation**: Involves standard data preparation processes, such as encoding variables, checking for and imputing missing data, and testing linear assumptions (e.g., linearity, normality, independence of errors, heteroscedasticity). Linear regression from a traditional statistics perspective also involves checking for multicollinearity using methods such as VIF (more than 4 if being conservative, and more than 10 if being more liberal) or correlation tests (examining if any pairs of variables are correlated more than 0.9). If multicollinearity issues exist, feature selection processes, such as dimensionality reduction (e.g., exploratory factor analysis, confirmatory factor analysis), centering (especially when interaction terms are included in the model), or excluding predictors with conceptually less importance may be performed.

  As linear regression relies on the ordinary least squares approach and gradient descent methods to identify the best regression coefficients   that reduce error variances of the model, standardizing predictors to the same scale is also vital to ensure the optimal solution.If your       features are on different scales, interpreting the model fit and its predictability would be challenging. For example, imagine you are          predicting customer satisfaction towards an airline service on a scale of 1 to 5 (this is your y or target outcome) by customer age (ranges     from 5 to 100) (X1) and customers’ frequency in flying per week (ranges from 0 to 7) (X2). An increase in one unit of the coefficient of your   X1 would be different from an increase in one unit of the coefficient of your X2, and X1 can increase up to 100, whereas X2 can increase up to  7. To know which feature yields more impact on the outcome, you have to standardize them to be on the same scale. While standardization is      important for linear regression, keep in mind that it might not be as crucial for some other ML algorithms, such as random forests or decision   trees, where different scales of features do not matter as much since the algorithm relies on decision boundaries in fitting the model.

* **Model Emphasis**: Emphasizes inference and determining if a model explains past data well, rather than predicting future data.
  
### Machine Learning
* **Dataset Usage**: Performs data separation into testing, training, and validation sets to ensure the model works well with unseen data. For projects with less computational power or budget, separating the data into testing and training sets without a validation set may be sufficient.
* **Data Preparation**: Similar to linear regression from a traditional standpoint, the ML approach should ensure that linear assumptions are met, missing data are imputed or excluded, all continious variables are standardized, and no multicollinearity issues exist. Unlike a traditional statistical approach, linear regression from an ML perspective uses regularization techniques (e.g., ridge, lasso, elastic net) and fine-tunes hyperparameters (e.g., lambda and alpha) to deal with multicollinearity or variance issues from meaningless variables. The process also involves K-fold cross-validation to adjust the model's hyperparameters before deployment. You will understand these concepts better later in the post.
* **Model Emphasis**: Unlike traditional statistics that focus on explaining existing data, ML focuses on building robust models that can generalize well to new data.

<img width="642" alt="Screen Shot 2024-06-25 at 3 11 42 PM" src="https://github.com/KayChansiri/LinearRegressionML/assets/157029107/95f956ba-fe96-4475-a546-8f09163d187e">

## Loss Function
The basis of linear regression is the loss function. To understand how linear regression works, you first have to understand how the loss function works. The loss function is a mathematical function that tries to minimize the distance between observed and predicted values so that our model is as accurate as possible. The function can be represented by the following equation:

<img width="357" alt="Screen Shot 2024-06-25 at 8 21 59 PM" src="https://github.com/KayChansiri/LinearRegressionML/assets/157029107/3aeb517a-efab-452b-a340-d6e91bbd8982">

Here SSR represents the sum of squared residuals. X and Y values are the observed data (i.e., the X and Y columns of the data you collected, and {β0,β1} are the unknown parameters. The goal of the loss function, as an optimization process, is to find the set of {β0,β1} that provides the minimum value of this function. If you are still confused about the idea, let's start by looking at a typical linear regression model equation below:

<img width="158" alt="Screen Shot 2024-06-25 at 7 21 36 PM" src="https://github.com/KayChansiri/LinearRegressionML/assets/157029107/c189b145-1833-4e8c-8fbd-3b2fcf444f8b">

Here, *i* refers to sample *i* . Say you have a dataset with 100 samples or observations, and your β<sub>0</sub> and  β<sub>1</sub>  are 2.5 and 0.8, respectively. If your sample *i*  has  X = 20, the model would predict their *y* as 2.5 + 0.8(20) = 18.5. However, if their true y score is 20, it means that our error term is 20 - 18.5 = 1.5. Remember, we have 100 observations in the dataset, so we will need to aggregate them in some way to get the total amount of errors. This can be calculated using different strategies, such as the sum of absolute residuals (SAR) and SSR. SSR is more preferable as SAR has absolute values that are mathematically more challenging to deal with.

Now, imagine we use SSR for all samples in the dataset to get the total errors of the model based on our β<sub>0</sub> and  β<sub>1</sub>, which are 2.5 and 0.8. One challenge is that we do not know if β<sub>0</sub> and  β<sub>1</sub> are the best coefficient estimates, so we have to try out many pairs of β<sub>0</sub> and β<sub>1</sub> and see what is the best that we could get, meaning what pair of β<sub>0</sub> and  β<sub>1</sub> could best reduce the sum of squared errors.

If you try different values of β<sub>0</sub> and  β<sub>1</sub> ranging from -10 to 10 and plot the beta values against the SSR, you will get something like the figure below. 


<img width="592" alt="Screen Shot 2024-06-25 at 7 41 10 PM" src="https://github.com/KayChansiri/LinearRegressionML/assets/157029107/ffc21698-3249-4ea2-9a2d-1f13e9d25234">

The lowest point is called the global minimum, which is the point where the values of β<sub>0</sub> and  β<sub>1</sub> yield the least error in the model. In other words, this is the point where the regression coefficient values provide the optimal solution. You may hear people use some alternative phrases to refer to the global minimum, such as the point where the model converges or where we find a convex surface.

Behind the scene when you run a linear regression model, the machine uses linear algebra to the find the global minimum, which can be simply written in to a matrix operation form as below. Now keep in mind that in this matrix operation example, we have only one predictor  β<sub>1</sub>  which I set in to range from 1 to 10, and the intercept (β<sub>0</sub> ), which I set it to be 1 for simplicity.  In the real world, you will 100% have more than one predictor and will get a more complex matrix operation equation. 


<img width="299" alt="Screen Shot 2024-06-25 at 8 09 27 PM" src="https://github.com/KayChansiri/LinearRegressionML/assets/157029107/424e6eba-7d7b-4e49-a279-9efeafcecf3e">


Now that we have the matrix form, we can find the values of the betas by performing some matrix transposition and inversion, as shown in the equation below:

<img width="206" alt="Screen Shot 2024-06-25 at 8 15 31 PM" src="https://github.com/KayChansiri/LinearRegressionML/assets/157029107/da19f29d-eba3-4e96-b61b-c87ed6c72ef6">

Note that in linear regression, the equation above provides a closed-form solution as we have a global minimum due to the convex nature of the loss function. For some other types of machine learning models that are non-linear (e.g., neural networks), you may end up having many local minima, as shown in the picture below. This can make the interpretation of the model more complicated. For those types of algorithms, we use an iterative process like gradient descent to find the best spot that is as close to the lowest point as possible, thereby obtaining the optimal beta values. These types of equations do not have a closed-form solution because their loss functions are non-convex and can have multiple local minima.



