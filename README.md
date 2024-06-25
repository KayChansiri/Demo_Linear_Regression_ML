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



