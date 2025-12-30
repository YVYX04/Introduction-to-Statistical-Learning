# Linear Regression
## The Basic Algorithm of Supervised Learning

## 1. Introduction

Linear regression is one of the simplest, yet most fundamental, algorithms in supervised learning, and it is specifically designed to address *regression* problems. To be precise, supervised learning comprises two main classes of prediction tasks: *regression* and classif*ication, which differ by the nature of the variable to be predicted.

A regression task consists in predicting a continuous numerical variable, such as fuel efficiency (`mpg`), income, or temperature. By contrast, a classification task aims to assign observations to a finite set of discrete categories (e.g. dog vs. cat). A common special case is *binary classification*, where the target variable takes only two possible values, often encoded as $0$ and $1$ (for instance, whether a patient tests positive or negative for a given disease).

In this chapter, the focus is exclusively on regression problems. Linear regression provides a baseline yet powerful framework for modeling the relationship between a set of explanatory variables and a continuous response variable, and serves as a conceptual and mathematical foundation for many more advanced methods in statistical learning. Finally, I repeat that most of my writings are heavily inspired by Hastie et al. (2009), unless explicitely mentioned otherwise.

## 2. The Simple Linear Regression

### Model Specification

The simple linear regression (SLR) framework is very straightforward.
our purpose is to find a relationship that allows us to explain the behavior of $Y$, the target (random) variable, in function of $X$, the independent (random) variable. Therefore, our model takes the form:

$$
Y = \beta_0 + \beta_1 X + \varepsilon
$$

where $\beta_0$ and $\beta_1$ are the *coefficients* of the model and $\varepsilon$ is the random error term with expected value of $0$. Once this model has been established, one can legitimaly ask the question, how do we produce predictions? First, we use *training data* to estimate the model's coefficients: $\hat{\beta}_0$ and $\hat{\beta}_1$ (the hat denotes estimated quantities). Then, we predict the value $\hat y$ with a new (e.g., the test set) set of predictors $x$:

$$
\hat y = \hat{\beta}_0 + \hat{\beta}_1 x
$$

### Computing the Coefficients

Once $\hat y$ has been predicted, one can compute the squared sum of the *residuals* (RSS) as:

$$
\mathrm{RSS} = \sum_{i = 1}^{n} e_i^2 = \sum_{i = 1}^{n} (y_i - \hat{y}_i)^2
$$

And to estimate the two coefficients $\hat{\beta}_0$ and $\hat{\beta}_1$, we now minimize the RSS (the MSE since we take the mean) on the training data:

$$
\min_{\hat{\beta}_0, \hat{\beta}_1} \quad \frac{1}{n} \sum_{i = 1}^{n} (y_i - \hat{y}_i)^2
$$

which is the same as:

$$
\min_{\hat{\beta}_0, \hat{\beta}_1} \quad \frac{1}{n} \sum_{i = 1}^{n}
(y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i)^2
$$

and with some calculus steps, one can easily find that:

$$
\hat{\beta}_1 = \frac{\overline{xy} - \overline{x} \cdot \overline{y}}{\overline{x^2} - \overline{x}^2}, \qquad
\hat{\beta}_0 = \overline{y} - \hat{\beta}_1 \overline{x}
$$

where $\overline{xy} = 1/n \cdot \sum_{i=1}^{n} x_i y_i$, $\overline{x} = 1/n \cdot \sum_{i=1}^{n} x_i$ (same for $\overline{y}$), then $\overline{x^2} = 1/n \cdot \sum_{i=1}^{n} x_i^2$, and finally $\overline{x}^2 = \bigl( 1/n \cdot \sum_{i=1}^{n} x_i \bigr)^2$.

### Synthetic Example

To visually and computationally demonstrate the results we obtained above, I create a synthetic data set composed of two variables: `life_exp`, and `yrs_school` (based on real empirical observations). This data set is available [here](data/life_expectancy_data.csv) and it is constructed as:

```text
   yrs_school   life_exp
0   13.578595  52.713117
1   17.792642  57.841715
2   10.167224  47.758930
3    0.602007  33.257639
4   15.585284  57.082617
```

Now, we are going to assume that the variable `yrs_school` is our random variable $X$ and `life_exp` is the random variable $Y. We first code a function to compute the coefficient of the SLR:

```python
def SLR_coefficients(X, y):
    """
    Calculate the coefficients for Simple Linear Regression.
    
    Parameters:
    X (array-like): Independent variable.
    y (array-like): Dependent variable.
    
    Returns:
    tuple: Intercept and slope of the regression line.
    """
    n = len(X)
    
    # Calculate slope
    num = np.sum(X * y) - n * np.mean(X) * np.mean(y)
    den = np.sum(X**2) - n * (np.mean(X))**2
    b1 = num / den  # slope

    # Calculate intercept
    b0 = np.mean(y) - b1 * np.mean(X)  # intercept
    
    return b0, b1
```

And if we compute the coefficients, we obtain:

```text
SLR Coefficients
Intercept (b0): 37.494628041499226
Slope (b1): 1.0155605359752424
```
and since this is not very compelling alone, we also realize a regression line with those exact same coefficients to see how well we fit the data (Fig. 1).

<figure style = "text-align: center;">
    <p align="center">
        <img src = "figures/slr_life_expectancy.png" style = "width: 60%;">
        <p align="center">
            <strong>Fig. 1</strong> Manually computed regression line on synthetic data. (Simple linear regression).
        </p>
    </p>
</figure>

And as one can see, the fit seems to be the best possible given the model specification.
Now that we have our estimated parameters, it could be tempting to interpret this coefficient
and say something like "without any education, a typical human being lives until $37$ years old
and subsequently, any additional year of schooling increase life expectancy by $1.016$ years".
This leads us to the next section.

### A Machine Learning Scientist is not an Econometrician 

It is crucial to emphasize that, in this simulated machine-learning setting, the objective is *prediction* rather than *causal inference*. The focus is therefore on out-of-sample performance, not on the substantive interpretation of parameter estimates. In this sense, the machine-learning scientist primarily asks whether the model generalizes well to unseen data, whereas the econometrician is concerned with whether a coefficient admits a defensible causal interpretation.

Such an interpretation requires the satisfaction of a strict set of assumptions (including correct functional form, random sampling, sufficient variation in the regressors, exogeneity, and homoskedasticity) without which no causal claim can be credibly sustained. As these conditions are not the object of the present analysis, issues such as coefficient bias, efficiency, and hypothesis testing are deliberately set aside. Readers interested in a rigorous treatment of causal interpretation within the linear regression framework are referred to the freely available and interactive textbook  [*Introduction to Econometrics with R*](https://www.econometrics-with-r.org/index.html).

### Quality of the Predictions

While we do not pretend to be econometricians, we can still, and in fact must, assess the quality of the fit. For this, we primarily rely on the *residual standard error* (RSE) defined as:

$$
\mathrm{RSE} = \sqrt{\frac{1}{n-2}\sum_{i=1}^{n} \bigl(y_i  - \hat{y}_i \bigr)^2}
$$

The fact that we divide by $n-2$ instead of $n$ is to account for the degree of freedom. If we were to divide by $n$ we would underestimate the noise in our regression, on average. Another commonly used metric is the $R^2$ score:

$$
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \overline{y})^2} = 1 - \frac{\mathrm{RSS}}{\mathrm{TSS}}
$$

In substance, the $R^2$ score $\in [0, 1]$ describes the share of variation in the target variable that we are able to explain with our predictor. It is straightforward to see that if our predictor can explain no variance at all, we will have $\mathrm{RSS} = \mathrm{TSS}$, meaning that not a single fraction of the total variance in $y$ was explained by the regression; this entails that $R^2$ would equal $0$.
Now, we could ask, what is a good benchmark for the $R^2$ score? Well the true answer is that there are no unique benchmark and it depends vastly on the nature of the problem. For instance, if we are observing chemical reactions grounded in the laws of physics and they predict a linear relation, we should observe an $R^2$ score extremelly close to $1$. On the other hand, if we are in the realm of quantitative finance and we try to understand if a momentum based long-short strategy can earn abnormal returns on the market, an $R^2$ score of $0.1$ would be quite a *tour de force*. In the next section I
discuss the multiple linear regression (MLR) problem, in which we use several predictors simultaneously to predict the response.

## 3. Multiple Linear Regression

to be written. 

## 4. Further Considerations

to be written.


## References

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *An introduction to statistical learning*.