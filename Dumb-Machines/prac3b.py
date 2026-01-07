import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, Lasso
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import SelectFromModel

# Step 1: Create dataset
np.random.seed(0)
n = 200
X1 = np.random.normal(0, 1, n)
X2 = X1 * 0.95 + np.random.normal(0, 0.1, n)
X3 = X1 * 0.5 + np.random.normal(0, 1, n)
X4 = np.random.normal(0, 1, n)
X5 = np.random.normal(0, 1, n)
y = 3.0 * X1 + 1.5 * X4 + np.random.normal(0, 0.8, n)

df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5, 'y': y})
print("=== Dataset Head ===")
print(df.head())

# Step 2: Train-Test Split
X = df[['X1','X2','X3','X4','X5']]
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 3: OLS (Statsmodels)
X_train_sm = sm.add_constant(X_train)
ols_model = sm.OLS(y_train, X_train_sm).fit()
print("\n=== OLS Summary ===")
print(ols_model.summary())

# Step 4: Check VIF
vif_data = pd.DataFrame()
vif_data['Feature'] = X_train.columns
vif_data['VIF'] = [variance_inflation_factor(X_train.values, i) for i in
range(X_train.shape[1])]
print("\n=== VIF Values ===")
print(vif_data)

# Step 5: Linear Regression
lr = LinearRegression().fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("\n=== Linear Regression ===")
print("Coefficients:", lr.coef_)
print("Intercept:", lr.intercept_)
print("R2:", r2_score(y_test, y_pred_lr))
print("RMSE:", mean_squared_error(y_test, y_pred_lr, squared=False))

# Step 6: Ridge Regression
alphas = np.logspace(-3, 3, 50)
ridge = RidgeCV(alphas=alphas, cv=5).fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
print("\n=== Ridge Regression ===")
print("Alpha:", ridge.alpha_)
print("Coefficients:", ridge.coef_)
print("R2:", r2_score(y_test, y_pred_ridge))
print("RMSE:", mean_squared_error(y_test, y_pred_ridge, squared=False))

# Step 7: Lasso Regression
lasso_cv = LassoCV(cv=5, max_iter=5000).fit(X_train, y_train)
y_pred_lasso = lasso_cv.predict(X_test)
print("\n=== Lasso Regression ===")
print("Alpha:", lasso_cv.alpha_)
print("Coefficients:", lasso_cv.coef_)
print("R2:", r2_score(y_test, y_pred_lasso))
print("RMSE:", mean_squared_error(y_test, y_pred_lasso, squared=False))

# Step 8: Feature Selection
selector = SelectFromModel(Lasso(alpha=lasso_cv.alpha_))
selector.fit(X_train, y_train)
selected = X.columns[selector.get_support()]
print("\n=== Selected Features ===")
print(selected.tolist())
lr_sel = LinearRegression().fit(X_train[selected], y_train)
y_pred_sel = lr_sel.predict(X_test[selected])
print("\n=== Final Model ===")
print("Coefficients:", lr_sel.coef_)
print("Intercept:", lr_sel.intercept_)
print("R2:", r2_score(y_test, y_pred_sel))
print("RMSE:", mean_squared_error(y_test, y_pred_sel, squared=False))

# Step 9: Predictions Compare
print("\n=== Sample Predictions (first 10) ===")
for i in range(10):
    print(
        f"Actual: {y_test.iloc[i]:.2f}", f"Pred(LR): {y_pred_lr[i]:.2f}",
        f"Ridge: {y_pred_ridge[i]:.2f}", f"Lasso: {y_pred_lasso[i]:.2f}",
        sep=" | "
    )
print("\n
Done! Multiple Linear Regression executed successfully.")
