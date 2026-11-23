import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# - För arbetet att strukturera klasserna har jag haft stor hjälp av boken 'Python från början' av Jan Skansholm.
# - För felsökning av kod i klasserna har jag tagit hjälp av Microsoft Copilot. 
# - Figurerna för att visa resultatet av regressionanalyserna var svåra att få till och jag har behövt flera källor:
# - https://docs.python.org/3.14/index.html, https://www.w3schools.com/python/default.asp, 
# - 'Python från början' samt kursmaterialet med videos och textdokumentation.

class BloodPressureInvestigator:
    """Analyserar ett dataset med hälsodata för att undersöka olika faktorers association med systoliskt blodtryck. 
    
    Klassen kan användas för:
    
    - Enkel linjär regression, med enstaka förklarande variabel
    - Multipel linjär regression, med flera förklarande variabler
    - Sammanfattning av de regressionsanalyser som gjorts
    - Visualisera regressionslinje och observerade data för enkel regression
    - Visualisera modellens prediktion mot faktiskt utfall i data (x-axel: observerat blodtryck, y-axel: modellens prediktion)
    - Visualisera residualer från multipel regression

    Som parameter tar klassen in ett dataset i form av pandas.DataFrame
    
    """

    def __init__(self, df):
        self.df = df
        self.model = None
        self.predictors = None

    def __repr__(self):

        if self.model is None:
            return "BloodPressureInvestigator(otränad modell)"

        return f"BloodPressureInvestigator(prediktorer={self.predictors})"

    def _check_model(self):
        if self.model is None:
            raise ValueError("Ingen modell har tränats ännu.")

    def simple_regression(self, predictor):
        X = self.df[[predictor]]
        y = self.df["systolic_bp"]

        self.model = LinearRegression()
        self.model.fit(X, y)

        self.predictors = [predictor]

        return {
            "predictor": predictor,
            "intercept": self.model.intercept_,
            "slope": self.model.coef_[0],
            "R2": self.model.score(X, y)
        }

    def multiple_regression(self, predictors):
        X = self.df[predictors]
        y = self.df["systolic_bp"]

        self.model = LinearRegression()
        self.model.fit(X, y)

        self.predictors = predictors

        return {
            "predictors": predictors,
            "intercept": self.model.intercept_,
            "coefficients": dict(zip(predictors, self.model.coef_)),
            "R2": self.model.score(X, y)
        }

    def print_summary(self):
        self._check_model()
        
        X = self.df[self.predictors]
        y = self.df["systolic_bp"]

        print('\n--- Sammanfattning av regressionsanalys ---')
        print(f"Prediktorer: {' | '.join(self.predictors)}")
        print(f'Intercept: {self.model.intercept_:.3f}')
        print('\nKoefficienter:')
        for predictor, coef in zip(self.predictors, self.model.coef_):
            print(f'{predictor:<12} {coef:.3f}')

        print(f'\nR2: {self.model.score(X, y):.3f}')

    def plot_regression(self, predictor):
        self._check_model()

        if len(self.predictors) != 1:
            raise ValueError("plot_regression() kan endast användas efter simple_regression().")

        X = self.df[predictor]
        y = self.df["systolic_bp"]

        y_pred = self.model.predict(self.df[[predictor]])

        # Sortera efter prediktorn
        sort_idx = X.argsort()

        plt.figure(figsize=(8, 5))
        plt.scatter(X, y, alpha=0.6, label="Observationer")
        plt.plot(X.iloc[sort_idx], y_pred[sort_idx], color="red", linewidth=2, label="Regressionslinje")
        plt.xlabel(predictor)
        plt.ylabel("Systoliskt blodtryck")
        plt.title(f"Blodtryck vs {predictor}")
        plt.legend()
        plt.show()

    def plot_predictions(self):
        self._check_model()
        
        X = self.df[self.predictors]
        y = self.df["systolic_bp"]

        y_pred = self.model.predict(X)

        plt.figure(figsize=(6, 6))
        plt.scatter(y, y_pred, alpha=0.6)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linestyle="--", linewidth=2)
        plt.xlabel("Observerat blodtryck")
        plt.ylabel("Predikterat blodtryck")
        plt.title("Träffsäkerhet multipel regression")
        plt.show()

    def plot_residuals(self):
        self._check_model()

        X = self.df[self.predictors]
        y_true = self.df["systolic_bp"]
        y_pred = self.model.predict(X)
        residuals = y_true - y_pred

        plt.figure(figsize=(8, 5))
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color="red", linestyle="--", label="Residual = 0")
        plt.xlabel("Predikterat blodtryck")
        plt.ylabel("Residual")
        plt.title("Residualer vid multipel regression")
        plt.legend()
        plt.show()