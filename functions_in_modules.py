import pandas as pd
import matplotlib.pyplot as plt

# - För klassens grundstruktur har jag haft stor hjälp av boken 'Python från början' av Jan Skansholm.
# - Övriga källor för denna klass är https://www.w3schools.com/python/default.asp och kursmaterialet.

class DataInvestigator:
    """ Klass för att undersöka och sammanfatta data."""

    def __init__(self, df):
        self.df = df

    def show_info(self):
        """Visar information om datatyper och eventuella saknade värden."""
        self.df.info()

    def show_duplicates(self):
        """Visar alla rader som ingår i en dublett."""
        
        duplicates = self.df[self.df.duplicated(keep=False)]

        if duplicates.empty:
            print("Inga dubletter hittades.")
        else:
            print(f"Hittade {duplicates.shape[0]} rader som ingår i dubletter:")
            print(duplicates)

    def show_describe(self):
        """Visar statistik."""
        print(self.df.describe())

    def show_head(self, n=3):
        """Visar de första raderna"""
        print("\n=== HEAD ===")
        print(self.df.head(n))

    def health_stats(self, cols):
        """Beräknar medelvärde, median, min och max för valda kolumner."""
        return self.df[cols].agg(['mean', 'median', 'min', 'max']).T

    def summary_report(self, cols=["age", "weight", "height", "systolic_bp", "cholesterol"]):
        """Skriver ut en sammanfattning vid behov."""
        
        self.show_info()
        print("\nBeskrivande statistik:")
        self.show_describe()

        print("\nFörsta raderna:")
        self.show_head()

        print("\nÖvergripande statistik:")
        print(self.health_stats(cols))