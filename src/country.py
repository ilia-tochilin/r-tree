import pandas as pd


class Country:
    def __init__(self):
        self.df = pd.read_csv('worldcitiespop.csv', usecols=['Country', 'Population', 'Latitude', 'Longitude'])

    def get_country(self, country_name: str):
        df = self.df[self.df['Country'] == country_name][['Longitude', 'Latitude']]
        df.columns = ['x1', 'y1']
        df['x2'] = df['x1'] + 0.001
        df['y2'] = df['y1'] + 0.001
        df = df[['x1', 'x2', 'y1', 'y2']]
        return df

    def get_country_with_population_and_name(self, country_name: str, population: int):
        df = self.df.loc[(self.df['Country'] == country_name) & (self.df['Population'] > population)]
        if df.empty:
            raise ValueError("Country with given index doesn't exist")
        df.columns = ['x1', 'y1']
        df['x2'] = df['x1'] + 0.001
        df['y2'] = df['y1'] + 0.001
        df = df[['x1', 'x2', 'y1', 'y2']]
        return df

    def get_country_with_population(self, population):
        df = self.df[self.df['Population'] >= population][['Longitude', 'Latitude']]
        df.columns = ['x1', 'y1']
        df['x2'] = df['x1'] + 0.001
        df['y2'] = df['y1'] + 0.001
        df = df[['x1', 'x2', 'y1', 'y2']]
        return df
