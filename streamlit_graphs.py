import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import shap
from final_movie_analysis.functions import create_X_y_before, create_X_y_opening, create_X_y_after

def graph_revenue():
    '''Shows revenue of movies'''
    df = pd.read_csv("movie_data.csv")
    # graph of the revenues
    mean_of = pd.DataFrame()
    mean_of = df.groupby('Release Date')[['Inflation Adjusted Domestic Revenue','Domestic Revenue', 'Total Box Office Revenue','International Revenue','Domestic Video Revenue']].mean()
    # 'Total Box Office Revenue','International Revenue',
    fig1, ax1 = plt.subplots()
    mean_of[['Inflation Adjusted Domestic Revenue','Domestic Revenue','Domestic Video Revenue']].plot(ax=ax1)
    plt.ylabel("Avg Inflation-Adjusted Domestic Revenue")
    plt.xticks(rotation=45)
    plt.title('Domestic Revenue over time')

    fig2, ax2 = plt.subplots()
    mean_of[['Total Box Office Revenue','International Revenue']].plot(ax=ax2)
    plt.ylabel("Avg Inflation-Adjusted Domestic Revenue")
    plt.xticks(rotation=45)
    plt.title('International and Total Revenue over time')
    return fig1, fig2


def graph_revenue_by_year():
    df = pd.read_csv("movie_data.csv")
    df['Release_Date'] = pd.to_datetime(df['Release Date'])
    df['year'] = df['Release_Date'].dt.year
    yearly = df.groupby('year')[['Inflation Adjusted Domestic Revenue','Domestic Revenue','International Revenue',
                                'Total Box Office Revenue']].mean().reset_index()
    
    fig, ax = plt.subplots()
    # Plot each line separately and give labels
    ax.plot(yearly['year'], yearly['Inflation Adjusted Domestic Revenue'],
            label='Inflation Adjusted Domestic Revenue')
    ax.plot(yearly['year'], yearly['Domestic Revenue'],
            label='Domestic Revenue')
    ax.plot(yearly['year'], yearly['Total Box Office Revenue'],
            label='Total Box Office Revenue')
    ax.plot(yearly['year'], yearly['International Revenue'],
            label='International Revenue')
    ax.set_xlabel('Year')
    ax.set_ylabel('Revenue')
    ax.set_title('Average Revenue by Year')
    plt.xticks(rotation=45)
    ax.legend()
    return fig


def graph_revenue_and_profit():
    df = pd.read_csv("movie_data.csv")
    mean_of = pd.DataFrame()
    mean_of = df.groupby('Release Date')[['Total Box Office Revenue','Profit']].mean()
    # 'Total Box Office Revenue','International Revenue',
    fig, ax = plt.subplots()
    mean_of[['Total Box Office Revenue','Profit']].plot(ax=ax)
    plt.ylabel("Money")
    plt.xticks(rotation=45)
    plt.title('Total Revenue and Profit over time')
    return fig


def season_earnings():
    '''Returns revenue by season'''
    df = pd.read_csv("movie_data.csv")
    fig, ax = plt.subplots()
    df.boxplot(column='Total Box Office Revenue', by='Season',ax=ax)
    plt.ylabel("Total Box Office Revenue")
    plt.title('Revenue by Season of Movie Release')
    return fig


def genre_earnings():
    '''Returns revenue by genre'''
    df = pd.read_csv("movie_data.csv")
    fig, ax = plt.subplots()
    df.boxplot(column='Total Box Office Revenue', by='Genre',ax=ax)
    plt.ylabel("Total Box Office Revenue")
    plt.xticks(rotation=45)
    plt.title('Revenue by Genre')
    return fig


def production_method_earnings():
    '''Returns revenue for different production methods'''
    df = pd.read_csv("movie_data.csv")
    fig, ax = plt.subplots()
    df.boxplot(column='Total Box Office Revenue', by='Production Method',ax=ax)
    plt.ylabel("Total Box Office Revenue")
    plt.xticks(rotation=45)
    plt.title('Revenue by Production Method')
    return fig


def ratings_earnings():
    '''Returns revenue by rating'''
    df = pd.read_csv("movie_data.csv")
    fig, ax = plt.subplots()
    df.boxplot(column='Total Box Office Revenue', by='MPAA Rating',ax=ax)
    plt.ylabel("Total Box Office Revenue")
    plt.xticks(rotation=45)
    return fig


def ml_graph_before():
    print("Most important features before a movie opens")
    X,y=create_X_y_before()
    return find_most_important_features_plot(X,y,1)


def ml_graph_once():
    print("Most important features right after a movie opens")
    X,y=create_X_y_opening()
    return find_most_important_features_plot(X,y,2)


def ml_graph_after():
    print("Most important features long after a movie opened")
    X,y=create_X_y_after()
    return find_most_important_features_plot(X,y,3)


def find_most_important_features_plot(X,y,XNum):
    '''Finds most important labels, from X, for predicting a target, y, shown with shap'''
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, show=False)
    ax = plt.gca()
    if XNum == 1:
        ax.set_title("Feature Importance Regarding Total Box Office Revenue Before Movie Opens")
        fig = plt.gcf()
    elif XNum == 2:
        ax.set_title("Feature Importance Regarding Total Box Office Revenue Once Movie Opens")
        fig = plt.gcf()
    elif XNum == 3:
        ax.set_title("Feature Importance Regarding Total Box Office Revenue After Movie Opens")
        fig = plt.gcf()
    return fig


def answer_question():
    print("Our research question is: What features of a movie have the most influence and which contribute most to a high revenue? (Concerning Universal movies from 2014-2024)?")
    print("\nAfter our analysis, our end answer is: The features with the most influence on the end total box office revenue change depending on whether the movie has come out and for how long.")
    print("If it has been out awhile, then its international revenue, if it just came out, its the opening weekend numbers, if it hasn't come out, then its the production budget.")
    print("If you want better odds at making a high revenue movie, then make an Adventure, Action, or Musical Movie using Digital Animation.")
    print("Make sure it is PG and that you release it in the summer.")
    print("You should also begin with as large a production budget as possible, make sure it appeals to those outside as well as inside the US, and choose the best weekend in the Summer to release it for the best opening weekend revenue possible.")

