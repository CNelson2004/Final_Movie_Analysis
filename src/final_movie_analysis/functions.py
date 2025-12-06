#transferring cells from ipynb files into proper functions 
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import lxml
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import shap


#Data Gathering
def setup_2(email:str ="cnelson1845@gmail.com",url1:str = "https://en.wikipedia.org/wiki/List_of_Universal_Pictures_films_(2010%E2%80%932019)",url2:str = "https://en.wikipedia.org/wiki/List_of_Universal_Pictures_films_(2020%E2%80%932029)",robots:int=5):
    '''Uses your email (for the header), and two urls and retrieves the response objects for those urls, also takes a wait time if its in the robots.txt'''
    user_agent:str =  f"For class web scraping assignment, email: {email}"
    headers: dict[str:str] = {"User-Agent": user_agent}
    r1:requests.Response = requests.get(url1, headers=headers)
    time.sleep(robots)
    r2:requests.Response = requests.get(url2, headers=headers)
    if r1.status_code != 200:
        raise KeyError("Couldn't reach url1")
    if r2.status_code != 200:
        raise KeyError("Couldn't reach url2")
    return r1, r2


def get_website_info(r1: requests.Response, r2: requests.Response) -> dict[str:str]:
    '''Note: Only works if you use the returned response objects from the default values in `setup_2`
    Returns a dictionary containing the title of the movie and its release year.
    For us, this is the universal studios movies from 2014-2024
    '''
    tables1 = pd.read_html(r1.text)
    tables2 = pd.read_html(r2.text)
    #getting the relevant tables (2014-2019 and 2020-2024)
    tables01 = tables1[5:10]
    tables02 = tables2[0]
    #tables02 includes some 2025 movies, so we will have to remove those later
    dfs = []
    for table in tables01:
        dfs.append(table)
    dfs.append(tables02)
    movies = pd.concat(dfs, ignore_index=True)
    #removing the unneeded notes column
    movies = movies.drop('Notes', axis=1)
    #Getting the title and release year from each column and putting it in a dictionary
    the_movies = {}
    for idx, movie in movies.iterrows():
        #getting title
        title = movie["Title"]
        #getting year
        year = movie["Release date"][-4:]
        #placing into dictionary
        the_movies[title] = year
    #removing movies from 2025
    correct_years = [2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024]
    bad_keys = []
    for key,value in the_movies.items():
        if int(value) not in correct_years:
            bad_keys.append(key)
    for key in bad_keys:
        del the_movies[key]
    return the_movies


def get_dirty_urls_for_the_numbers(the_movies:dict[str:str]):
    '''formats movie titles into urls for the-numbers, 
        and returns those urls and a dictionary with those titles linked to their release year as a string.
        Please note the the dictionary must have both the title and the year as a string'''
    urls = []
    clean_dict = {}
    base = "https://www.the-numbers.com/movie/"
    suffix = "#tab=summary"
    for title in the_movies:
        clean = title
        #removing specific characters
        clean = clean.replace(" ‡","").replace(" †","").replace(" §","").replace(" *","")
        #removing special characters
        clean = clean.replace(":","").replace("!","").replace("'","").replace(",","").replace(".","").replace("[N 1]","").replace("[a]","")
        #replacing special characters
        clean = clean.replace("/","-").replace("&","and").replace("í","i").replace("á","a")
        #replacing spaces with dashes
        clean = clean.replace(" ","-")
        #moving 'The' and 'A' and 'La' to the end of title 
        clean = re.sub(r'^The-(.+)$', r'\1-The', clean)
        clean = re.sub(r'^A-(.+)$', r'\1-A', clean)
        clean = re.sub(r'^La-(.+)$', r'\1-La', clean)
        #adding it to another dictionary to keep track of years
        year = the_movies[title]
        clean_dict[clean] = year
        formatted_year = f"-({year})"
        #creating full url with it (make sure year is still there)
        full_url = f"{base}{clean}{formatted_year}{suffix}"
        urls.append(full_url)
    return urls,clean_dict


def get_clean_and_bad_urls_for_the_numbers(urls:list[str],clean_dict: dict[str:str],email:str = "cnelson1845@gmail.com",robots:int = 5):
    '''returns the cleaned urls, and nonfunctioning dirty urls from `get_dirty_urls_for_the_numbers.
        this was placed into a second function due to the time it takes to execute.
        Thus function, once again, requires your email to remake the response object.
        Also, it needs the time from robots.txt
        There is a built in timer so you can see the progress'''
    #we need to divide up our urls into chunks of 50 so we don't overload the website
    chunk_size = 50
    n_chunks = (len(urls) + chunk_size - 1) // chunk_size  
    sections = np.array_split(urls, n_chunks)
    #This takes about 50 minutes for our 2014-2024 movie dataset
    #fixing the bad ones
    #setting up our variables
    user_agent:str =  f"For class web scraping assignment, email: {email}"
    headers: dict[str:str] = {"User-Agent": user_agent}
    marked = {}
    i = -1
    #Just for me to see times during this
    start = time.time()
    print("starting first section")
    for k,section in enumerate(sections):
        for url in section:
            i += 1
            #abiding by robots.txt
            time.sleep(robots)
            r = requests.get(url,headers=headers)
            if r.status_code == 200:
                continue
            #checking if the website broke
            if r.status_code == 503:
                print(f"503 Error: currently on i: {i} out of 232, with url: {url}")
                break
            #setting up variables
            current = url
            title_year = url.rsplit("/", 1)[-1].split("#", 1)[0]
            title = re.split(r"-\(\d{4}\)$", title_year)[0]
            year = int(clean_dict[title])
            #removing year from title 
            current = re.sub(r"-\(\d{4}\)$", "", url)
            time.sleep(5)
            s = requests.get(current,headers=headers)
            if s.status_code == 200:
                urls[i] = current
                continue
            #attempting to add country to make url work
            current = re.sub(r'(?=#)', f'-({year}-United-Kingdom)', url)
            time.sleep(robots)
            s = requests.get(current,headers=headers)
            if s.status_code == 200:
                urls[i] = current
                continue
            current = re.sub(r'(?=#)', f'-(UK)-({year})', url)
            time.sleep(robots)
            s = requests.get(current,headers=headers)
            if s.status_code == 200:
                urls[i] = current
                continue
            #setting year back by 1
            current = re.sub(r'(?=#)', f'-({year-1})', url)
            time.sleep(robots)
            s = requests.get(current,headers=headers)
            if s.status_code == 200:
                urls[i] = current
                continue
            current = re.sub(r'(?=#)', f'-({year}-Japan)', url)
            time.sleep(robots)
            s = requests.get(current,headers=headers)
            if s.status_code == 200:
                urls[i] = current
                continue
            current = re.sub(r'(?=#)', f'-(Italy)-({year})', url)
            time.sleep(robots)
            s = requests.get(current,headers=headers)
            if s.status_code == 200:
                urls[i] = current
                continue
            current = re.sub(r'(?=#)', f'-({year}-Australia)', url)
            time.sleep(robots)
            s = requests.get(current,headers=headers)
            if s.status_code == 200:
                urls[i] = current
                continue
            current = re.sub(r'(?=#)', f'-({year}-France)', url)
            time.sleep(robots)
            s = requests.get(current,headers=headers)
            if s.status_code == 200:
                urls[i] = current
                continue
            #if it doesn't we add it to marked
            marked[i] = url
            print(f"Added url number: {i}, which is {url}, to marked at ~{(time.time()-start)//60} minutes")
        #Getting time so I know its working
        print(f"Time of section {k+1}/{len(sections)} completion: ~{(time.time()-start)//60} minutes")
    #pretty printing it so I can read it    
    return urls, marked


def get_final_clean_urls(urls:list[str],marked:dict[int:str]):
    '''Returns fully cleaned urls
        Removes any bad urls, since that means that the-numbers doesn't have them, or they are in a different language in their url.
        I can't account for different languages, and there are few enough that dropping them doesn't hurt.'''
    #removing backwards to ensure everything is deleted properly
    for key in sorted(marked.keys(), reverse=True):
        urls.pop(key)
    return urls


def retrieve_information(urls: list[str], email:str = "cnelson1845@gmail.com",robots:int = 5):
    '''This is not specific to my variables, and can be used with any functioning list of the-numbers urls relating to movies
        This retrieves the movie's title, release date, video release date, inflation adjusted domestic revenue, domestic revenue,
        international revenue, total theater revenue, domestic dvd revenue, domestic bluray revenue, total domestic video revenue,
        opening weekend revenue, production budget, number of theaters it was viewed in, MPAA film rating, runtime, genre(s),
        franchise, and production method, and returns it as a dataframe.
        It returns a dirty dataset in a pandas dataframe and saves that dirty dataframe.
        It includes a timer to monitor progress.
        Also requires email and robots.txt wait time again.'''
    #This block takes 25-45 minutes for our 2014-2024 movies
    #getting relevant information from each url
    title = []
    date = []
    vid_date = []
    inf_adj_dom_rev = []
    dom_rev = []
    inter_rev = []
    total_theater_rev = []
    dom_dvd_rev = []
    dom_bluray_rev = []
    dom_video_rev = []
    opening_weekend = []
    budget = []
    theaters = []
    film_rating = []
    runtime = []
    genre = []
    franchise = []
    production_method = []
    user_agent:str =  f"For class web scraping assignment, email: {email}"
    headers: dict[str:str] = {"User-Agent": user_agent}
    for j,url in enumerate(urls):
        time.sleep(robots)
        r = requests.get(url,headers=headers)
        #Checking if something in my cleaning went wrong
        if r.status_code != 200:
            #this means something is wrong with the website, and we will simply consider it as the-numbers not having it.
            print(f"Because status code is not 200-Removed data for url: {url}, with status: {r.status_code}")
            continue
        soup = BeautifulSoup(r.text)
        #getting title and release year
        header = soup.find("h1")
        #This is here just in case
        if header is None:
            #this means something is wrong with the website, and we will simply consider it as the-numbers not having it.
            print(f"Because header is None-Removed data for url: {url}, with header: {header.text}")
            continue
        if re.match(r'^(.*)\s\((\d{4})\)$', header.text) is None:
            #this means something is wrong with the website, and we will simply consider it as the-numbers not having it.
            print(f"Because header doesn't match-Removed data for url: {url}, with header: {header.text}")
            continue
        title_ = re.match(r'^(.*)\s\((\d{4})\)$', header.text).group(1)
        title.append(title_)
        #getting box office numbers
        label = soup.find("b", string="Domestic Box Office")
        if label is not None:
            value_cell = label.find_parent("td").find_next_sibling("td")
            dbo = value_cell.get_text(strip=True)
            dom_rev.append(dbo)
        else:
            dom_rev.append(None)
        label = soup.find("b", string="International Box Office")
        if label is not None:
            value_cell = label.find_parent("td").find_next_sibling("td")
            ibo = value_cell.get_text(strip=True)
            inter_rev.append(ibo)
        else:
            inter_rev.append(None)
        label = soup.find("b", string="Worldwide Box Office")
        if label is not None:
            value_cell = label.find_parent("td").find_next_sibling("td")
            wwbo = value_cell.get_text(strip=True)
            total_theater_rev.append(wwbo)
        else:
            total_theater_rev.append(None)
        label = soup.find("b", string="Infl. Adj. Dom. BO")
        if label is not None:
            value_cell = label.find_parent("td").find_next_sibling("td")
            iadbo = value_cell.get_text(strip=True)
            inf_adj_dom_rev.append(iadbo)
        else:
            inf_adj_dom_rev.append(None)
        #getting video sales
        label = soup.find("b", string="Est. Domestic DVD Sales")
        if label is not None:
            value_cell = label.find_parent("td").find_next_sibling("td")
            dvd = value_cell.get_text(strip=True)
            dom_dvd_rev.append(dvd)
        else:
            dom_dvd_rev.append(None)
        label = soup.find("b", string="Est. Domestic Blu-ray Sales")
        if label is not None:
            value_cell = label.find_parent("td").find_next_sibling("td")
            bluray = value_cell.get_text(strip=True)
            dom_bluray_rev.append(bluray)
        else:
            dom_bluray_rev.append(None)
        label = soup.find("b", string="Total Est. Domestic Video Sales")
        if label is not None:
            value_cell = label.find_parent("td").find_next_sibling("td")
            dvs = value_cell.get_text(strip=True)
            dom_video_rev.append(dvs)
        else:
            dom_video_rev.append(None)
        #getting opening weekend
        label = soup.find("b", string=lambda t: t and "Opening" in t and "Weekend" in t)
        if label is not None:
            value_cell = label.find_parent("td").find_next_sibling("td")
            ow = value_cell.get_text(strip=True)
            opening_weekend.append(ow)
        else:
            opening_weekend.append(None)
        #getting production budget
        label = soup.find("b", string=lambda t: t and "Production" in t and "Budget" in t)
        if label is not None:
            value_cell = label.find_parent("td").find_next_sibling("td")
            pb = value_cell.get_text(strip=True)
            budget.append(pb)
        else:
            budget.append(None)
        #getting theater counts
        label = soup.find("b", string="Theater counts:")
        if label is not None:
            value_cell = label.find_parent("td").find_next_sibling("td")
            tc = value_cell.get_text(strip=True)
            theaters.append(tc)
        else:
            theaters.append(None)
        #getting dates
        label = soup.find("b", string="Domestic Releases:")
        if label is not None:
            value_cell = label.find_parent("td").find_next_sibling("td")
            dom_rel = value_cell.get_text(strip=True)
            date.append(dom_rel)
        else:
            date.append(None)
        label = soup.find("b", string=lambda t: t and "Video" in t and "Release" in t)
        if label is not None:
            value_cell = label.find_parent("td").find_next_sibling("td")
            v_rel = value_cell.get_text(strip=True)
            vid_date.append(v_rel)
        else:
            vid_date.append(None)
        #getting other features of the movie
        label = soup.find("b", string=lambda t: t and "MPAA" in t and "Rating" in t)
        if label is not None:
            value_cell = label.find_parent("td").find_next_sibling("td")
            rating = value_cell.get_text(strip=True)
            film_rating.append(rating)
        else:
            film_rating.append(None)
        label = soup.find("b", string="Running Time:")
        if label is not None:
            value_cell = label.find_parent("td").find_next_sibling("td")
            r_time = value_cell.get_text(strip=True)
            runtime.append(r_time)
        else:
            runtime.append(None)
        label = soup.find("b", string="Franchise:")
        if label is not None:
            value_cell = label.find_parent("td").find_next_sibling("td")
            f = value_cell.get_text(strip=True)
            franchise.append(f)
        else:
            franchise.append(None)
        label = soup.find("b", string="Genre:")
        if label is not None:
            value_cell = label.find_parent("td").find_next_sibling("td")
            g = value_cell.get_text(strip=True)
            genre.append(g)
        else:
            genre.append(None)
        label = soup.find("b", string=lambda t: t and "Method" in t and "Production" in t)
        if label is not None:
            value_cell = label.find_parent("td").find_next_sibling("td")
            pm = value_cell.get_text(strip=True)
            production_method.append(pm)
        else:
            production_method.append(None)
        #Just a progress bar for my sake
        print(f"Finished url {j+1}/{len(urls)}, or {round((j + 1) / len(urls) * 100, 2)}%")
        #putting the information into a dataframe
    df: pd.DataFrame = pd.DataFrame({"Title":title,"Release Date":date,"Inflation Adjusted Domestic Revenue":inf_adj_dom_rev,"Domestic Revenue":dom_rev,"International Revenue":inter_rev,
                        "Total Box Office Revenue":total_theater_rev,"Domestic Video Revenue":dom_video_rev,"Opening Weekend":opening_weekend,"Production Budget":budget,
                        "Theater Number":theaters,"MPAA Rating":film_rating,"Runtime":runtime,"Genre":genre,"Franchise":franchise,"Production Method":production_method,
                        "Video Release Date":vid_date,"Domestic DVD Revenue":dom_dvd_rev,"Domestic Bluray Revenue":dom_bluray_rev})
    df.to_csv("dirty_data.csv")
    return df
    

def retrieve_dirty_dataset_specific():
    '''Creates and return our specific dirty dataset dataframe, which we cleaned and used for analysis.'''
    r1,r2 = setup_2(email="cnelsosn1845@gmail.com")
    print("retrieved response objects")
    the_movies = get_website_info(r1,r2)
    print("retrieved website info")
    dirty_urls,clean_dict = get_dirty_urls_for_the_numbers(the_movies)
    print("got dirty urls")
    urls,marked = get_clean_and_bad_urls_for_the_numbers(dirty_urls,clean_dict,email="cnelson1845@gmail.com")
    clean_urls = get_final_clean_urls(urls,marked)
    print("cleaned urls")
    dirty_df = retrieve_information(clean_urls,email="cnelson1845@gmail.com")
    print("got info from websites")
    return dirty_df


#Data Cleaning
def clean_dataframe(df:pd.DataFrame):
    '''Takes a dirty dataframe with data from the-numbers and makes it clean.'''
    #changing all None values to np.nan
    df = df.map(lambda x: np.nan if x is None else x)
    df = df.map(lambda x: np.nan if x == 'n/a' else x)
    #removing extra parentheses (and other extra information)
    cols = ["Opening Weekend","Production Budget","Release Date"]
    df[cols] = df[cols].apply(lambda s: s.str.replace(r"\(.*?\)", "", regex=True).str.strip())
    cols = ["Release Date","Video Release Date"]
    df[cols] = df[cols].apply(lambda s: s.str.replace(r"by.*$", "", regex=True, case=False).str.strip())
    df["MPAA Rating"] = df["MPAA Rating"].str.extract(r"^(G|PG|PG-13|R|NC-17|M|GP|X|Not Rated|Unrated|NR)", expand=False).str.strip()
    df["Runtime"] = df["Runtime"].str.replace(" minutes", "", regex=True).str.strip()
    df["Theater Number"] = df["Theater Number"].str.replace(r"^[^/]*/", "", regex=True)
    df["Theater Number"] = df["Theater Number"].str.replace(r"max.*$", "", regex=True).str.strip()
    #removing dollar signs
    cols = ["Inflation Adjusted Domestic Revenue","Domestic Revenue","International Revenue","Total Box Office Revenue","Domestic Video Revenue","Production Budget","Domestic DVD Revenue","Domestic Bluray Revenue","Opening Weekend"]
    df[cols] = df[cols].apply(lambda s: s.str.replace(r"\$", "", regex=True).str.strip())
    #removing commas
    cols = ["Inflation Adjusted Domestic Revenue","Domestic Revenue","International Revenue","Total Box Office Revenue","Domestic Video Revenue","Production Budget","Domestic DVD Revenue","Domestic Bluray Revenue","Opening Weekend","Theater Number"]
    df[cols] = df[cols].apply(lambda s: s.str.replace(r",", "", regex=True).str.strip())
    #converting release date to datetime object
    cols = ["Release Date", "Video Release Date"]
    df[cols] = df[cols].apply(pd.to_datetime, errors="coerce")
    #converting everything to its proper object type
    cols = ["MPAA Rating","Genre","Franchise","Production Method"]
    df[cols] = df[cols].astype(str)
    cols = ["Inflation Adjusted Domestic Revenue","Domestic Revenue","International Revenue","Total Box Office Revenue","Opening Weekend","Production Budget","Theater Number","Runtime","Domestic DVD Revenue","Domestic Bluray Revenue", "Domestic Video Revenue"]
    df[cols] = df[cols].astype("Int64")
    #also making sure turning everything to strings didn't mess up the np.nans
    df = df.map(lambda x: np.nan if x == 'nan' else x)
    return df


def add_columns_to_df(df:pd.DataFrame):
    '''Adds additional month, year, season, and profits column for additional analysis later.
        Returns clean and full dataframe and saves it.'''
    #Adding additional columns for analysis
    #making month and year their own column
    df["Month"] = (df["Release Date"]).dt.month
    df["Year"] = (df["Release Date"]).dt.year
    #adding season column
    def get_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        elif month in [9, 10, 11]:
            return "Fall"
        else: 
            return np.nan
    df["Season"] = df["Month"].apply(get_season)
    #adding profit column (Total Box Office Revenue - production budget)
    df["Profit"] = df["Total Box Office Revenue"] - df["Production Budget"]
    df.to_csv("movie_data.csv", index=False)
    return df


def retrieve_clean_dataset_specific():
    '''Retrieves our specific dataset in a cleaned dataframe
        You must use output from `retrieve_dirty_dataset_specific` for this to work.'''
    dirty_df = pd.read_csv("dirty_data.csv")
    clean_df = clean_dataframe(dirty_df)
    full_df = add_columns_to_df(clean_df)
    return full_df


#Data Analysis
def earnings_correlation():
    '''Shows correlation between different earnings statistics'''
    df = pd.read_csv("movie_data.csv")
    print(df.corr(numeric_only=True))


def graph_revenue():
    '''Shows revenue of movies'''
    df = pd.read_csv("movie_data.csv")
    # graph of the revenues
    mean_of = pd.DataFrame()
    mean_of = df.groupby('Release Date')[['Inflation Adjusted Domestic Revenue','Domestic Revenue', 'Total Box Office Revenue','International Revenue','Domestic Video Revenue']].mean()
    # 'Total Box Office Revenue','International Revenue',
    mean_of[['Inflation Adjusted Domestic Revenue','Domestic Revenue','Domestic Video Revenue']].plot()
    plt.xticks(rotation=45)
    mean_of[['Total Box Office Revenue','International Revenue']].plot()
    plt.xticks(rotation=45)
    # the graphs show the correlations and the difference in there earnings
    plt.show()


def describe_revenue():
    '''Shows summary statistics for three big revenue types'''
    df = pd.read_csv("movie_data.csv")
    df['Inflation Adjusted Domestic Revenue'].describe()
    df['Domestic Revenue'].describe()
    df['International Revenue'].describe()


def season_earnings():
    '''Returns revenue by season'''
    df = pd.read_csv("movie_data.csv")
    df.boxplot(column='Inflation Adjusted Domestic Revenue', by='Season')
    plt.title('Revenue by Season of Movie Release')
    plt.show()


def genre_earnings():
    '''Returns revenue by genre'''
    df = pd.read_csv("movie_data.csv")
    df.boxplot(column='Inflation Adjusted Domestic Revenue', by='Genre')
    plt.xticks(rotation=45)
    plt.title('Revenue by Genre')
    plt.show()


def production_method_earnings():
    '''Returns revenue for different production methods'''
    df = pd.read_csv("movie_data.csv")
    df.boxplot(column='Inflation Adjusted Domestic Revenue', by='Production Method')
    plt.xticks(rotation=45)
    plt.title('Revenue by production method')
    plt.show()


def ratings_earnings():
    '''Returns revenue by rating'''
    df = pd.read_csv("movie_data.csv")
    df.boxplot(column='Inflation Adjusted Domestic Revenue', by='MPAA Rating')
    plt.xticks(rotation=45)
    plt.show()


def findings():
    '''prints the findings from our analysis'''
    print("product budget is most strongly correlated with international earnings, this implies that the high earning movies had the most funding (on average)")
    print("Summer is the time for big box office hits, and fall is the time for movies that don't do well")
    print("Adventure has the largest range, and seems to have the best chance to do well, but action is close behind")
    print("Digital animation seems to usually do the best, I presume this is because it appeals to a wide range of people")
    print("Overall, it seems that making movies that appeal to everyone, and releasing them in the summer is the best chance for making a high earning movie")

def do_analysis_all():
    earnings_correlation()
    describe_revenue()
    season_earnings()
    genre_earnings()
    production_method_earnings()
    ratings_earnings()


#ML Analysis
def format_data():
    '''Fixing NaN values by imputing most numeric values with median, due to skewness, except for month, which is imputed with mode. Any string columns are removed'''
    df_ml = pd.read_csv("movie_data.csv")
    #dropping rows without target
    df_ml = df_ml.dropna(subset=["Total Box Office Revenue"])
    #dropping text columns we can't use
    df_ml = df_ml.drop(columns=[
        "Title", "Season", "MPAA Rating", "Genre", "Franchise", "Production Method"
    ])
    #dropping the time-data columns since we have month and year of film release separated
    df_ml = df_ml.drop(columns=[
        "Release Date", "Video Release Date"
    ])
    #filling in with median
    num_cols = [
        "Inflation Adjusted Domestic Revenue", "Domestic Revenue",
        "International Revenue", "Domestic Video Revenue",
        "Opening Weekend", "Production Budget", "Theater Number",
        "Runtime", "Domestic DVD Revenue", "Domestic Bluray Revenue",
        "Year", "Profit"
    ]
    for col in num_cols:
        df_ml[col] = df_ml[col].fillna(df_ml[col].median())
    #filling in month with mode
    df_ml["Month"] = df_ml["Month"].fillna(df_ml["Month"].mode()[0])
    #saving df
    df_ml.to_csv("ml_movie_data.csv", index=False)
    return df_ml


def create_y(df_ml):
    '''getting y for X and y'''
    # Log-transform the target due to skewness of target
    df_ml["log_revenue"] = np.log1p(df_ml["Total Box Office Revenue"])
    y = df_ml["log_revenue"]
    return y


def create_X_y_after():
    '''This is used after the movie has been released for awhile'''
    df_ml = pd.read_csv("ml_movie_data.csv")
    y = create_y(df_ml)
    X = df_ml[["Inflation Adjusted Domestic Revenue","Domestic Revenue","Domestic Video Revenue","Opening Weekend","Production Budget","Theater Number","Runtime","Domestic DVD Revenue","Domestic Bluray Revenue","Month","Year","Profit"]]
    return X,y


def create_X_y_opening():
    '''This is used after the movie has had its opening weekend'''
    df_ml = pd.read_csv("ml_movie_data.csv")
    y = create_y(df_ml)
    X = df_ml[["Opening Weekend","Production Budget","Theater Number","Runtime","Month","Year"]]
    return X,y


def create_X_y_before():
    '''This is used before the movie comes out and we only know the budget, year and month it will be out'''
    df_ml = pd.read_csv("ml_movie_data.csv")
    y = create_y(df_ml)
    X = df_ml[["Production Budget","Runtime","Month"]]
    return X,y


def find_most_important_features_plot(X,y):
    '''Finds most important labels, from X, for predicting a target, y, shown with shap'''
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X)


def find_most_important_features_numbers(X,y):
    '''Finds most important labels, from X, for predicting a target, y, shown numbers from pandas sklearn'''
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    print("From pandas:")
    rf_importances = pd.Series(model.feature_importances_, index=X.columns)
    print(rf_importances.sort_values(ascending=False))
    print()
    print("From sklearn")
    perm = permutation_importance(model, X, y, n_repeats=20, random_state=42)
    perm_importances = pd.Series(perm.importances_mean, index=X.columns)
    print(perm_importances.sort_values(ascending=False))


def do_ml_analysis_plots():
    format_data()
    print("Most important features before a movie opens")
    X,y=create_X_y_before()
    find_most_important_features_plot(X,y)
    print("Most important features right after a movie opens")
    X,y=create_X_y_opening()
    find_most_important_features_plot(X,y)
    print("Most important features long after a movie opened")
    X,y=create_X_y_after()
    find_most_important_features_plot(X,y)


def do_ml_analysis_numbers():
    format_data()
    print("Most important features before a movie opens")
    X,y=create_X_y_before()
    find_most_important_features_numbers(X,y)
    print("\n")
    print("Most important features right after a movie opens")
    X,y=create_X_y_opening()
    find_most_important_features_numbers(X,y)
    print("\n")
    print("Most important features long after a movie opened")
    X,y=create_X_y_after()
    find_most_important_features_numbers(X,y)


def ml_analysis_findings(): 
    print("Based upon the data, it seems that before a movie opens, the budget is the best predictor of revenue, bigger budget likely means bigger revenue")
    print("After a movie has had its opening weekend, those opening weekend numbers are the best for determining overall revenue")
    print("AFter a movie has released, then using its US domestic box office revenue is a good way to predict overall revenue, this tells us domestic contributes far more than internal toe revenue.")


#Conclusion
def printing_full_dataset():
    from IPython.display import display

    df = pd.read_csv("movie_data.csv")
    pd.set_option('display.max_columns', None)
    display(df.head())


def data_creation():
    dirty_df = retrieve_dirty_dataset_specific()
    print("dirty dataset created and saved")
    print(dirty_df.head())
    df = retrieve_clean_dataset_specific()
    print("clean dataset created and saved")
    print(df.head())
    return df


def do_analysis_specific():
    '''Does our analysis for our specific data to answer our research question using all analysis'''
    print("Our research question is: What features of a movie can best be used to predict its revenue?")
    print("The following is our analysis")
    do_analysis_all()
    print("Our general findings were:\n")
    findings()
    print("The following is our machine learning analysis")
    do_ml_analysis_numbers()
    do_ml_analysis_plots()
    print("Our general findings were:\n")
    ml_analysis_findings()
    print("The answer we found was: ")
    print("This is because: ...")
    pass


def totality():
    data_creation()
    do_analysis_specific()



if __name__ == "__main__":
    #printing_full_dataset()
    #totality()
    #do_analysis_all()
    do_analysis_specific()
    #do_ml_analysis_plots()
    #do_ml_analysis_numbers()

# package installation note: first you must 'uv add' all dependencies into your environment, then you can download it.
# uv pip install -i https://test.pypi.org/simple/ final-movie-analysis==0.1.1
