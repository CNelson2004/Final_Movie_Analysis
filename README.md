# final-movie-analysis
A little package for analyzing universal studios movies from 2014-2024

## Description
This package demonstrates:
- How to scrape movie information from the web for a dirty dataset and clean it (`data_creation`)
- View the dataset (`print_dataset_head`)
- How to analyze a movie dataset (`do_analysis_specific`)
- How to check missingness in the dataset (`get_NAs`)
- How do it all in one step (`totality`)
- How to analyze feature importance (`do_ml_analysis_plots`, `do_ml_analysis_numbers`, `ml_analysis_findings`)
- How to analyze feature significance (`season_significance`, `rating_significance`, `genre_significance`, `production_method_significance`)
- How to analyze factor relation (`factors_analysis`, `earnings_correlation`, `season_earnings`, `genre_earnings`, `production_method_earnings`, `ratings_earnings`)
- How to analyze a feature over time (`graph_revenue`, `graph_revenue_by_year`, `graph_revenue_and_profit`, `analyze_revenue`)
- How to check one feature specifically (`describe_revenue`)
- How to interpret analysis (`factors_findings`, `revenue_findings`)

## Installation
```bash
pip install final-movie-analysis
```

## Usage
```py
import sys, os
sys.path.append(os.path.abspath("src"))
from final_movie_analysis.functions import totality
#Create and analyze the data
totality()
```

## Dependencies
- requests>=2.32.5
- bs4>=0.0.2
- numpy>=2.3.4
- pandas>=2.3.3
- matplotlib>=3.10.7
- lxml>=6.0.2
- scipy>=1.16.3
- scikit-learn>=1.7.2
- shap>=0.49.1
- pingouin>=0.5.5

## License
MIT
