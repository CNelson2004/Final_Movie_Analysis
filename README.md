# final-movie-analysis
A little package for analyzing universal studios movies from 2014-2024

## Description
This package demonstrates:
- How to scrape information from a webpage for a dirty dataset (`retrieve_dirty_dataset_specific`)
- How to make a dirty dataset clean (`retrieve_clean_dataset_specific`)
- How to analyze a movie dataset (`do_analysis_specific`)

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
- Python >= 3.12
- bs4>=0.0.2
- ipykernel >= 7.1.0
- lxml >= 6.0.2
- matplotlib >= 3.10.7
- numpy >= 2.3.4
- pandas >= 2.3.3
- requests >= 2.32.5
- seaborn >= 0.13.2
- pingouin>=0.5.5
- scikit-learn>=1.7.2
- scipy>=1.16.3
- shap>=0.49.1

## License
MIT
