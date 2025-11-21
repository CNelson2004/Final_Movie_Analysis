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
from final-movie-analysis import retrieve_dirty_dataset_specific, retrieve_clean_dataset_specific, do_analysis_specific

#obtain dirty dataset
dirty_df = retrieve_dirty_dataset_specific()
print("dirty dataset created and saved")
#clean the dirty dataset
df = retrieve_clean_dataset_specific(dirty_df)
print("clean dataset created and saved")
#analyze the clean dataset
do_analysis_specific(df)
```

## Dependencies
- Python >= 3.12
- ipykernel >= 7.1.0"
- lxml >= 6.0.2"
- matplotlib >= 3.10.7"
- numpy >= 2.3.4"
- pandas >= 2.3.3"
- requests >= 2.32.5"
- seaborn >= 0.13.2"

## License
MIT
