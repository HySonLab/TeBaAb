This folder is used to create the `sabdab_info.csv` file, which is a processed dataset derived from the SAbDAb database. 

First, download the SAbDAb database:

```bash
wget https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/archive/all/
```

Extract it to `/sabdab_dataset/all_structures/imgt`

Run `simplify_sabdab_summary.py` to process the SAbDAb data and generate `sabdab_info.csv`. This CSV file will be used by downstream scripts.