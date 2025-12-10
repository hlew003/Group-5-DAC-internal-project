# HDB Chatbot Artifacts

The trained model file is ~6.5GB, so it is not committed. Regenerate artifacts locally before running `streamlit_app.py`.

## Steps to regenerate
1) Use the project venv and install deps:
```
source hdb_chatbot/.venv/bin/activate
pip install -r hdb_chatbot/requirements.txt
```

2) In the `Group_5_HDB_Resale.ipynb` notebook (using the same venv kernel), run the cells that:
   - Create `hdb_chatbot/geocode/distance_lookup.parquet` from `df_4` (the geocoded dataframe).
   - Train the RandomForest pipeline on `df_ML` and save `hdb_chatbot/geocode/rf_hdb_pipeline.pkl`.

3) Verify the files exist:
```
ls -lh hdb_chatbot/geocode/distance_lookup.parquet hdb_chatbot/geocode/rf_hdb_pipeline.pkl
```

4) Run the app:
```
streamlit run hdb_chatbot/streamlit_app.py
```

## Notes
- Make sure `df_ML` has `trans_year` (e.g., from `month`), and `df_4` contains the geocoded columns used in the lookup (`town`, `latitude`, `longitude`, and the distance columns).
- Keep your `.env` (API keys) out of git.
