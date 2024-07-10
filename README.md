## Documentation group 9 (Pointers)

### Python environment and requirements.txt

We worked on Google Colab all the time, but it is reproducible on environments with python 3.10.14 and these modules:

```python
transformers==4.41.2
tensorflow==2.15.0
torch==2.3.0
scikit-learn==1.2.2
polars==0.20.8
jupyter==1.0.0
pyarrow==14.0.2
pandas==2.0.3
```

### run.sh

Contains code that should run notebooks in *examples/model_runners*. However, if it doesn't run correctly, we suggest you to setup environment with python version 3.10.14 and previously noted modules and then iterate manually over each notebook, it should work that way. Then, you can also run make_beyond_accuracy_model_name.ipynb notebooks in *examples/beyond_acc*. 

- Beyond accuracy notebooks, as others notebooks from *examples/model_runners* will generate intermediate as well as some final results that should be same or very similar to the ones in report that are inside notebooks. We didn't store results in separate .csv or similar types of files. 

### src/downloads/ 

Contain zipped .txt files of top 3 predicted scores (of inview articles) for each impression (they are basically lists of top 3 inview article ids). These zip archives are later used in beyond accuracy notebooks to calculate beyond accuracy metrics.  Predictions are made in *gru_danish_bert.ipynb*, *npa_danish_bert.ipynb* and *nrms_danish_bert.ipynb*. Therefore, if you want to generate these files of lists that have top 3 predicted scores (even if we have provided them), then you need to attach the following code at the end of each earlier noted notebooks (models with danish transformer): 

```python
df_validation = df_validation.with_columns(
    pl.struct(['article_ids_inview', 'ranked_scores']).apply(
        lambda row: [x for _, x in sorted(zip(row['ranked_scores'], row['article_ids_inview']))[:3]]
    ).alias('top_3_articles')
)

write_submission_file(
    impression_ids=df_validation[DEFAULT_IMPRESSION_ID_COL],
    prediction_scores=df_validation["top_3_articles"],
    path="downloads/ba_nrms_danish.txt",  ## here you change the name of model to match with corresponding open notebook
)
```



### Setting up data folder:

Structure should look like this (as stated in report we only used ebnerd_small dataset but we also need embeddings for beyond accuracy metrics and ebnerd_large articles.parquet):

    data/
        ebnerd_demo/
        ebnerd_small/
            train/
                behaviors.parquet
                history.parquet
            validation/
                behaviors.parquet
                history.parquet
            articles.parquet
        ebnerd_large/
            articles.parquet
        embeddings/
            Ekstra_Bladet_contrastive_vector/contrastive_vector.parquet
            Ekstra_Bladet_word2vec/document_vector.parquet

### Prediction folder
Here we have all prediction .txt files of models that we trained (they are all generated from validation dataset and therefore not uploaded to codabench)

### Models

It was taking too long to save the models with checkpoints, so we didn't include any model weights in our JupyterLab or in GitLab repository. It was much faster to run the notebook all over again and evaluate models in runtime. 

### GPU

All code needs to be run on gpus 

 