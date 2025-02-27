# Spaceflights tutorial FAQs

```{note}
If you can't find the answer you need here, [ask the Kedro community for help](spaceflights_tutorial.md#get-help)!
```

## How do I resolve these common errors?

### DataSet errors
#### DataSetError: Failed while loading data from data set
You're [testing whether Kedro can load the raw test data](./set_up_data.md#test-that-kedro-can-load-the-csv-data) and see the following:

```python
DataSetError: Failed while loading data from data set
CSVDataSet(filepath=...).
[Errno 2] No such file or directory: '.../companies.csv'
```

or a similar error for the `shuttles` or `reviews` data.

Have you downloaded [the three sample data files](./set_up_data.md#download-datasets) and stored them in the `data/raw` folder?

#### DataSetNotFoundError: DataSet not found in the catalog

Do you see the following for one of the raw datasets, make sure you have saved `catalog.yml`?

```python
DataSetNotFoundError: DataSet 'companies' not found in the catalog
```

Call `exit()` within the IPython session and restart `kedro ipython` (or type `@kedro_reload` into the IPython console to reload Kedro into the session without restarting). Then try again.


#### DataSetError: An exception occurred when parsing config for DataSet

Are you seeing a message saying that an exception occurred?

```bash
DataSetError: An exception occurred when parsing config for DataSet
'data_processing.preprocessed_companies':
Object 'ParquetDataSet' cannot be loaded from 'kedro.extras.datasets.pandas'. Please see the
documentation on how to install relevant dependencies for kedro.extras.datasets.pandas.ParquetDataSet:
https://kedro.readthedocs.io/en/stable/kedro_project_setup/dependencies.html
```

The Kedro Data Catalog is missing [dependencies needed to parse the data](../kedro_project_setup/dependencies.md#install-dependencies-related-to-the-data-catalog). Check that you've added [all the project dependencies to `requirements.txt`](./tutorial_template.md#project-dependencies) and then call `pip install -r src/requirements.txt` to install them.

### Pipeline run

To successfully run the pipeline, all required input datasets must already exist, otherwise you may get an error similar to this:


```bash
kedro run --pipeline=data_science

2019-10-04 12:36:12,135 - root - INFO - ** Kedro project kedro-tutorial
2019-10-04 12:36:12,158 - kedro.io.data_catalog - INFO - Loading data from `model_input_table` (CSVDataSet)...
2019-10-04 12:36:12,158 - kedro.runner.sequential_runner - WARNING - There are 3 nodes that have not run.
You can resume the pipeline run with the following command:
kedro run
Traceback (most recent call last):
  ...
  File "pandas/_libs/parsers.pyx", line 382, in pandas._libs.parsers.TextReader.__cinit__
  File "pandas/_libs/parsers.pyx", line 689, in pandas._libs.parsers.TextReader._setup_parser_source
FileNotFoundError: [Errno 2] File b'data/03_primary/model_input_table.csv' does not exist: b'data/03_primary/model_input_table.csv'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  ...
    raise DataSetError(message) from exc
kedro.io.core.DataSetError: Failed while loading data from data set CSVDataSet(filepath=data/03_primary/model_input_table.csv, save_args={'index': False}).
[Errno 2] File b'data/03_primary/model_input_table.csv' does not exist: b'data/03_primary/model_input_table.csv'
```
