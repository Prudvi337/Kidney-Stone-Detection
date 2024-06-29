# Kidney-Stone-Detection
To detect stones in kidneys using Machine learning Models
<a id="content"></a>
## Notebook content

This notebook contains a Scikit-learn representation of AutoAI pipeline. This notebook introduces commands for retrieving data, training the model, and testing the model. 

Some familiarity with Python is helpful. This notebook uses Python 3.10 and scikit-learn 1.1.1.
## Notebook goals

-  Scikit-learn pipeline definition
-  Pipeline training 
-  Pipeline evaluation

## Contents

This notebook contains the following parts:

**[Setup](#setup)**<br>
&nbsp;&nbsp;[Package installation](#install)<br>
&nbsp;&nbsp;[AutoAI experiment metadata](#variables_definition)<br>
&nbsp;&nbsp;[Watson Machine Learning connection](#connection)<br>
**[Pipeline inspection](#inspection)** <br>
&nbsp;&nbsp;[Read training data](#read)<br>
&nbsp;&nbsp;[Train and test data split](#split)<br>
&nbsp;&nbsp;[Create pipeline](#preview_model_to_python_code)<br>
&nbsp;&nbsp;[Train pipeline model](#train)<br>
&nbsp;&nbsp;[Test pipeline model](#test_model)<br>
**[Store the model](#saving)**<br>
**[Summary and next steps](#summary_and_next_steps)**<br>
**[Copyrights](#copyrights)**
<a id="setup"></a>
# Setup
<a id="install"></a>
## Package installation
Before you use the sample code in this notebook, install the following packages:
 - ibm-watsonx-ai,
 - autoai-libs,
 - scikit-learn

!pip install ibm-watsonx-ai | tail -n 1
!pip install autoai-libs==1.17.2 | tail -n 1
!pip install scikit-learn==1.1.1 | tail -n 1
!pip install -U 'lale>=0.7,<0.8' | tail -n 1
Filter warnings for this notebook.
import warnings

warnings.filterwarnings('ignore')
<a id="variables_definition"></a>
## AutoAI experiment metadata
The following cell contains the training data connection details.  
**Note**: The connection might contain authorization credentials, so be careful when sharing the notebook.
from ibm_watsonx_ai.helpers import DataConnection
from ibm_watsonx_ai.helpers import ContainerLocation

training_data_references = [
    DataConnection(
        data_asset_id='f8f05505-f2b2-4c89-b5b3-aefaac72d02c'
    ),
]
training_result_reference = DataConnection(
    location=ContainerLocation(
        path='auto_ml/1cae7a7d-fdf9-4e48-838d-9ffad7b43e07/wml_data/31dceb21-616c-40e8-837d-3574490917f2/data/automl',
        model_location='auto_ml/1cae7a7d-fdf9-4e48-838d-9ffad7b43e07/wml_data/31dceb21-616c-40e8-837d-3574490917f2/data/automl/model.zip',
        training_status='auto_ml/1cae7a7d-fdf9-4e48-838d-9ffad7b43e07/wml_data/31dceb21-616c-40e8-837d-3574490917f2/training-status.json'
    )
)
The following cell contains input parameters provided to run the AutoAI experiment in Watson Studio.
experiment_metadata = dict(
    prediction_type='binary',
    prediction_column='target',
    holdout_size=0.1,
    scoring='accuracy',
    csv_separator=',',
    random_state=33,
    max_number_of_estimators=2,
    training_data_references=training_data_references,
    training_result_reference=training_result_reference,
    deployment_url='https://us-south.ml.cloud.ibm.com',
    project_id='786b6989-9431-4e2a-8aa9-cd31fc9c04ef',
    positive_label=1,
    drop_duplicates=True,
    include_batched_ensemble_estimators=[],
    feature_selector_mode='auto'
)
## Set `n_jobs` parameter to the number of available CPUs
import os, ast
CPU_NUMBER = 1
if 'RUNTIME_HARDWARE_SPEC' in os.environ:
    CPU_NUMBER = int(ast.literal_eval(os.environ['RUNTIME_HARDWARE_SPEC'])['num_cpu'])
<a id="connection"></a>
## Watson Machine Learning connection

This cell defines the credentials required to work with the Watson Machine Learning service.

**Action**: Provide the IBM Cloud apikey, For details, see [documentation](https://cloud.ibm.com/docs/account?topic=account-userapikey).
api_key = 'PUT_YOUR_APIKEY_HERE'
wml_credentials = {
    "apikey": api_key,
    "url": experiment_metadata['deployment_url']
}
from ibm_watsonx_ai import APIClient

wml_client = APIClient(wml_credentials)

if 'space_id' in experiment_metadata:
    wml_client.set.default_space(experiment_metadata['space_id'])
else:
    wml_client.set.default_project(experiment_metadata['project_id'])
    
training_data_references[0].set_client(wml_client)
<a id="inspection"></a>
# Pipeline inspection
<a id="read"></a>
## Read training data

Retrieve training dataset from AutoAI experiment as pandas DataFrame.

**Note**: If reading data results in an error, provide data as Pandas DataFrame object, for example, reading .CSV file with `pandas.read_csv()`. 

It may be necessary to use methods for initial data pre-processing like: e.g. `DataFrame.dropna()`, `DataFrame.drop_duplicates()`, `DataFrame.sample()`.

X_train, X_test, y_train, y_test = training_data_references[0].read(experiment_metadata=experiment_metadata, with_holdout_split=True, use_flight=True)
<a id="preview_model_to_python_code"></a>
## Create pipeline
In the next cell, you can find the Scikit-learn definition of the selected AutoAI pipeline.
#### Import statements.
from autoai_libs.transformers.exportable import NumpyColumnSelector
from autoai_libs.transformers.exportable import FloatStr2Float
from autoai_libs.transformers.exportable import NumpyReplaceMissingValues
from autoai_libs.transformers.exportable import NumImputer
from autoai_libs.transformers.exportable import OptStandardScaler
from autoai_libs.transformers.exportable import float32_transform
from autoai_libs.cognito.transforms.transform_utils import TA2
import numpy as np
import autoai_libs.utils.fc_methods
from autoai_libs.cognito.transforms.transform_utils import FS1
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
#### Pre-processing & Estimator.
numpy_column_selector = NumpyColumnSelector(columns=[0, 1, 2, 3, 4, 5])
float_str2_float = FloatStr2Float(
    dtypes_list=[
        "float_num", "float_num", "float_int_num", "float_num",
        "float_int_num", "float_num",
    ],
    missing_values_reference_list=[],
)
numpy_replace_missing_values = NumpyReplaceMissingValues(
    filling_values=float("nan"), missing_values=[]
)
num_imputer = NumImputer(missing_values=float("nan"), strategy="median")
opt_standard_scaler = OptStandardScaler(use_scaler_flag=False)
ta2 = TA2(
    fun=np.add,
    name="sum",
    datatypes1=[
        "intc", "intp", "int_", "uint8", "uint16", "uint32", "uint64", "int8",
        "int16", "int32", "int64", "short", "long", "longlong", "float16",
        "float32", "float64",
    ],
    feat_constraints1=[autoai_libs.utils.fc_methods.is_not_categorical],
    datatypes2=[
        "intc", "intp", "int_", "uint8", "uint16", "uint32", "uint64", "int8",
        "int16", "int32", "int64", "short", "long", "longlong", "float16",
        "float32", "float64",
    ],
    feat_constraints2=[autoai_libs.utils.fc_methods.is_not_categorical],
    col_names=[
        "gravity", "ph", "osmolality", "conductivity", "urea", "calcium",
    ],
    col_dtypes=[
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
        np.dtype("float32"), np.dtype("float32"), np.dtype("float32"),
    ],
)
fs1 = FS1(
    cols_ids_must_keep=range(0, 6),
    additional_col_count_to_keep=8,
    ptype="classification",
)
extra_trees_classifier = ExtraTreesClassifier(n_jobs=CPU_NUMBER, random_state=33)

#### Pipeline.
pipeline = make_pipeline(
    numpy_column_selector,
    float_str2_float,
    numpy_replace_missing_values,
    num_imputer,
    opt_standard_scaler,
    float32_transform(),
    ta2,
    fs1,
    extra_trees_classifier,
)
<a id="train"></a>
## Train pipeline model

### Define scorer from the optimization metric
This cell constructs the cell scorer based on the experiment metadata.
from sklearn.metrics import get_scorer

scorer = get_scorer(experiment_metadata['scoring'])
<a id="test_model"></a>
### Fit pipeline model
In this cell, the pipeline is fitted.
pipeline.fit(X_train.values, y_train.values.ravel());
<a id="test_model"></a>
## Test pipeline model
Score the fitted pipeline with the generated scorer using the holdout dataset.
score = scorer(pipeline, X_test.values, y_test.values)
print(score)
pipeline.predict(X_test.values[:5])
<a id="saving"></a>
## Store the model

In this section you will learn how to store the trained model.
model_metadata = {
    wml_client.repository.ModelMetaNames.NAME: 'P4 - Pretrained AutoAI pipeline'
}

stored_model_details = wml_client.repository.store_model(model=pipeline, meta_props=model_metadata, experiment_metadata=experiment_metadata)
Inspect the stored model details.
stored_model_details
<a id="deployment"></a>
## Create online deployment
You can use commands bellow to promote the model to space and create online deployment (web service).

<a id="working_spaces"></a>
### Working with spaces

In this section you will specify a deployment space for organizing the assets for deploying and scoring the model. If you do not have an existing space, you can use [Deployment Spaces Dashboard](https://dataplatform.cloud.ibm.com/ml-runtime/spaces?context=cpdaas) to create a new space, following these steps:

- Click **New Deployment Space**.
- Create an empty space.
- Select Cloud Object Storage.
- Select Watson Machine Learning instance and press **Create**.
- Copy `space_id` and paste it below.

**Tip**: You can also use the API to prepare the space for your work. Learn more [here](https://github.com/IBM/watson-machine-learning-samples/blob/master/notebooks/python_sdk/instance-management/Space%20management.ipynb).

**Action**: Assign or update space ID below.
space_id = "PUT_YOUR_SPACE_ID_HERE"

model_id = wml_client.spaces.promote(asset_id=stored_model_details["metadata"]["id"], source_project_id=experiment_metadata["project_id"], target_space_id=space_id)
#### Prepare online deployment
wml_client.set.default_space(space_id)

deploy_meta = {
        wml_client.deployments.ConfigurationMetaNames.NAME: "Incrementally trained AutoAI pipeline",
        wml_client.deployments.ConfigurationMetaNames.ONLINE: {},
    }

deployment_details = wml_client.deployments.create(artifact_uid=model_id, meta_props=deploy_meta)
deployment_id = wml_client.deployments.get_id(deployment_details)
#### Test online deployment
import pandas as pd

scoring_payload = {
    "input_data": [{
        'values': pd.DataFrame(X_test[:5])
    }]
}

wml_client.deployments.score(deployment_id, scoring_payload)
<a id="cleanup"></a>
### Deleting deployment
You can delete the existing deployment by calling the `wml_client.deployments.delete(deployment_id)` command.
To list the existing web services, use `wml_client.deployments.list()`.
