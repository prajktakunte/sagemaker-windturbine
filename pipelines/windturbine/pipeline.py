"""Example workflow pipeline script for windturbine pipeline.
                                                                                 . -ModelStep
                                                                                .
    Process-> DataQualityCheck/DataBiasCheck -> Train -> Evaluate -> Condition .
                                                  |                              .
                                                  |                                . -(stop)
                                                  |
                                                   -> CreateModel-> ModelBiasCheck/ModelExplainabilityCheck
                                                           |
                                                           |
                                                            -> BatchTransform -> ModelQualityCheck

Implements a get_pipeline(**kwargs) method.
"""
import os
import sys
import boto3
import sagemaker
import sagemaker.session
from datetime import datetime
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput, CreateModelInput, TransformInput
from sagemaker.model import Model
from sagemaker.transformer import Transformer

from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
    FileSource
)
from sagemaker.drift_check_baselines import DriftCheckBaselines
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.workflow.functions import Join
from sagemaker.workflow.fail_step import FailStep
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
)
from sagemaker.workflow.functions import (
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterBoolean,
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
    CreateModelStep,
    TransformStep
)
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.clarify_check_step import (
    DataBiasCheckConfig,
    ClarifyCheckStep,
    ModelBiasCheckConfig,
    ModelPredictedLabelConfig,
    ModelExplainabilityCheckConfig,
    SHAPConfig
)
from sagemaker.workflow.quality_check_step import (
    DataQualityCheckConfig,
    ModelQualityCheckConfig,
    QualityCheckStep,
)
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join
from sagemaker.model_monitor import DatasetFormat, model_monitoring
from sagemaker.clarify import (
    BiasConfig,
    DataConfig,
    ModelConfig
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession


BASE_DIR = os.path.dirname(os.path.realpath(__file__))
# ----------------- #
# --- Constants --- #
# ----------------- #

# Instances
PROCESSING_INSTANCE = "ml.m5.large"
TRAINING_INSTANCE = "ml.m5.large"
TRANSFORM_INSTANCES = ["ml.m5.large"]
INFERENCE_INSTANCES = ["ml.m5.large", "ml.m5.xlarge"]

# S3 Bucket where the data is stored
BUCKET_NAME = "sagemaker-us-west-2-494881137120"
BUCKET = f's3://{BUCKET_NAME}'

# Raw data paths
RAW_DATA_FOLDER = 'windturbine'
RAW_DATA_FILE = 'data.csv'
RAW_DATA_PATH = os.path.join(BUCKET, RAW_DATA_FOLDER, RAW_DATA_FILE)

# Path where the processed objects will be stored
now = datetime.now() # get current time to ensure uniqueness of the output folders
PROCESSED_DATA_FOLDER = 'processed_' + now.strftime("%Y-%m-%d_%H%M_%S%f")
PROCESSED_DATA_PATH = os.path.join(BUCKET, PROCESSED_DATA_FOLDER)

# Paths for model train, validation, test split
TRAIN_DATA_PATH = os.path.join(PROCESSED_DATA_PATH, 'train.csv')
TRAIN_DATA_PATH_W_HEADER = os.path.join(PROCESSED_DATA_PATH, 'train_w_header.csv')
VALIDATION_DATA_PATH = os.path.join(PROCESSED_DATA_PATH, 'validation.csv')
TEST_DATA_PATH = os.path.join(PROCESSED_DATA_PATH, 'test.csv')
TEST_DATA_PATH_W_HEADER = os.path.join(PROCESSED_DATA_PATH, 'test_w_header.csv')


# Model package group name
MODEL_PACKAGE_GROUP_NAME = "WindTurbinePackageGroup"
# Pipeline name
PIPELINE_NAME = "WindTurbinePipeline"
# Job prefix, will be used to identify our project
BASE_JOB_PREFIX = 'WindTurbineError'
# Preprocessor job name
PREPROCESSOR_JOB_NAME = 'SKLearn-Preprocessor'
# Name for the processing step
PROCESSING_STEP_NAME = f'{BASE_JOB_PREFIX}Preprocessing'
# Training job name
TRAINING_JOB_NAME = 'XGBoost-training'
# Name for the training step
TRAINING_STEP_NAME = f'{BASE_JOB_PREFIX}Training'
# Training job name
EVALUATION_JOB_NAME = 'XGBoost-evaluation'
# Name for the evaluation step
EVALUATION_STEP_NAME = f'{BASE_JOB_PREFIX}Evaluation'
# Name for the model register step
REGISTER_MODEL_STEP_NAME = f'{BASE_JOB_PREFIX}RegisterModel'
# Name for the condition step
CONDITON_STEP_NAME = f'{BASE_JOB_PREFIX}AccuracyCondition'
# Name for the fail ste
FAIL_STEP_NAME = f'{BASE_JOB_PREFIX}Fail'
# Accuracy value that the model need to reach as minimum
ACCURACY_CONDITION  = 6.0
MSE_VALUE = 0.05

def get_sagemaker_client(region):
     """Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
        """
     boto_session = boto3.Session(region_name=region)
     sagemaker_client = boto_session.client("sagemaker")
     return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )


def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_name=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.describe_project(ProjectName=sagemaker_project_name)
        sagemaker_project_arn = response["ProjectArn"]
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
    region,
    role=None,
    default_bucket=None,
    sagemaker_project_name=None,
    model_package_group_name=MODEL_PACKAGE_GROUP_NAME,
    pipeline_name=PIPELINE_NAME,
    base_job_prefix=BASE_JOB_PREFIX,
    processing_instance_type=PROCESSING_INSTANCE,
    training_instance_type=TRAINING_INSTANCE,
    transform_instances = TRANSFORM_INSTANCES,
    inference_instances = INFERENCE_INSTANCES,
    preprocessor_job_name: str = PREPROCESSOR_JOB_NAME,
    processing_step_name: str = PROCESSING_STEP_NAME,
    training_job_name: str = TRAINING_JOB_NAME,
    training_step_name: str = TRAINING_STEP_NAME,
    evaluation_job_name: str = EVALUATION_JOB_NAME,
    evaluation_step_name: str = EVALUATION_STEP_NAME,
    register_model_step_name: str = REGISTER_MODEL_STEP_NAME,
    condition_step_name: str = CONDITON_STEP_NAME,
    fail_step_name: str = FAIL_STEP_NAME,
    accuracy_value: float = ACCURACY_CONDITION
):
    """Gets a SageMaker ML Pipeline instance working with on windturbine data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts
        sagemaker_project_name: Name of the project.
    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    #default_bucket = sagemaker_session.default_bucket()
    #default_bucket = ''
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    # Print the role for debugging
    print(f"SageMaker assumes role: {role}.")

    pipeline_session = get_pipeline_session(region, default_bucket)
    
    # parameters for pipeline execution
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    input_data = ParameterString(
        name="InputDataUrl",
        default_value=RAW_DATA_PATH,
    )

    batch_data = ParameterString(
        name="BatchData",
        default_value=''
    )
    # for data quality check step
    skip_check_data_quality = ParameterBoolean(name="SkipDataQualityCheck", default_value=True)
    register_new_baseline_data_quality = ParameterBoolean(name="RegisterNewDataQualityBaseline", default_value=True)
    supplied_baseline_statistics_data_quality = ParameterString(name="DataQualitySuppliedStatistics", default_value='')
    supplied_baseline_constraints_data_quality = ParameterString(name="DataQualitySuppliedConstraints", default_value='')
    baseline_used_For_Drift_CheckConstraints  = ParameterString(name="BaselineUsedForDriftCheckConstraints", default_value='')
    baseline_used_For_Drift_CheckStatistics = ParameterString(name="BaselineUsedForDriftCheckStatistics", default_value='')
    
    # for data bias check step
    skip_check_data_bias = ParameterBoolean(name="SkipDataBiasCheck", default_value = True)
    register_new_baseline_data_bias = ParameterBoolean(name="RegisterNewDataBiasBaseline", default_value=True)
    supplied_baseline_constraints_data_bias = ParameterString(name="DataBiasSuppliedBaselineConstraints", default_value='')

    # for model quality check step
    skip_check_model_quality = ParameterBoolean(name="SkipModelQualityCheck", default_value = True)
    register_new_baseline_model_quality = ParameterBoolean(name="RegisterNewModelQualityBaseline", default_value=True)
    supplied_baseline_statistics_model_quality = ParameterString(name="ModelQualitySuppliedStatistics", default_value='')
    supplied_baseline_constraints_model_quality = ParameterString(name="ModelQualitySuppliedConstraints", default_value='')

    # for model bias check step
    skip_check_model_bias = ParameterBoolean(name="SkipModelBiasCheck", default_value=True)
    register_new_baseline_model_bias = ParameterBoolean(name="RegisterNewModelBiasBaseline", default_value=True)
    supplied_baseline_constraints_model_bias = ParameterString(name="ModelBiasSuppliedBaselineConstraints", default_value='')

    # for model explainability check step
    skip_check_model_explainability = ParameterBoolean(name="SkipModelExplainabilityCheck", default_value=True)
    register_new_baseline_model_explainability = ParameterBoolean(name="RegisterNewModelExplainabilityBaseline", default_value=True)
    supplied_baseline_constraints_model_explainability = ParameterString(name="ModelExplainabilitySuppliedBaselineConstraints", default_value='')

    # processing step for feature engineering
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}-{preprocessor_job_name}",
        sagemaker_session=pipeline_session,
        role=role,
    )

    step_args = sklearn_processor.run(
        inputs=[ProcessingInput(
                        source=RAW_DATA_PATH,
                        destination='/opt/ml/processing/input')],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        arguments=["--input-data", input_data],
    )

    step_process = ProcessingStep(
        name=processing_step_name,
        step_args=step_args,
    )

    ### Calculating the Data Quality

    # `CheckJobConfig` is a helper function that's used to define the job configurations used by the `QualityCheckStep`.
    # By separating the job configuration from the step parameters, the same `CheckJobConfig` can be used across multiple
    # steps for quality checks.
    # The `DataQualityCheckConfig` is used to define the Quality Check job by specifying the dataset used to calculate
    # the baseline, in this case, the training dataset from the data processing step, the dataset format, in this case,
    # a csv file with no headers, and the output path for the results of the data quality check.

    check_job_config = CheckJobConfig(
        role=role,
        instance_count=1,
        instance_type="ml.c5.xlarge",
        volume_size_in_gb=120,
        sagemaker_session=pipeline_session,
    )

    data_quality_check_config = DataQualityCheckConfig(
        baseline_dataset=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
        dataset_format=DatasetFormat.csv(header=False, output_columns_position="START"),
        output_s3_uri=Join(on='/', values=['s3:/', default_bucket, base_job_prefix, ExecutionVariables.PIPELINE_EXECUTION_ID, 'dataqualitycheckstep'])
    )

    data_quality_check_step = QualityCheckStep(
        name="DataQualityCheckStep",
        skip_check=skip_check_data_quality,
        register_new_baseline=register_new_baseline_data_quality,
        quality_check_config=data_quality_check_config,
        check_job_config=check_job_config,
        supplied_baseline_statistics=supplied_baseline_statistics_data_quality,
        supplied_baseline_constraints=supplied_baseline_constraints_data_quality,
        model_package_group_name=model_package_group_name
    )


    #### Calculating the Data Bias

    # The job configuration from the previous step is used here and the `DataConfig` class is used to define how
    # the `ClarifyCheckStep` should compute the data bias. The training dataset is used again for the bias evaluation,
    # the column representing the label is specified through the `label` parameter, and a `BiasConfig` is provided.

    # In the `BiasConfig`, we specify a facet name (the column that is the focal point of the bias calculation),
    # the value of the facet that determines the range of values it can hold, and the threshold value for the label.
    # More details on `BiasConfig` can be found at
    # https://sagemaker.readthedocs.io/en/stable/api/training/processing.html#sagemaker.clarify.BiasConfig

    data_bias_analysis_cfg_output_path = f"s3://{default_bucket}/{base_job_prefix}/databiascheckstep/analysis_cfg"

    data_bias_data_config = DataConfig(
        s3_data_input_path=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
        s3_output_path=Join(on='/', values=['s3:/', default_bucket, base_job_prefix, ExecutionVariables.PIPELINE_EXECUTION_ID, 'databiascheckstep']),
        label=0,
        dataset_type="text/csv",
        s3_analysis_config_output_path=data_bias_analysis_cfg_output_path,
    )

    # We are using this bias config to configure clarify to detect bias based on the first feature in the featurized vector for Sex
    data_bias_config = BiasConfig(
        label_values_or_threshold=[15.0], facet_name=[8], facet_values_or_threshold=[[0.5]]
    )

    data_bias_check_config = DataBiasCheckConfig(
        data_config=data_bias_data_config,
        data_bias_config=data_bias_config,
    )

    data_bias_check_step = ClarifyCheckStep(
        name="DataBiasCheckStep",
        clarify_check_config=data_bias_check_config,
        check_job_config=check_job_config,
        skip_check=skip_check_data_bias,
        register_new_baseline=register_new_baseline_data_bias,
        model_package_group_name=model_package_group_name
    )

    #Define a Training step
    #model path where you want to save the models from training
    model_path = f"s3://{default_bucket}/{base_job_prefix}/WindturbineTrain"

    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.0-1",
        py_version="py3",
        instance_type=training_instance_type,
    )

    xgb_train = Estimator(
        image_uri=image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        output_path=model_path,
        base_job_name=f"{base_job_prefix}-{training_job_name}",
        sagemaker_session=pipeline_session,
        role=role,
    )

    xgb_train.set_hyperparameters(
        objective="reg:squarederror",
        num_round=50,
        max_depth=5,
        eta=0.2,
        gamma=4,
        min_child_weight=6,
        subsample=0.7,
        silent=0,
    )

    step_args = xgb_train.fit(
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
    )
    #Create a TrainingStep using the estimator instance and properties of the ProcessingStep
    step_train = TrainingStep(
        name=training_step_name,
        step_args=step_args,
        depends_on=["DataQualityCheckStep", "DataBiasCheckStep"],
    )

    #Create an instance of a ScriptProcessor that is used to define a processing step for model evaluation
    #The result of this model evaluation is used in the condition step to determine which execute path to take.
    script_eval = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}-{evaluation_job_name}",
        sagemaker_session=pipeline_session,
        role=role,
    )
    step_args = script_eval.run(
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", 
                             source="/opt/ml/processing/evaluation",
                             destination=PROCESSED_DATA_PATH),
        ],
        code=os.path.join(BASE_DIR, "evaluate.py"),
    )
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    step_eval = ProcessingStep(
        name=evaluation_step_name,
        step_args=step_args,
        property_files=[evaluation_report],
    )

    #create a SageMaker model from the output of the training step. This model is used for batch transformation on a new dataset. 
    model = Model(
        image_uri=image_uri,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=pipeline_session,
        role=role,
    )

    step_args = model.create(
        instance_type="ml.m5.large",
        accelerator_type="ml.eia1.medium",
    )

    step_create_model = ModelStep(
        name="WindTurbineModel",
        step_args=step_args,
    )

    transformer = Transformer(
        model_name=step_create_model.properties.ModelName,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        accept="text/csv",
        assemble_with="Line",
        output_path=f"s3://{default_bucket}/WindTurbineTransform",
        sagemaker_session=pipeline_session,
    )

    # The output of the transform step combines the prediction and the input label.
    # The output format is `prediction, original label`

    transform_inputs = TransformInput(
        data=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
    )

    step_args = transformer.transform(
        data=transform_inputs.data,
        input_filter="$[1:]",
        join_source="Input",
        output_filter="$[0,-1]",
        content_type="text/csv",
        split_type="Line",
    )

    step_transform = TransformStep(
        name="WindTurbineTransform",
        step_args=step_args,
    )

    ### Check the Model Quality

    # In this `QualityCheckStep` we calculate the baselines for statistics and constraints using the
    # predictions that the model generates from the test dataset (output from the TransformStep). We define
    # the problem type as 'Regression' in the `ModelQualityCheckConfig` along with specifying the columns
    # which represent the input and output. Since the dataset has no headers, `_c0`, `_c1` are auto-generated
    # header names that should be used in the `ModelQualityCheckConfig`.

    model_quality_check_config = ModelQualityCheckConfig(
        baseline_dataset=step_transform.properties.TransformOutput.S3OutputPath,
        dataset_format=DatasetFormat.csv(header=False),
        output_s3_uri=Join(on='/', values=['s3:/', default_bucket, base_job_prefix, ExecutionVariables.PIPELINE_EXECUTION_ID, 'modelqualitycheckstep']),
        problem_type='Regression',
        inference_attribute='_c0',
        ground_truth_attribute='_c1'
    )

    model_quality_check_step = QualityCheckStep(
        name="ModelQualityCheckStep",
        skip_check=skip_check_model_quality,
        register_new_baseline=register_new_baseline_model_quality,
        quality_check_config=model_quality_check_config,
        check_job_config=check_job_config,
        supplied_baseline_statistics=supplied_baseline_statistics_model_quality,
        supplied_baseline_constraints=supplied_baseline_constraints_model_quality,
        model_package_group_name=model_package_group_name
    )

    ### Check for Model Bias

    # Similar to the Data Bias check step, a `BiasConfig` is defined and Clarify is used to calculate
    # the model bias using the training dataset and the model.


    model_bias_analysis_cfg_output_path = f"s3://{default_bucket}/{base_job_prefix}/modelbiascheckstep/analysis_cfg"

    model_bias_data_config = DataConfig(
        s3_data_input_path=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
        s3_output_path=Join(on='/', values=['s3:/', default_bucket, base_job_prefix, ExecutionVariables.PIPELINE_EXECUTION_ID, 'modelbiascheckstep']),
        s3_analysis_config_output_path=model_bias_analysis_cfg_output_path,
        label=0,
        dataset_type="text/csv",
    )

    model_config = ModelConfig(
        model_name=step_create_model.properties.ModelName,
        instance_count=1,
        instance_type='ml.m5.large',
    )

    # We are using this bias config to configure clarify to detect bias based on the first feature in the featurized vector for Sex
    model_bias_config = BiasConfig(
        label_values_or_threshold=[15.0], facet_name=[8], facet_values_or_threshold=[[0.5]]
    )

    model_bias_check_config = ModelBiasCheckConfig(
        data_config=model_bias_data_config,
        data_bias_config=model_bias_config,
        model_config=model_config,
        model_predicted_label_config=ModelPredictedLabelConfig()
    )

    model_bias_check_step = ClarifyCheckStep(
        name="ModelBiasCheckStep",
        clarify_check_config=model_bias_check_config,
        check_job_config=check_job_config,
        skip_check=skip_check_model_bias,
        register_new_baseline=register_new_baseline_model_bias,
        supplied_baseline_constraints=supplied_baseline_constraints_model_bias,
        model_package_group_name=model_package_group_name
    )

    ### Check Model Explainability

    # SageMaker Clarify uses a model-agnostic feature attribution approach, which you can used to understand
    # why a model made a prediction after training and to provide per-instance explanation during inference. The implementation
    # includes a scalable and efficient implementation of SHAP, based on the concept of a Shapley value from the field of
    # cooperative game theory that assigns each feature an importance value for a particular prediction.

    # For Model Explainability, Clarify requires an explainability configuration to be provided. In this example, we
    # use `SHAPConfig`. For more information of `explainability_config`, visit the Clarify documentation at
    # https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-model-explainability.html.

    model_explainability_analysis_cfg_output_path = "s3://{}/{}/{}/{}".format(
        default_bucket,
        base_job_prefix,
        "modelexplainabilitycheckstep",
        "analysis_cfg"
    )

    model_explainability_data_config = DataConfig(
        s3_data_input_path=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
        s3_output_path=Join(on='/', values=['s3:/', default_bucket, base_job_prefix, ExecutionVariables.PIPELINE_EXECUTION_ID, 'modelexplainabilitycheckstep']),
        s3_analysis_config_output_path=model_explainability_analysis_cfg_output_path,
        label=0,
        dataset_type="text/csv",
    )
    shap_config = SHAPConfig(
        seed=123,
        num_samples=10
    )
    model_explainability_check_config = ModelExplainabilityCheckConfig(
        data_config=model_explainability_data_config,
        model_config=model_config,
        explainability_config=shap_config,
    )
    model_explainability_check_step = ClarifyCheckStep(
        name="ModelExplainabilityCheckStep",
        clarify_check_config=model_explainability_check_config,
        check_job_config=check_job_config,
        skip_check=skip_check_model_explainability,
        register_new_baseline=register_new_baseline_model_explainability,
        supplied_baseline_constraints=supplied_baseline_constraints_model_explainability,
        model_package_group_name=model_package_group_name
    )



    model_metrics = ModelMetrics(
        model_data_statistics=MetricsSource(
            #s3_uri=data_quality_check_step.properties.CalculatedBaselineStatistics,
            s3_uri= 's3://sagemaker-project-p-vf8ny7tgrt9v',
            content_type="application/json",
        ),
        model_data_constraints=MetricsSource(
            #s3_uri=data_quality_check_step.properties.CalculatedBaselineConstraints,
            s3_uri= 's3://sagemaker-project-p-vf8ny7tgrt9v',
            content_type="application/json",
        ),
        bias_pre_training=MetricsSource(
            #s3_uri=data_bias_check_step.properties.CalculatedBaselineConstraints,
            s3_uri= 's3://sagemaker-project-p-vf8ny7tgrt9v',
            content_type="application/json",
        ),
        model_statistics=MetricsSource(
            #s3_uri=model_quality_check_step.properties.CalculatedBaselineStatistics,
            s3_uri= 's3://sagemaker-project-p-vf8ny7tgrt9v',
            content_type="application/json",
        ),
        model_constraints=MetricsSource(
            #s3_uri=model_quality_check_step.properties.CalculatedBaselineConstraints,
            s3_uri= 's3://sagemaker-project-p-vf8ny7tgrt9v',
            content_type="application/json",
        ),
        bias_post_training=MetricsSource(
            #s3_uri=model_bias_check_step.properties.CalculatedBaselineConstraints,
            s3_uri= 's3://sagemaker-project-p-vf8ny7tgrt9v',
            content_type="application/json",
        ),
        bias=MetricsSource(
            # This field can also be set as the merged bias report
            # with both pre-training and post-training bias metrics
            #s3_uri=model_bias_check_step.properties.CalculatedBaselineConstraints,
            s3_uri= 's3://sagemaker-project-p-vf8ny7tgrt9v',
            content_type="application/json",
        ),
        explainability=MetricsSource(
            #s3_uri=model_explainability_check_step.properties.CalculatedBaselineConstraints,
            s3_uri= 's3://sagemaker-project-p-vf8ny7tgrt9v',
            content_type="application/json",
        )
    )

    drift_check_baselines = DriftCheckBaselines(
        model_data_statistics=MetricsSource(
            #s3_uri=data_quality_check_step.properties.BaselineUsedForDriftCheckStatistics,
            s3_uri= 's3://sagemaker-project-p-vf8ny7tgrt9v',
            content_type="application/json",
        ),
        model_data_constraints=MetricsSource(
            #s3_uri=data_quality_check_step.properties.BaselineUsedForDriftCheckConstraints,
            s3_uri= 's3://sagemaker-project-p-vf8ny7tgrt9v',
            content_type="application/json",
        ),
        bias_pre_training_constraints=MetricsSource(
            #s3_uri=data_bias_check_step.properties.BaselineUsedForDriftCheckConstraints,
            s3_uri= 's3://sagemaker-project-p-vf8ny7tgrt9v',
            content_type="application/json",
        ),
        bias_config_file=FileSource(
            #s3_uri=model_bias_check_config.monitoring_analysis_config_uri,
            s3_uri= 's3://sagemaker-project-p-vf8ny7tgrt9v',
            content_type="application/json",
        ),
        model_statistics=MetricsSource(
            #s3_uri=model_quality_check_step.properties.BaselineUsedForDriftCheckStatistics,
            s3_uri= 's3://sagemaker-project-p-vf8ny7tgrt9v',
            content_type="application/json",
        ),
        model_constraints=MetricsSource(
            #s3_uri=model_quality_check_step.properties.BaselineUsedForDriftCheckConstraints,
            s3_uri= 's3://sagemaker-project-p-vf8ny7tgrt9v',
            content_type="application/json",
        ),
        bias_post_training_constraints=MetricsSource(
            #s3_uri=model_bias_check_step.properties.BaselineUsedForDriftCheckConstraints,
            s3_uri= 's3://sagemaker-project-p-vf8ny7tgrt9v',
            content_type="application/json",
        ),
        explainability_constraints=MetricsSource(
            #s3_uri=model_explainability_check_step.properties.BaselineUsedForDriftCheckConstraints,
            s3_uri= 's3://sagemaker-project-p-vf8ny7tgrt9v',
            content_type="application/json",
        ),
        explainability_config_file=FileSource(
            #s3_uri=model_explainability_check_config.monitoring_analysis_config_uri,
            s3_uri= 's3://sagemaker-project-p-vf8ny7tgrt9v',
            content_type="application/json",
        )
    )

    ### Register the model

    # The two parameters in `RegisterModel` that hold the metrics calculated by the `ClarifyCheckStep` and
    # `QualityCheckStep` are `model_metrics` and `drift_check_baselines`.

    # `drift_check_baselines` - these are the baseline files that will be used for drift checks in
    # `QualityCheckStep` or `ClarifyCheckStep` and model monitoring jobs that are set up on endpoints hosting this model.
    # `model_metrics` - these should be the latest baslines calculated in the pipeline run. This can be set
    # using the step property `CalculatedBaseline`

    # The intention behind these parameters is to give users a way to configure the baselines associated with
    # a model so they can be used in drift checks or model monitoring jobs. Each time a pipeline is executed, users can
    # choose to update the `drift_check_baselines` with newly calculated baselines. The `model_metrics` can be used to
    # register the newly calculated baslines or any other metrics associated with the model.

    # Every time a baseline is calculated, it is not necessary that the baselines used for drift checks are updated to
    # the newly calculated baselines. In some cases, users may retain an older version of the baseline file to be used
    # for drift checks and not register new baselines that are calculated in the Pipeline run.

    model = Model(
        image_uri=image_uri,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=pipeline_session,
        role=role,
    )

    step_args = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=inference_instances,
        transform_instances=inference_instances,
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
        drift_check_baselines=drift_check_baselines,
    )

    step_register = ModelStep(
        name=register_model_step_name,
        step_args=step_args,
    )

    # condition step for evaluating model quality and branching execution
    cond_lte = ConditionLessThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="regression_metrics.mse.value"
        ),
        right=MSE_VALUE,
    )

    step_fail = FailStep(
        name=fail_step_name,
        error_message=Join(on=" ", values=["Execution failed due to MSE value > ", MSE_VALUE]),
    )
    step_cond = ConditionStep(
        name="CheckMSEWindTurbineEvaluation",
        conditions=[cond_lte],
        if_steps=[step_register],
        else_steps=[step_fail],
    )

    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            model_approval_status,
            input_data,
            skip_check_data_quality,
            register_new_baseline_data_quality,
            supplied_baseline_statistics_data_quality,
            supplied_baseline_constraints_data_quality,
            skip_check_data_bias,
            register_new_baseline_data_bias,
            supplied_baseline_constraints_data_bias,
            skip_check_model_quality,
            register_new_baseline_model_quality,
            supplied_baseline_statistics_model_quality,
            supplied_baseline_constraints_model_quality,
            skip_check_model_bias,
            register_new_baseline_model_bias,
            supplied_baseline_constraints_model_bias,
            skip_check_model_explainability,
            register_new_baseline_model_explainability,
            supplied_baseline_constraints_model_explainability
        ],
        #steps=[step_process, step_train, step_eval, step_cond],
#        steps=[step_process, data_quality_check_step, data_bias_check_step, step_train, step_eval, step_cond],
        steps=[step_process, data_quality_check_step, data_bias_check_step, step_train, step_create_model, step_transform, model_quality_check_step, model_bias_check_step, model_explainability_check_step, step_eval, step_cond],
        sagemaker_session=pipeline_session,
    )
    return pipeline
