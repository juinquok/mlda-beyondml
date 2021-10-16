# This lambda function updates the overall deployment for the greengrass IOT and also triggers the deployment

import json
import boto3

RPI_IOT_GROUP = "beyondml"
INFERENCE_COMPONENT_NAME = "beyondml-inference"
MODEL_COMPONENT_NAME = "beyondml-model"


def lambda_handler(event, context):
    iot = boto3.client('iot')
    gg2 = boto3.client('greengrassv2')
    pipeline = boto3.client('codepipeline')

    try:

        codepipeline_jobId = event["CodePipeline.job"]['id']

    except Exception as e:
        print("This job is not triggered from a codepipeline job")

    try:

        print("Getting updated component versions")

        all_components = gg2.list_components(maxResults=5)
        for comp in all_components['components']:
            if comp['componentName'] == INFERENCE_COMPONENT_NAME:
                inference_component_version = comp['latestVersion']['componentVersion']
            if comp['componentName'] == MODEL_COMPONENT_NAME:
                model_component_version = comp['latestVersion']['componentVersion']

        grp_response = iot.list_thing_groups(maxResults=100)

        print("Getting thing groups")

        for group in grp_response['thingGroups']:
            if group['groupName'] == RPI_IOT_GROUP:
                target_group_arn = group['groupArn']

        print('Getting all current deployments')

        all_deployments = gg2.list_deployments(
            targetArn=target_group_arn, historyFilter='LATEST_ONLY', maxResults=100)

        deployment_id = all_deployments['deployments'][0]['deploymentId']

        deployment_dict = gg2.get_deployment(deploymentId=deployment_id)

        deployment_components = deployment_dict['components']

        print("Updating component details")

        deployment_components[INFERENCE_COMPONENT_NAME]['componentVersion'] = inference_component_version
        deployment_components[MODEL_COMPONENT_NAME]['componentVersion'] = model_component_version

        deployment_iotJobConfiguration = deployment_dict['iotJobConfiguration']

        deployment_deploymentPolicies = deployment_dict['deploymentPolicies']

        print("Beginning deployment")

        deploy_response = gg2.create_deployment(targetArn=target_group_arn,
                                                deploymentName=deployment_dict['deploymentName'],
                                                components=deployment_components,
                                                iotJobConfiguration=deployment_iotJobConfiguration,
                                                deploymentPolicies=deployment_deploymentPolicies)

        print("Deployment updated")
        print(deploy_response)

        try:

            pipeline.put_job_success_result(jobId=codepipeline_jobId)

        except Exception as e:
            pass

        return {
            'statusCode': 200
        }

    except Exception as e:
        print("Error in updating deployment: {}".format(e))

        pipeline.put_job_failure_result(jobId=codepipeline_jobId, failureDetails={
            'type': 'JobFailed',
            'message': e
        })

        return {
            'statusCode': 500
        }
