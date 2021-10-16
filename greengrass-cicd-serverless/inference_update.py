# This lambda function updates the greengrass IOT inference component with the latest source files and the model version

import json
import boto3
import ast

INFERENCE_COMPONENT_NAME = "beyondml-inference"
MODEL_COMPONENT_NAME = "beyondml-model"

inference_component_arn = None
model_component_version = ""


def lambda_handler(event, context):

    gg2 = boto3.client('greengrassv2')
    s3 = boto3.resource('s3')
    codepipeline = boto3.client('codepipeline')

    codepipeline_jobId = event["CodePipeline.job"]['id']

    try:
        all_components = gg2.list_components(maxResults=5)
        for comp in all_components['components']:
            if comp['componentName'] == INFERENCE_COMPONENT_NAME:
                inference_component_arn = comp['latestVersion']['arn']
            if comp['componentName'] == MODEL_COMPONENT_NAME:
                model_component_version = comp['latestVersion']['componentVersion']

        inference_component_json = gg2.get_component(
            recipeOutputFormat='JSON', arn=inference_component_arn)['recipe'].decode('utf8')
        inf_comp = ast.literal_eval(inference_component_json)

        # increment component version
        inf_comp_version_num = inf_comp['ComponentVersion']
        if inf_comp_version_num[-1] == 9:
            new_version_mid = int(inf_comp_version_num[2]) + 1
            new_comp_version = inf_comp_version_num[0:2] + \
                str(new_version_mid) + ".0"
        else:
            new_comp_version = inf_comp_version_num[0:-
                                                    1] + str(int(inf_comp_version_num[-1]) + 1)

        print("Component before update")
        print(inf_comp)

        inf_comp['ComponentVersion'] = new_comp_version

        # remove exisiting SHA digest
        del inf_comp['Manifests'][0]['Artifacts'][0]['Digest']
        del inf_comp['Manifests'][0]['Artifacts'][0]['Algorithm']

        # set the model store update
        inf_comp['ComponentDependencies'][MODEL_COMPONENT_NAME]['VersionRequirement'] = str(
            "=" + model_component_version)

        print("Component after update")
        print(inf_comp)

        print("Updating component now")
        comp_update = gg2.create_component_version(
            inlineRecipe=json.dumps(inf_comp))
        print(comp_update)

        comp_update['creationTimestamp'] = comp_update['creationTimestamp'].isoformat()

        if len(comp_update['status']['errors']) > 0:

            print("Errors found in component update, sending failure to CodePipeline")

            codepipeline.put_job_failure_result(jobId=codepipeline_jobId, failureDetails={
                'type': 'JobFailed',
                'message': "Error in component update"
            })

            return {
                'statusCode': 505
            }

        else:
            print("Success! Sending success to CodePipeline")

            codepipeline.put_job_success_result(jobId=codepipeline_jobId)
            return {
                'statusCode': 200
            }

    except Exception as e:
        print("Error in updating component: {}".format(e))

        codepipeline.put_job_failure_result(jobId=codepipeline_jobId, failureDetails={
            'type': 'JobFailed',
            'message': e
        })

        return {
            'statusCode': 500
        }
