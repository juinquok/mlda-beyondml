import threading
import time
from os import makedirs, path

import numpy as np
import config_utils
import IPCUtils as ipc_utils
import InferenceEngine from InferenceEngine


def set_configuration(config):
    r"""
    Sets a new config object with the combination of updated and default configuration as applicable.
    Calls inference code with the new config and indicates that the configuration changed.
    """
    new_config = {}

    if "PublishResultsOnTopic" in config:
        config_utils.TOPIC = config["PublishResultsOnTopic"]
    else:
        config_utils.TOPIC = ""
        config_utils.logger.warning(
            "Topic to publish inference results is empty.")

    if "labels" in config:
        config_utils.LABELS = np.array(config["labels"])
    else:
        config_utils.LABELS = np.empty(0)
        config_utils.logger.warning(
            "Labels are using default empty array")

    # Run inference with the updated config indicating the config change.
    run_inference(new_config, True)


def run_inference(new_config, config_changed):
    r"""
    Uses the new config to run inference.

    :param new_config: Updated config if the config changed. Else, the last updated config.
    :param config_changed: Is True when run_inference is called after setting the newly updated config.
    Is False if run_inference is called using scheduled thread as the config hasn't changed.
    """

    if config_changed:
        if config_utils.SCHEDULED_THREAD is not None:
            config_utils.SCHEDULED_THREAD.stop()
        config_changed = False

        inference_engine = InferenceEngine(
            config_utils.LABELS, config_utils.CAMERA)
        config_utils.SCHEDULED_THREAD = inference_engine.startPrediction()

# End of helper methods


# Get intial configuration from the recipe and run inference for the first time.
set_configuration(ipc_utils.IPCUtils().get_configuration())

# Subscribe to the subsequent configuration changes
ipc_utils.IPCUtils().get_config_updates()

# Initial init - start camera thread
config_utils.CAMERA = VideoCapture().startCapture()

# Keeps checking for the updated_config value every 10 minutes.
while True:
    if config_utils.UPDATED_CONFIG:
        set_configuration(ipc_utils.IPCUtils().get_configuration())
        config_utils.UPDATED_CONFIG = False
    time.sleep(600)
