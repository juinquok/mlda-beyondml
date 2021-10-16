import logging
import os
import sys

from awsiot.greengrasscoreipc.model import QOS

# Set all the constants
SCORE_THRESHOLD = 0.7
LABELS = np.empty(0)
TOPIC = ""

# Intialize all the variables with default values
CAMERA = None
SCHEDULED_THREAD = None

# Get a logger
logger = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)
