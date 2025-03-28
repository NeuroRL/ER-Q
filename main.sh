#!/bin/sh

python -m utils.executor.simulator_nn

echo PCXX ENV_NAME zikken finished!! | sh slack.sh
