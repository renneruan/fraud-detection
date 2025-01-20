#!/bin/bash

set -a

if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

set +a