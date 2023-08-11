#!/bin/bash

# Replace the value of VUE_APP_API_BASE_URL with the value of the environment variable
sed -i "s~VITE_API_URL=.*~VITE_API_URL=$VITE_API_URL~" .env

echo "VITE_API_URL is set to: $VITE_API_URL"

# show the content of the .env file
cat .env
