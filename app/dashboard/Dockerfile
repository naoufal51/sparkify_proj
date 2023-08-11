# Stage 1: Build the Vue.js app
FROM node:lts-alpine as build-stage

WORKDIR /app

# Copy package*.json files to the working directory
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application code to the working directory
COPY . .

# Build the app for production
RUN npm run build

# Stage 2: Serve the app with Nginx
FROM nginx:stable-alpine as production-stage

# Install bash
RUN apk add --no-cache bash

# Set working directory to where Vue.js app is built
WORKDIR /usr/share/nginx/html


# Copy the built app to the Nginx html directory
COPY --from=build-stage /app/dist .

COPY ./nginx.conf /etc/nginx/nginx.conf 

# copy env file
COPY .env /usr/share/nginx/html/.env

# Add shell script that will generate the env.js file
COPY ./generate-env.sh ./generate-env.sh
RUN chmod +x generate-env.sh


# Expose port 80
EXPOSE 80

# Run generate-env.sh and then start Nginx
CMD ["/bin/bash", "-c", "nginx -g 'daemon off;'"]

