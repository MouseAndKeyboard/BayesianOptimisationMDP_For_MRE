# frontend/Dockerfile

# Use the official Node.js 16 image as the base
FROM node:18-alpine

# Set working directory inside the container
WORKDIR /app

# Install dependencies
# Copy package.json and package-lock.json first for better caching
COPY package.json package-lock.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application code
COPY . .

# Expose port 3000 to allow access to the Next.js app
EXPOSE 3000

# Start the development server with live-reloading
CMD ["npm", "run", "dev"]
