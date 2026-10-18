# ==============================================================================
# React / Vite Frontend Service (Node.js)
# ==============================================================================

FROM node:20-alpine

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm ci || npm install

# Copy source code and build production assets
COPY . .
RUN npm run build

# Expose Vite preview/dev port
EXPOSE 3000

# Start Vite preview server on port 3000
CMD ["npm", "run", "preview", "--", "--host", "0.0.0.0", "--port", "3000"]
