# AML Command Center static dashboard for Railway.
FROM node:22-alpine AS build

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

COPY index.html vite.config.ts tsconfig.json ./
COPY public ./public
COPY src ./src

# Bake the backend WebSocket URL at build time (wss://your-server.up.railway.app).
ARG VITE_AML_WS=ws://127.0.0.1:8765
ENV VITE_AML_WS=$VITE_AML_WS

RUN npm run build

FROM node:22-alpine
WORKDIR /app
RUN npm install -g serve@14
COPY --from=build /app/dist ./dist

ENV NODE_ENV=production
EXPOSE 3000
CMD ["sh", "-c", "serve -s dist -l tcp://0.0.0.0:${PORT:-3000}"]
