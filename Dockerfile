# Image for the Renku app launcher. Built for linux/amd64 by GitHub Actions,
# because image builds are disabled on ephemeral Renku deployments.
#
# The app code lives in /app rather than /home/renku/work: Renku mounts the
# project's data connectors into the work directory, which would shadow
# anything baked in there.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Renku runs session and app containers as uid/gid 1000.
RUN groupadd --gid 1000 renku \
 && useradd --uid 1000 --gid 1000 --create-home --home-dir /home/renku renku

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py mnist_data.py mnist_model.py sdsc_plotly_theme.py train.py ./

RUN mkdir -p /home/renku/work && chown -R 1000:1000 /home/renku /app

USER 1000
WORKDIR /home/renku/work
EXPOSE 8080

CMD ["streamlit", "run", "/app/app.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
