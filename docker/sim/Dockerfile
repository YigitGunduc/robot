FROM python:3.11-slim-bookworm

ARG MUJOCO_MENAGERIE_REF=da76818e269b82289eba39808e2fb91d679d6994
ENV DEBIAN_FRONTEND=noninteractive \
    MUJOCO_GL=osmesa \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      git \
      libgl1 \
      libglfw3 \
      libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app
RUN python -m pip install --no-cache-dir ".[sim]" \
    && mkdir -p /app/third_party \
    && git clone --filter=blob:none --no-checkout \
      https://github.com/google-deepmind/mujoco_menagerie.git \
      /app/third_party/mujoco_menagerie \
    && git -C /app/third_party/mujoco_menagerie sparse-checkout set unitree_g1 \
    && git -C /app/third_party/mujoco_menagerie checkout "$MUJOCO_MENAGERIE_REF"

ENTRYPOINT ["g1-stack"]
CMD ["sim-smoke", "--steps", "1000"]

