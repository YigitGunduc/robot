ARG SONIC_TENSORRT_IMAGE=nvcr.io/nvidia/tensorrt:25.08-py3
FROM ${SONIC_TENSORRT_IMAGE}

ARG SONIC_REF=a0732b642c0333077e127a2f56ab0014c196bca4
ENV DEBIAN_FRONTEND=noninteractive \
    APP_ROOT=/workspace/g1-stack \
    SONIC_ROOT=/opt/GR00T-WholeBodyControl \
    SONIC_REF=${SONIC_REF} \
    TensorRT_ROOT=/opt/TensorRT \
    CUDA_HOME=/usr/local/cuda \
    CUDAToolkit_ROOT=/usr/local/cuda \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      git \
      git-lfs \
      iproute2 \
      libegl1 \
      libgl1 \
      libglfw3 \
      libx11-6 \
      lsb-release \
      net-tools \
      procps \
      sudo \
      tini \
      util-linux \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

RUN python -m pip install --no-cache-dir huggingface-hub

# Normalize the NGC TensorRT layout to the path expected by SONIC's setup script.
RUN mkdir -p /opt/TensorRT/include /opt/TensorRT/lib \
    && find /usr -type f -name 'Nv*.h' -exec sh -c \
      'for file do ln -sf "$file" "/opt/TensorRT/include/$(basename "$file")"; done' sh {} + \
    && find /usr -type f \( -name 'libnvinfer*.so*' -o -name 'libnvonnxparser*.so*' \) -exec sh -c \
      'for file do ln -sf "$file" "/opt/TensorRT/lib/$(basename "$file")"; done' sh {} + \
    && test -f /opt/TensorRT/include/NvInfer.h \
    && find /opt/TensorRT/lib -name 'libnvinfer.so*' -print -quit | grep -q .

RUN git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git "$SONIC_ROOT" \
    && git -C "$SONIC_ROOT" checkout "$SONIC_REF" \
    && git -C "$SONIC_ROOT" lfs pull \
    && test "$(git -C "$SONIC_ROOT" rev-parse HEAD)" = "$SONIC_REF"

WORKDIR /opt/GR00T-WholeBodyControl
RUN python download_from_hf.py \
    && find gear_sonic_deploy/policy/release gear_sonic_deploy/planner/target_vel/V2 \
      -type f -exec sha256sum {} + | sort > /opt/sonic-model.sha256

WORKDIR /opt/GR00T-WholeBodyControl/gear_sonic_deploy
RUN chmod +x scripts/install_deps.sh \
    && scripts/install_deps.sh \
    && bash -lc 'source scripts/setup_env.sh && just build'

WORKDIR /opt/GR00T-WholeBodyControl
RUN bash install_scripts/install_mujoco_sim.sh

WORKDIR /workspace/g1-stack
COPY pyproject.toml README.md uv.lock ./
COPY src ./src
RUN python -m pip install --no-cache-dir -e ".[sim]"

COPY . .
RUN ./scripts/fetch_mujoco_assets.sh

COPY docker/sonic/entrypoint.sh /usr/local/bin/sonic-container
COPY docker/sonic/run_stack.sh /usr/local/bin/sonic-stack
RUN chmod +x /usr/local/bin/sonic-container /usr/local/bin/sonic-stack \
    && bash -n /usr/local/bin/sonic-container \
    && bash -n /usr/local/bin/sonic-stack \
    && command -v setsid >/dev/null \
    && g1-stack --help >/dev/null

WORKDIR /workspace/g1-stack
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/sonic-container"]
CMD ["app"]
