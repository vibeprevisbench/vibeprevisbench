FROM python:3.12-slim-bookworm
ARG BLENDER_VERSION=5.2.2
# sha256 of blender-5.2.2-linux-x64.tar.xz, from https://download.blender.org/release/Blender5.2/blender-5.2.2.sha256
ARG BLENDER_SHA256=84098912789dc450e95697c4184fb8a90acbe5111c2ba4aede3fecb57806a168
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl xz-utils procps ffmpeg libegl-mesa0 libglx-mesa0 libgles2 libx11-6 libxext6 libxrender1 libxi6 libxfixes3 libxkbcommon0 libsm6 libice6 libgl1 libegl1 libglib2.0-0 libgomp1 && rm -rf /var/lib/apt/lists/*
RUN curl --fail --location --retry 3 https://download.blender.org/release/Blender5.2/blender-${BLENDER_VERSION}-linux-x64.tar.xz -o /tmp/blender.tar.xz && echo "${BLENDER_SHA256}  /tmp/blender.tar.xz" | sha256sum -c - && tar -xJf /tmp/blender.tar.xz -C /opt && mv /opt/blender-${BLENDER_VERSION}-linux-x64 /opt/blender && rm /tmp/blender.tar.xz
RUN pip install --no-cache-dir imageio-ffmpeg==0.6.0 && python -c "import imageio_ffmpeg,shutil; shutil.copy2(imageio_ffmpeg.get_ffmpeg_exe(),'/usr/local/bin/ffmpeg')"
ENV LP_NUM_THREADS=2
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV PATH="/opt/blender:${PATH}"
WORKDIR /work/blender_agent
CMD ["sleep", "infinity"]
