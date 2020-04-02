import glslangModule from "https://unpkg.com/@webgpu/glslang@0.0.8/dist/web-devel/glslang.js";

(async () => {
  if (!navigator.gpu) {
    console.log(
      "WebGPU is not supported. Enable chrome://flags/#enable-unsafe-webgpu flag."
    );
    return;
  }

  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();

  const genMatrix = (a, b) => {
    const sum = a * b;
    const rtnArray = new Float32Array(sum + 2);
    rtnArray[0] = a;
    rtnArray[1] = b;
    for (let i = 2; i < sum + 2; i++) {
      rtnArray[i] = i;
    }
    return rtnArray;
  };

  // First Matrix

  //const firstMatrix = new Float32Array([
  //  2 /* rows */,
  //  4 /* columns */,
  //  1,
  //  2,
  //  3,
  //  4,
  //  5,
  //  6,
  //  7,
  //  8
  //]);
  const firstMatrix = genMatrix(256, 256);

  const [gpuBufferFirstMatrix, arrayBufferFirstMatrix] = device.createBufferMapped({
    size: firstMatrix.byteLength,
    usage: GPUBufferUsage.STORAGE
  });
  new Float32Array(arrayBufferFirstMatrix).set(firstMatrix);
  gpuBufferFirstMatrix.unmap();

  // Second Matrix

  //const secondMatrix = new Float32Array([
  //  4 /* rows */,
  //  2 /* columns */,
  //  1,
  //  2,
  //  3,
  //  4,
  //  5,
  //  6,
  //  7,
  //  8
  //]);
  const secondMatrix = genMatrix(256, 256);

  const [gpuBufferSecondMatrix, arrayBufferSecondMatrix] = device.createBufferMapped({
    size: secondMatrix.byteLength,
    usage: GPUBufferUsage.STORAGE
  });
  new Float32Array(arrayBufferSecondMatrix).set(secondMatrix);
  gpuBufferSecondMatrix.unmap();

  // Result Matrix

  const resultMatrixBufferSize =
    Float32Array.BYTES_PER_ELEMENT * (2 + firstMatrix[0] * secondMatrix[1]);
  const resultMatrixBuffer = device.createBuffer({
    size: resultMatrixBufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });

  // Bind group layout and bind group

  const bindGroupLayout = device.createBindGroupLayout({
    bindings: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        type: "readonly-storage-buffer"
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        type: "readonly-storage-buffer"
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        type: "storage-buffer"
      }
    ]
  });

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    bindings: [
      {
        binding: 0,
        resource: {
          buffer: gpuBufferFirstMatrix
        }
      },
      {
        binding: 1,
        resource: {
          buffer: gpuBufferSecondMatrix
        }
      },
      {
        binding: 2,
        resource: {
          buffer: resultMatrixBuffer
        }
      }
    ]
  });

  // Compute shader code (GLSL)

  const computeShaderCode = `#version 450
  layout (local_size_x = 8,
              local_size_y = 16,
              local_size_z = 1) in;

  layout(std430, set = 0, binding = 0) readonly buffer FirstMatrix {
      vec2 size;
      float numbers[];
  } firstMatrix;

  layout(std430, set = 0, binding = 1) readonly buffer SecondMatrix {
      vec2 size;
      float numbers[];
  } secondMatrix;

  layout(std430, set = 0, binding = 2) buffer ResultMatrix {
      vec2 size;
      float numbers[];
  } resultMatrix;

  void main() {
    resultMatrix.size = vec2(firstMatrix.size.x, secondMatrix.size.y);

    ivec2 resultCell = ivec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y);
    float result = 0.0;
    for (int i = 0; i < firstMatrix.size.y; i++) {
      int a = i + resultCell.x * int(firstMatrix.size.y);
      int b = resultCell.y + i * int(secondMatrix.size.y);
      result += firstMatrix.numbers[a] * secondMatrix.numbers[b];
    }

    int index = resultCell.y + resultCell.x * int(secondMatrix.size.y);
    resultMatrix.numbers[index] = result;
  }
  `;

  // Pipeline setup

  const glslang = await glslangModule();

  const computePipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout]
    }),
    computeStage: {
      module: device.createShaderModule({
        code: glslang.compileGLSL(computeShaderCode, "compute")
      }),
      entryPoint: "main"
    }
  });

  // Commands submission
  const run = async (times) => {
    const results = [];
    const single = async () => {
      const commandEncoder = device.createCommandEncoder();

      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(computePipeline);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.dispatch(Math.ceil(firstMatrix[0] / 8) /* x */, Math.ceil(secondMatrix[1] / 16) /* y */);
      passEncoder.endPass();

      // Get a GPU buffer for reading in an unmapped state.
      const gpuReadBuffer = device.createBuffer({
        size: resultMatrixBufferSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
      });

      // Encode commands for copying buffer to buffer.
      commandEncoder.copyBufferToBuffer(
        resultMatrixBuffer /* source buffer */,
        0 /* source offset */,
        gpuReadBuffer /* destination buffer */,
        0 /* destination offset */,
        resultMatrixBufferSize /* size */
      );

      // Submit GPU commands.
      const gpuCommands = commandEncoder.finish();
      device.defaultQueue.submit([gpuCommands]);

      // Read buffer.
      const arrayBuffer = await gpuReadBuffer.mapReadAsync();
      console.log(new Float32Array(arrayBuffer));
    }
    // warm up
    await single();
    for (let i = 0; i < times; i++) {
      const start = performance.now();
      await single();
      const t = performance.now() -start;
      results.push(t);
      console.log(`time cost: ${t}`);
    }
    console.log(results);
    console.log(results.reduce((a, b) => a + b, 0) / times);
  }
  await run(50);
})();
