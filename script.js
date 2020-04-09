import glslangModule from "https://unpkg.com/@webgpu/glslang@0.0.8/dist/web-devel/glslang.js";

var dimAOuter = 256;
var dimInner = 256;
var dimBOuter = 256;
var localSizeX = 8;
var localSizeY = 16;
var workPerThread = [4, 4];
var device;
var bindGroup;
var computePipeline;
var resultMatrixBuffer;
var gpuReadBuffer;
var firstMatrix, secondMatrix;

var bindGroupLayout;
var glslang;
var matmulPackedCode;

async function pre(dimAOuter, dimInner, dimBOuter, localSizeX, localSizeY) {
  dimAOuter = dimAOuter;
  dimInner = dimInner;
  dimBOuter = dimBOuter;
  localSizeX = localSizeX;
  localSizeY = localSizeY;
  if (!navigator.gpu) {
    console.log(
      "WebGPU is not supported. Enable chrome://flags/#enable-unsafe-webgpu flag."
    );
    return;
  }

  const adapter = await navigator.gpu.requestAdapter();
  device = await adapter.requestDevice();

  // Uniform Buffer
  const uniformData = new Int32Array([
    dimAOuter /* A rows */,
    dimInner /* A columns */,
    dimInner /* B rows */,
    dimBOuter /* B columns */
  ]);

  const [uniformBuffer, arrayBufferData] = device.createBufferMapped({
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM
  });
  new Int32Array(arrayBufferData).set(uniformData);
  uniformBuffer.unmap();

  // First Matrix
  firstMatrix = new Float32Array(dimAOuter * dimInner);
  for(var i = 0; i < dimAOuter * dimInner; i++){
    firstMatrix[i] = Math.random();
  }

  const [gpuBufferFirstMatrix, arrayBufferFirstMatrix] = device.createBufferMapped({
    size: firstMatrix.byteLength,
    usage: GPUBufferUsage.STORAGE
  });
  new Float32Array(arrayBufferFirstMatrix).set(firstMatrix);
  gpuBufferFirstMatrix.unmap();

  // Second Matrix
  secondMatrix = new Float32Array(dimInner * dimBOuter);
  for(var i = 0; i < dimInner * dimBOuter; i++){
    secondMatrix[i] = Math.random();
  }

  const [gpuBufferSecondMatrix, arrayBufferSecondMatrix] = device.createBufferMapped({
    size: secondMatrix.byteLength,
    usage: GPUBufferUsage.STORAGE
  });
  new Float32Array(arrayBufferSecondMatrix).set(secondMatrix);
  gpuBufferSecondMatrix.unmap();

  // Result Matrix
  const resultMatrixBufferSize =
    Float32Array.BYTES_PER_ELEMENT * (uniformData[0] * uniformData[3]);
  resultMatrixBuffer = device.createBuffer({
    size: resultMatrixBufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });

  // Bind group layout and bind group

  bindGroupLayout = device.createBindGroupLayout({
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
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        type: "uniform-buffer"
      }
    ]
  });

  bindGroup = device.createBindGroup({
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
      },
      {
        binding: 3,
        resource: {
          buffer: uniformBuffer
        }
      }
    ]
  });

  // Compute shader code (GLSL)

  matmulPackedCode = `#version 450
  layout (local_size_x = ${localSizeX},
    local_size_y = ${localSizeY},
    local_size_z = 1) in;
  layout(std430, set = 0, binding = 0) readonly buffer FirstMatrix {
      float numbers[];
  } firstMatrix;

  layout(std430, set = 0, binding = 1) readonly buffer SecondMatrix {
      float numbers[];
  } secondMatrix;

  layout(std430, set = 0, binding = 2) buffer ResultMatrix {
      float numbers[];
  } resultMatrix;

  layout(std140, set = 0, binding = 3) uniform Uniforms {
    ivec2 aShape; ivec2 bShape;
  };

  int dimAOuter = aShape[0];
  int dimInner = aShape[1];
  int dimBOuter = bShape[1];

  bool coordsInBounds(ivec2 coord, ivec2 shape) {
    return all(greaterThanEqual(coord, ivec2(0))) &&
        all(lessThan(coord, shape));
  }  

  float mm_readA(int row, int col) {
    return coordsInBounds(ivec2(row, col), ivec2(dimAOuter, dimInner)) ?
            firstMatrix.numbers[row * dimInner + col] : 0;
    //return firstMatrix.numbers[row * dimInner + col];
  }

  float mm_readB(int row, int col) {
    return coordsInBounds(ivec2(row, col), ivec2(dimInner, dimBOuter)) ?
            secondMatrix.numbers[row * dimBOuter + col] : 0;
    //return secondMatrix.numbers[row * dimBOuter + col];
  }

  void mm_write(int row, int col, float value) {
    resultMatrix.numbers[row * dimBOuter + col] = value;
  }

  const int RowPerThread = ${workPerThread[1]};
  const int ColPerThread = ${workPerThread[0]};
  const int TileAOuter = int(gl_WorkGroupSize.y) * RowPerThread;
  const int TileBOuter = int(gl_WorkGroupSize.x) * ColPerThread;
  const int TileInner = TileAOuter > TileBOuter ? TileAOuter : TileBOuter;

  shared float mm_Asub[TileAOuter][TileInner];
  shared float mm_Bsub[TileInner][TileBOuter];

  void mm_matMul(int dimAOuter, int dimInner, int dimBOuter) {
    int tileRow = int(gl_LocalInvocationID.y) * RowPerThread;
    int tileCol = int(gl_LocalInvocationID.x) * ColPerThread;

    int globalRow = int(gl_GlobalInvocationID.y) * RowPerThread;
    int globalCol = int(gl_GlobalInvocationID.x) * ColPerThread;

    int numTiles = (dimInner - 1) / TileInner + 1;

    float acc[RowPerThread][ColPerThread];
    float ACached;
    float BCached[ColPerThread];

    // Without this initialization strange values show up in acc.
    for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
      for (int innerCol = 0; innerCol < ColPerThread; innerCol++) {
        acc[innerRow][innerCol] = 0.0;
      }
    }

    const int ColPerThreadA = TileInner / int(gl_WorkGroupSize.x);
    int tileColA = int(gl_LocalInvocationID.x) * ColPerThreadA;
    const int RowPerThreadB = TileInner / int(gl_WorkGroupSize.y);
    int tileRowB = int(gl_LocalInvocationID.y) * RowPerThreadB;

    // Loop over shared dimension.
    for (int t = 0; t < numTiles; t++) {
      // Load one tile of A into local memory.
      for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
        for (int innerCol = 0; innerCol < ColPerThreadA; innerCol++) {
          int inputRow = tileRow + innerRow;
          int inputCol = tileColA + innerCol;

          mm_Asub[inputRow][inputCol] = mm_readA(
              globalRow + innerRow,
              t * TileInner + inputCol);
        }
      }
      // Load one tile of B into local memory.
      for (int innerRow = 0; innerRow < RowPerThreadB; innerRow++) {
        for (int innerCol = 0; innerCol < ColPerThread; innerCol++) {
          int inputRow = tileRowB + innerRow;
          int inputCol = tileCol + innerCol;

          mm_Bsub[inputRow][inputCol] = mm_readB(
            t * TileInner + inputRow,
            globalCol + innerCol);;
        }
      }

      barrier();

      // Compute acc values for a single thread.
      for (int k = 0; k < TileInner; k++) {
        for (int inner = 0; inner < ColPerThread; inner++) {
          BCached[inner] = mm_Bsub[k][tileCol + inner];
        }

        for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
          ACached = mm_Asub[tileRow + innerRow][k];
          for (int innerCol = 0; innerCol < ColPerThread; innerCol++) {
            acc[innerRow][innerCol] += ACached * BCached[innerCol];
          }
        }
      }

      barrier();
    }

    for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
      for (int innerCol = 0; innerCol < ColPerThread; innerCol++) {

        if ((globalCol + innerCol) < dimBOuter &&
            (globalRow + innerRow) < dimAOuter) {
          mm_write(globalRow + innerRow,
                   globalCol + innerCol,
                   acc[innerRow][innerCol]);
        }
      }
    }
  }

    void main() {
      mm_matMul(dimAOuter, dimInner, dimBOuter);
    }
  `;

  const computeShaderNaiveCode = `#version 450
  layout (local_size_x = ${localSizeX},
    local_size_y = ${localSizeY},
    local_size_z = 1) in;
  layout(std430, set = 0, binding = 0) readonly buffer FirstMatrix {
      float numbers[];
  } firstMatrix;

  layout(std430, set = 0, binding = 1) readonly buffer SecondMatrix {
      float numbers[];
  } secondMatrix;

  layout(std430, set = 0, binding = 2) buffer ResultMatrix {
      float numbers[];
  } resultMatrix;

  layout(std140, set = 0, binding = 3) uniform Uniforms {
    ivec2 aShape; ivec2 bShape;
  };

  int dimAOuter = aShape[0];
  int dimInner = aShape[1];
  int dimBOuter = bShape[1];

  float mm_readA(int row, int col) {
    return firstMatrix.numbers[row * dimInner + col];
  }

  float mm_readB(int row, int col) {
    return secondMatrix.numbers[row * dimBOuter + col];
  }

  void mm_write(int row, int col, float value) {
    resultMatrix.numbers[row * dimBOuter + col] = value;
  }

  const int MatTileSize = int(gl_WorkGroupSize.x);  // .x == .y
  shared float mm_Asub[MatTileSize][MatTileSize];
  shared float mm_Bsub[MatTileSize][MatTileSize];

  void mm_matMul(int dimAOuter, int dimInner, int dimBOuter) {
      int localRow = int(gl_LocalInvocationID.y);  // 0..MatTileSize
      int localCol = int(gl_LocalInvocationID.x);  // 0..MatTileSize
      int globalRow = int(gl_GlobalInvocationID.y);  // AOuter
      int globalCol = int(gl_GlobalInvocationID.x);  // Inner

      float acc = 0.0;

      int numTiles = (dimInner - 1) / MatTileSize + 1;

      for (int t = 0; t < numTiles; t++) {
        // Load one tile of A and B into local memory
        int tiledACol = MatTileSize * t + localCol;
        int tiledBRow = MatTileSize * t + localRow;
        mm_Asub[localRow][localCol] = mm_readA(globalRow, tiledACol);
        mm_Bsub[localRow][localCol] = mm_readB(tiledBRow, globalCol);

        // Synchronise to make sure the tile is loaded
        barrier();

        for (int k = 0; k < MatTileSize; k++) {
          acc += mm_Asub[localRow][k] * mm_Bsub[k][localCol];
        }

        // Synchronise before loading the next tile
        barrier();
      }

      if (globalCol < dimBOuter && globalRow < dimAOuter) {
        mm_write(globalRow, globalCol, acc);
      }
    }

    void main() {
      mm_matMul(dimAOuter, dimInner, dimBOuter);
    }
  `;

  // Pipeline setup

  glslang = await glslangModule();

  computePipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout]
    }),
    computeStage: {
      module: device.createShaderModule({
        code: glslang.compileGLSL(matmulPackedCode, "compute")
      }),
      entryPoint: "main"
    }
  });

  // Commands submission

  const commandEncoder = device.createCommandEncoder();

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(computePipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatch(uniformData[3] / (localSizeX * workPerThread[0]) /* x */,
                       uniformData[0] / (localSizeY * workPerThread[1])  /* y */);
  passEncoder.endPass();

  // Get a GPU buffer for reading in an unmapped state.
  gpuReadBuffer = device.createBuffer({
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
  const arrayBuffer = new Float32Array(await gpuReadBuffer.mapReadAsync());
  gpuReadBuffer.unmap();
  let acc = 0, m = Math.floor(dimAOuter*Math.random()),  n = Math.floor(dimBOuter*Math.random())
  for(let k=0; k<dimInner; k++) acc += firstMatrix[m * dimInner + k] * secondMatrix[k * dimBOuter + n];
  console.log(`result[${m}, ${n}] = ${arrayBuffer[m * dimBOuter + n]}, expectedResult = ${acc}`);
  /*
  for (var i = 0; i < dimAOuter; i++)
  for (var j = 0; j< dimBOuter; j++)
  {
    let test = 0;
    for (var k =0; k < dimInner; k++)
    {
      test += firstMatrix[i * dimInner + k] * secondMatrix[k * dimBOuter + j];
    }
    console.log(`result[${i}, ${j}] = ${arrayBuffer[i * dimBOuter + j]}, expectedResult = ${test}`);
  }
  */
};

export async function run(dimAOuter=256, dimInner=256, dimBOuter=256,
    localSizeX=16, localSizeY=16,times=50){
  dimAOuter = dimAOuter;
  dimInner = dimInner;
  dimBOuter = dimBOuter;
  localSizeX = localSizeX;
  localSizeY = localSizeY;
  var arrayBuffer;

  await pre(dimAOuter, dimInner, dimBOuter, localSizeX, localSizeY);

  var ti0 = performance.now()
  for (let i = 0; i<times;i++) {
  const commandEncoder = device.createCommandEncoder();

  const passEncoder = commandEncoder.beginComputePass();
  computePipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout]
    }),
    computeStage: {
      module: device.createShaderModule({
        code: glslang.compileGLSL(matmulPackedCode, "compute")
      }),
      entryPoint: "main"
    }
  });
  passEncoder.setPipeline(computePipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatch(dimBOuter / (localSizeX * workPerThread[0]) /* x */,
                       dimAOuter / (localSizeY * workPerThread[1])  /* y */);
  passEncoder.endPass();

  // Encode commands for copying buffer to buffer.
  commandEncoder.copyBufferToBuffer(
    resultMatrixBuffer /* source buffer */,
    0 /* source offset */,
    gpuReadBuffer /* destination buffer */,
    0 /* destination offset */,
    dimAOuter * dimBOuter /* size */
  );

  // Submit GPU commands.
  const gpuCommands = commandEncoder.finish();
  device.defaultQueue.submit([gpuCommands]);

  // Read buffer.
  arrayBuffer = new Float32Array(await gpuReadBuffer.mapReadAsync());
  gpuReadBuffer.unmap();
  }
  var ti = performance.now()
  let acc = 0, m = Math.floor(dimAOuter*Math.random()),  n = Math.floor(dimBOuter*Math.random())
  for(let k=0; k<dimInner; k++) acc += firstMatrix[m * dimInner + k] * secondMatrix[k * dimBOuter + n];
 // console.log(`result[${m}, ${n}] = ${arrayBuffer[m * dimBOuter + n]}, expectedResult = ${acc}`);

  document.getElementById('output').innerText =
    ` time = ${Math.round(10*(ti - ti0))/10/times}ms
     GFLOPS=${Math.round(2*dimAOuter*dimBOuter*dimInner/(ti - ti0)/10000)/100}
     result[${m}, ${n}] = ${arrayBuffer[m * dimBOuter + n]}, expectedResult = ${acc}`;
}