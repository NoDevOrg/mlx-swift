// Copyright © 2024 Apple Inc.

import Cmlx

#if !(os(macOS) || os(iOS) || os(tvOS) || os(visionOS))

    extension MLXFast {

        /// A custom CUDA kernel compiled with NVRTC, the CUDA counterpart of
        /// ``metalKernel(name:inputNames:outputNames:source:header:ensureRowContiguous:atomicOutputs:)``.
        ///
        /// `source` is the kernel body; MLX generates the `__global__` signature
        /// from `inputNames` (const pointers, plus `<name>_shape`, `_strides`,
        /// `_ndim` when the body uses them) and `outputNames`. `header` goes
        /// before the kernel, outside its namespace. `grid` is in threads, as in
        /// the Metal API: blocks = grid / threadGroup per axis.
        final public class CUDAKernel: @unchecked Sendable {
            let kernel: mlx_fast_cuda_kernel
            public let outputNames: [String]

            init(
                name: String, inputNames: some Sequence<String>, outputNames: some Sequence<String>,
                source: String, header: String = "", ensureRowContiguous: Bool = true,
                sharedMemory: Int = 0
            ) {
                self.outputNames = Array(outputNames)
                let input_names = mlx_vector_string_new()
                defer { mlx_vector_string_free(input_names) }
                for name in inputNames {
                    mlx_vector_string_append_value(input_names, name)
                }
                let output_names = mlx_vector_string_new()
                defer { mlx_vector_string_free(output_names) }
                for name in self.outputNames {
                    mlx_vector_string_append_value(output_names, name)
                }
                self.kernel = mlx_fast_cuda_kernel_new(
                    name.cString(using: .utf8),
                    input_names, output_names,
                    source.cString(using: .utf8),
                    header.cString(using: .utf8),
                    ensureRowContiguous, Int32(sharedMemory))
            }

            deinit {
                mlx_fast_cuda_kernel_free(kernel)
            }

            public func callAsFunction(
                _ inputs: [any ScalarOrArray],
                template: [(String, any KernelTemplateArg)]? = nil,
                grid: (Int, Int, Int),
                threadGroup: (Int, Int, Int),
                outputShapes: some Sequence<[Int]>,
                outputDTypes: some Sequence<DType>,
                initValue: Float? = nil,
                verbose: Bool = false,
                stream: StreamOrDevice = .default
            ) -> [MLXArray] {
                let config = mlx_fast_cuda_kernel_config_new()
                defer { mlx_fast_cuda_kernel_config_free(config) }
                if let template {
                    for (name, arg) in template {
                        switch arg {
                        case let value as Bool:
                            mlx_fast_cuda_kernel_config_add_template_arg_bool(config, name, value)
                        case let value as Int:
                            mlx_fast_cuda_kernel_config_add_template_arg_int(
                                config, name, Int32(value))
                        case let value as DType:
                            mlx_fast_cuda_kernel_config_add_template_arg_dtype(
                                config, name, value.cmlxDtype)
                        default:
                            fatalError(
                                "Unable to handle KernelTemplateArg \(name) with type: \(type(of: arg))."
                            )
                        }
                    }
                }
                mlx_fast_cuda_kernel_config_set_grid(
                    config, grid.0.int32, grid.1.int32, grid.2.int32)
                mlx_fast_cuda_kernel_config_set_thread_group(
                    config, threadGroup.0.int32, threadGroup.1.int32, threadGroup.2.int32)
                for (shape, dtype) in zip(outputShapes, outputDTypes) {
                    mlx_fast_cuda_kernel_config_add_output_arg(
                        config, shape.map { Int32($0) }, shape.count, dtype.cmlxDtype)
                }
                if let initValue {
                    mlx_fast_cuda_kernel_config_set_init_value(config, initValue)
                }
                mlx_fast_cuda_kernel_config_set_verbose(config, verbose)
                let inputs = new_mlx_vector_array(inputs.map { $0.asMLXArray(dtype: nil) })
                defer { mlx_vector_array_free(inputs) }
                var result = mlx_vector_array_new()
                mlx_fast_cuda_kernel_apply(&result, kernel, inputs, config, stream.ctx)
                defer { mlx_vector_array_free(result) }
                return mlx_vector_array_values(result)
            }
        }

        public static func cudaKernel(
            name: String, inputNames: some Sequence<String>, outputNames: some Sequence<String>,
            source: String, header: String = "", ensureRowContiguous: Bool = true,
            sharedMemory: Int = 0
        ) -> CUDAKernel {
            CUDAKernel(
                name: name, inputNames: inputNames, outputNames: outputNames,
                source: source, header: header,
                ensureRowContiguous: ensureRowContiguous, sharedMemory: sharedMemory)
        }
    }  // MLXFast

#endif
