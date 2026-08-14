{ inputs, ... }:

{
  perSystem =
    {
      config,
      lib,
      system,
      ...
    }:
    {
      devShells =
        let
          pkgs = import inputs.nixpkgs { inherit system; };
          stdenv = pkgs.stdenv;
          scripts = config.packages.python-scripts;
          generated = lib.pipe (config.packages) [
            (lib.concatMapAttrs (
              name: package: {
                ${name} = pkgs.mkShell {
                  name = "${name}";
                  inputsFrom = [ package ];
                  shellHook = ''
                    echo "Entering ${name} devShell"
                  '';
                };
                "${name}-extra" =
                  if (name == "python-scripts") then
                    null
                  else
                    pkgs.mkShell {
                      name = "${name}-extra";
                      inputsFrom = [
                        package
                        scripts
                      ];
                      # Extra packages that *may* be used by some scripts
                      packages = [
                        pkgs.python3Packages.tiktoken
                      ];
                      shellHook = ''
                        echo "Entering ${name} devShell"
                        addToSearchPath "LD_LIBRARY_PATH" "${lib.getLib stdenv.cc.cc}/lib"
                      '';
                    };
              }
            ))
            (lib.filterAttrs (name: value: value != null))
          ];
        in
        generated
        # The FATE MoE-cache code (llama-fate, ggml-cuda) only builds with
        # GGML_CUDA=ON, so the default devShell should provide the CUDA
        # toolchain (nvcc, cudart, cublas) via the CUDA-enabled package.
        // lib.optionalAttrs pkgs.stdenv.isLinux {
          default = pkgs.mkShell {
            name = "default";
            inputsFrom = [ config.packages.cuda ];
            shellHook = ''
              echo "Entering llama.cpp CUDA devShell"
              # CMake links against the CUDA toolkit's stub libcuda
              # (cuda_cudart/lib/stubs), which the loader picks up from the
              # binary RUNPATH at runtime and fails with "CUDA driver is a
              # stub library". Prepend the real driver lib directory so the
              # actual libcuda.so.1 is loaded first.
              if [ -d /run/opengl-driver/lib ]; then
                export LD_LIBRARY_PATH="/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
              fi
            '';
          };
        };
    };
}
