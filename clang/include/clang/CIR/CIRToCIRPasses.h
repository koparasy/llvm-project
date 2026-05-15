//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares an interface for running CIR-to-CIR passes.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_CIRTOCIRPASSES_H
#define CLANG_CIR_CIRTOCIRPASSES_H

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"

#include <memory>

namespace clang {
class ASTContext;
}

namespace llvm::vfs {
class FileSystem;
} // namespace llvm::vfs

namespace mlir {
class MLIRContext;
class ModuleOp;
} // namespace mlir

namespace cir {

class LowerModule;

// CIR-to-CIR pipeline split into two halves so that LTO/Combine flows can
// stop after the pre-lowering phase, ship the result to disk, and resume
// later with a separate cc1 invocation.
//
// Pre-lowering passes are target-agnostic and rely on a live ASTContext for
// fact materialization (cir-materialize-ast-facts) and stub passes
// (cir-idiom-recognizer).  Post-lowering passes are target-bound and
// AST-free, driven by a `cir::LowerModule` built from the surrounding cc1
// invocation.

mlir::LogicalResult
runPreLoweringPasses(mlir::ModuleOp theModule, mlir::MLIRContext &mlirCtx,
                     clang::ASTContext &astCtx, bool enableVerifier,
                     bool enableIdiomRecognizer, bool enableCIRSimplify);

mlir::LogicalResult
runPostLoweringPasses(mlir::ModuleOp theModule, mlir::MLIRContext &mlirCtx,
                      cir::LowerModule &lowerModule,
                      llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> vfs,
                      bool enableVerifier);

// Convenience wrapper: pre + post in one call, used by in-process CIRGen.
mlir::LogicalResult
runCIRToCIRPasses(mlir::ModuleOp theModule, mlir::MLIRContext &mlirCtx,
                  clang::ASTContext &astCtx, cir::LowerModule &lowerModule,
                  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> vfs,
                  bool enableVerifier, bool enableIdiomRecognizer,
                  bool enableCIRSimplify);

} // namespace cir

#endif // CLANG_CIR_CIRTOCIRPASSES_H_
