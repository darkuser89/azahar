// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version

// Standalone tests use the REAL PICA generator without starting the emulator's logging service.
// Escalate generator errors to test failures instead of accepting a placeholder shader.
// This file is linked only into standalone test executables, never into the Metal library.
#include "common/logging/log.h"
#include <iostream>
#include <stdexcept>

namespace Common::Log {
void FmtLogMessageImpl(Class log_class, Level level, const char*, unsigned, const char* function,
                       fmt::string_view format, const fmt::format_args& arguments) {
    const auto message = fmt::vformat(format, arguments);
    (void)log_class;
    (void)function;
    if (level >= Level::Error) throw std::runtime_error(message);
    std::cerr << message << '\n';
}
void Stop() {}
} // namespace Common::Log
