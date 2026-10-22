// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#pragma once

#include <algorithm>
#include <initializer_list>
#include <iterator>
#include <regex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace Metal {

// Each prefix must be a literal beginning of a possible nonempty match.
// The original regex still decides every match, including word boundaries.
class ShaderTextMatcher final {
public:
    ShaderTextMatcher(std::string_view source_, const std::regex& pattern_,
                      std::span<const std::string_view> prefixes)
        : source{source_}, pattern{pattern_} {
        for (const auto prefix : prefixes) {
            if (prefix.empty()) throw std::invalid_argument("Empty shader match prefix");
            candidates.push_back({prefix, source.find(prefix)});
        }
    }

    ShaderTextMatcher(std::string_view source_, const std::regex& pattern_,
                      std::initializer_list<std::string_view> prefixes)
        : ShaderTextMatcher{source_, pattern_, std::span{prefixes.begin(), prefixes.size()}} {}

    bool Next(std::size_t& position, std::cmatch& match) {
        while (true) {
            auto first = std::string_view::npos;
            for (auto& candidate : candidates) {
                if (candidate.position < cursor)
                    candidate.position = source.find(candidate.prefix, cursor);
                first = std::min(first, candidate.position);
            }
            if (first == std::string_view::npos) return false;
            auto flags = std::regex_constants::match_continuous;
            if (first) flags |= std::regex_constants::match_prev_avail;
            if (std::regex_search(source.data() + first, source.data() + source.size(),
                                  match, pattern, flags)) {
                if (!match.length()) throw std::logic_error("Empty shader regex match");
                cursor = first + std::size_t(match.length());
                position = first;
                return true;
            }
            cursor = first + 1;
        }
    }

private:
    struct Candidate { std::string_view prefix; std::size_t position; };
    std::string_view source;
    const std::regex& pattern;
    std::vector<Candidate> candidates;
    std::size_t cursor{};
};

// Shader replacements use literals and match/capture references ($&, $1, ...),
// not regex_replace's unmatched-prefix/suffix format directives.
inline std::string ReplaceShaderText(std::string_view source, const std::regex& pattern,
                                   std::span<const std::string_view> prefixes,
                                   std::string_view replacement) {
    ShaderTextMatcher matcher{source, pattern, prefixes};
    std::string result;
    result.reserve(source.size());
    std::size_t position{}, consumed{};
    std::cmatch match;
    while (matcher.Next(position, match)) {
        result.append(source.substr(consumed, position - consumed));
        match.format(std::back_inserter(result), replacement.data(),
                     replacement.data() + replacement.size());
        consumed = position + std::size_t(match.length());
    }
    result.append(source.substr(consumed));
    return result;
}

inline std::string ReplaceShaderText(std::string_view source, const std::regex& pattern,
                                   std::initializer_list<std::string_view> prefixes,
                                   std::string_view replacement) {
    return ReplaceShaderText(source, pattern, std::span{prefixes.begin(), prefixes.size()}, replacement);
}
} // namespace Metal
