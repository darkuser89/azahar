// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
#pragma once

#include "mtl_shader_translation.h"
#include <functional>
#include <future>
#include <thread>
#include <list>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace Metal {
enum class TranslationInputKind { GLSL, SPIRV };
struct TranslationInput {
    TranslationInputKind kind;
    ShaderStage stage;
    std::string_view bytes;
    std::span<const ShaderBindingRemap> bindings;
};
struct TranslationInputHash {
    std::size_t operator()(const TranslationInput& input) const {
        auto hash = std::hash<std::string_view>{}(input.bytes);
        const auto add = [&](std::uint32_t word) {
            hash ^= std::size_t(word) + 0x9e3779b9u + (hash << 6) + (hash >> 2);
        };
        add(std::uint32_t(input.kind)); add(std::uint32_t(input.stage));
        for (const auto& binding : input.bindings) {
            add(binding.descriptor_set); add(binding.binding); add(binding.buffer);
            add(binding.texture); add(binding.sampler);
        }
        return hash;
    }
};
struct TranslationCacheStats {
    std::array<std::uint64_t, 2> lookups{}, hits{}, misses{};
    std::uint64_t evictions{}, unretained{}, duplicate_compiles{}, coalesced_compiles{};
    std::size_t entries{}, retained_payload_bytes{};
};
TranslationCacheStats GetTranslationCacheStats();

// Only immutable CPU compiler results. No device, GPU resource, PICA program
// identity or mutable emulated state is shared by this bounded cache.
template <typename Hasher = TranslationInputHash>
class BasicTranslationCache final {
public:
    explicit BasicTranslationCache(std::size_t max_entries_ = 256,
                                   std::size_t max_payload_bytes_ = 32 * 1024 * 1024)
        : max_entries{max_entries_}, max_payload_bytes{max_payload_bytes_} {}

    template <typename Builder>
    TranslatedShader Get(const TranslationInput& input, Builder&& builder) {
        const auto kind = std::size_t(input.kind);
        if (kind >= 2) throw std::invalid_argument("Invalid translation cache input kind");
        const auto hash = hasher(input);
        std::shared_ptr<Flight> flight;
        bool leader = false;
        {
            std::scoped_lock lock{mutex};
            ++stats.lookups[kind];
            if (const auto* found = Find(input, hash)) {
                ++stats.hits[kind];
                return found->shader;
            }
            ++stats.misses[kind];
            // Nested builders never wait for another builder: even a custom
            // cross-key dependency cycle must not deadlock the compiler pool.
            if (!building) for (const auto& active : flights) {
                if (active->hash == hash && Equal(active->key, input)) {
                    flight = active;
                    ++stats.coalesced_compiles;
                    break;
                }
            }
            const auto key_bytes = input.bytes.size() + input.bindings.size_bytes();
            if (!flight && max_entries && flights.size() < 64 &&
                key_bytes <= max_payload_bytes && flight_bytes <= max_payload_bytes - key_bytes) {
                try {
                    flight = std::make_shared<Flight>(Key{input.kind, input.stage, std::string{input.bytes},
                        {input.bindings.begin(), input.bindings.end()}}, hash, key_bytes);
                    flights.push_back(flight);
                    flight_bytes += key_bytes;
                    leader = true;
                } catch (const std::bad_alloc&) { flight.reset(); }
            }
        }
        if (flight && !leader) return flight->future.get();
        const auto retire = [&] {
            if (!flight) return;
            std::scoped_lock lock{mutex};
            std::erase(flights, flight);
            flight_bytes -= flight->bytes;
        };
        struct BuildingScope {
            BuildingScope() { ++building; }
            ~BuildingScope() { --building; }
        } scope;
        try {
            auto result = BuildAndRetain(input, hash, std::forward<Builder>(builder));
            if (flight) flight->promise.set_value(result);
            retire();
            return result;
        } catch (...) {
            if (flight) flight->promise.set_exception(std::current_exception());
            retire();
            throw;
        }
    }

private:
    template <typename Builder>
    TranslatedShader BuildAndRetain(const TranslationInput& input, std::size_t hash, Builder&& builder) {
        // Snapshot the key before the builder; never keep a borrowed source,
        // module or remapping table. Compilation runs without the cache lock,
        // including nested GLSL -> SPIR-V lookups and concurrent first use.
        std::optional<Key> key;
        try {
            if (max_entries && max_payload_bytes)
                key.emplace(Key{input.kind, input.stage, std::string{input.bytes},
                                {input.bindings.begin(), input.bindings.end()}});
        } catch (const std::bad_alloc&) {}
        auto result = builder(); // Compiler errors propagate; failed results are not cached.
        if (!key || Payload(*key, result) > max_payload_bytes) {
            std::scoped_lock lock{mutex}; ++stats.unretained; return result;
        }
        std::optional<Entry> pending;
        try {
            pending.emplace(Entry{std::move(*key), result, hash, 0});
            pending->payload = Payload(pending->key, pending->shader);
        } catch (const std::bad_alloc&) {
            std::scoped_lock lock{mutex}; ++stats.unretained; return result;
        }
        std::scoped_lock lock{mutex};
        const auto owned = pending->key.View();
        if (Find(owned, hash)) { ++stats.duplicate_compiles; return result; }
        if (pending->payload > max_payload_bytes) { ++stats.unretained; return result; }
        while (entries.size() >= max_entries ||
               stats.retained_payload_bytes > max_payload_bytes - pending->payload) {
            auto& oldest = entries.front();
            const auto [begin, end] = index.equal_range(oldest.hash);
            for (auto it = begin; it != end; ++it) {
                if (it->second == &oldest) { index.erase(it); break; }
            }
            stats.retained_payload_bytes -= oldest.payload;
            entries.pop_front(); ++stats.evictions;
        }
        try {
            entries.push_back(std::move(*pending));
            try { index.emplace(hash, &entries.back()); }
            catch (const std::bad_alloc&) { entries.pop_back(); ++stats.unretained; return result; }
        } catch (const std::bad_alloc&) { ++stats.unretained; return result; }
        stats.retained_payload_bytes += entries.back().payload;
        return result;
    }

public:
    TranslationCacheStats Stats() const {
        std::scoped_lock lock{mutex}; auto result = stats; result.entries = entries.size(); return result;
    }

private:
    struct Key {
        TranslationInputKind kind;
        ShaderStage stage;
        std::string bytes;
        std::vector<ShaderBindingRemap> bindings;
        TranslationInput View() const { return {kind, stage, bytes, bindings}; }
    };
    struct Entry { Key key; TranslatedShader shader; std::size_t hash, payload; };
    struct Flight {
        Key key;
        std::size_t hash, bytes;
        std::promise<TranslatedShader> promise;
        std::shared_future<TranslatedShader> future;
        Flight(Key key_, std::size_t hash_, std::size_t bytes_)
            : key{std::move(key_)}, hash{hash_}, bytes{bytes_}, future{promise.get_future().share()} {}
    };
    inline static thread_local unsigned building{};
    std::vector<std::shared_ptr<Flight>> flights;
    std::size_t flight_bytes{};
    static bool Equal(const Key& key, const TranslationInput& input) {
        if (key.kind != input.kind || key.stage != input.stage || key.bytes != input.bytes ||
            key.bindings.size() != input.bindings.size()) return false;
        for (std::size_t n = 0; n < key.bindings.size(); ++n) {
            const auto& a = key.bindings[n]; const auto& b = input.bindings[n];
            if (std::tie(a.descriptor_set, a.binding, a.buffer, a.texture, a.sampler) !=
                std::tie(b.descriptor_set, b.binding, b.buffer, b.texture, b.sampler)) return false;
        }
        return true;
    }
    const Entry* Find(const TranslationInput& input, std::size_t hash) const {
        const auto [begin, end] = index.equal_range(hash);
        for (auto it = begin; it != end; ++it) if (Equal(it->second->key, input)) return it->second;
        return nullptr;
    }
    static std::size_t Payload(const Key& key, const TranslatedShader& shader) {
        auto bytes = sizeof(Entry);
        const auto add = [&](std::size_t amount) {
            bytes = amount > std::numeric_limits<std::size_t>::max() - bytes
                ? std::numeric_limits<std::size_t>::max() : bytes + amount;
        };
        add(key.bytes.capacity()); add(1); add(key.bindings.capacity() * sizeof(ShaderBindingRemap));
        add(shader.source.capacity()); add(1); add(shader.entry_point.capacity()); add(1);
        add(shader.spirv.capacity() * sizeof(std::uint32_t));
        add(shader.resources.capacity() * sizeof(ShaderResource));
        for (const auto& resource : shader.resources) {
            add(resource.name.capacity()); add(1); add(resource.members.capacity() * sizeof(ShaderMemberLayout));
            for (const auto& member : resource.members) { add(member.name.capacity()); add(1); }
        }
        // Container/index nodes are additionally bounded by max_entries.
        return bytes;
    }
    const std::size_t max_entries, max_payload_bytes;
    Hasher hasher;
    mutable std::mutex mutex;
    std::list<Entry> entries;
    std::unordered_multimap<std::size_t, Entry*> index;
    TranslationCacheStats stats;
};
using TranslationCache = BasicTranslationCache<>;
} // namespace Metal
